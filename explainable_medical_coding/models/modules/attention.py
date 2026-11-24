import math
from typing import Callable, Optional

import torch
import torch.nn as nn


def _broadcast_token_mask(
    attention_mask: torch.Tensor,
    scores: torch.Tensor,
    token_dim: int,
) -> torch.Tensor:
    """Pad and broadcast a [B, T] mask to an attention score tensor."""
    token_dim = token_dim % scores.ndim
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch_size, sequence_length]")
    if attention_mask.size(0) != scores.size(0):
        raise ValueError("attention_mask batch size does not match attention scores")
    if attention_mask.size(1) > scores.size(token_dim):
        raise ValueError("attention mask is longer than the encoded sequence")

    attention_mask = torch.nn.functional.pad(
        attention_mask,
        (0, scores.size(token_dim) - attention_mask.size(1)),
        value=0,
    ).to(device=scores.device, dtype=torch.bool)
    shape = [1] * scores.ndim
    shape[0] = scores.size(0)
    shape[token_dim] = scores.size(token_dim)
    return attention_mask.view(shape)


def mask_attention_scores(
    scores: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    token_dim: int,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Mask token scores with the finite minimum of their floating dtype."""
    if attention_mask is None:
        return scores, None
    mask = _broadcast_token_mask(attention_mask, scores, token_dim)
    return scores.masked_fill(~mask, torch.finfo(scores.dtype).min), mask


def masked_softmax(
    scores: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    token_dim: int,
) -> torch.Tensor:
    """Softmax over valid tokens without NaNs for fully masked examples."""
    masked_scores, mask = mask_attention_scores(scores, attention_mask, token_dim)
    softmax_dtype = None
    if masked_scores.device.type == "cpu" and masked_scores.dtype in {
        torch.float16,
        torch.bfloat16,
    }:
        softmax_dtype = torch.float32
    probabilities = torch.softmax(
        masked_scores,
        dim=token_dim,
        dtype=softmax_dtype,
    ).to(dtype=masked_scores.dtype)
    if mask is None:
        return probabilities
    return probabilities.masked_fill(~mask, 0.0)


class LabelAttention(nn.Module):
    def __init__(self, input_size: int, projection_size: int, num_classes: int):
        super().__init__()
        self.first_linear = nn.Linear(input_size, projection_size, bias=False)
        self.second_linear = nn.Linear(projection_size, num_classes, bias=False)
        self.third_linear = nn.Linear(input_size, num_classes)
        self._init_weights(mean=0.0, std=0.03)

    def forward(
        self,
        x: torch.Tensor,
        output_attention: bool = False,
        attention_masks: Optional[torch.Tensor] = None,
        attn_grad_hook_fn: Optional[Callable] = None,
    ) -> torch.Tensor:
        """LAAT attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """
        weights = torch.tanh(self.first_linear(x))  # [batch_size, seq_len, proj_size]
        att_scores = self.second_linear(weights)  # [batch_size, seq_len, num_classes]
        att_weights = masked_softmax(
            att_scores,
            attention_masks,
            token_dim=1,
        ).transpose(1, 2)
        weighted_output = att_weights @ x  # [batch_size, num_classes, input_size]
        output = (
            self.third_linear.weight.mul(weighted_output)
            .sum(dim=2)
            .add(self.third_linear.bias)
        )  # [batch_size, num_classes]
        if output_attention:
            return output, att_weights
        return output

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        torch.nn.init.normal_(self.first_linear.weight, mean, std)
        torch.nn.init.normal_(self.second_linear.weight, mean, std)
        torch.nn.init.normal_(self.third_linear.weight, mean, std)


class LabelCrossAttention(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        scale: float = 1.0,
        semantic_query_adapter: bool = False,
        semantic_query_fragments: int = 3,
        semantic_query_alpha: float = 0.1,
        semantic_query_frequency_gate: bool = False,
    ):
        super().__init__()
        self.weights_k = nn.Linear(input_size, input_size, bias=False)
        self.label_representations = torch.nn.Parameter(
            torch.rand(num_classes, input_size), requires_grad=True
        )
        self.weights_v = nn.Linear(input_size, input_size)
        self.output_linear = nn.Linear(input_size, 1)
        self.layernorm = nn.LayerNorm(input_size)
        self.num_classes = num_classes
        self.scale = scale
        self.semantic_query_adapter = bool(semantic_query_adapter)
        if self.semantic_query_adapter:
            if semantic_query_fragments <= 0:
                raise ValueError("semantic_query_fragments must be positive")
            self.register_buffer(
                "semantic_fragment_embeddings",
                torch.zeros(num_classes, semantic_query_fragments, input_size),
            )
            self.register_buffer(
                "semantic_fragment_mask",
                torch.zeros(num_classes, semantic_query_fragments, dtype=torch.bool),
            )
            self.semantic_pool_scorer = nn.Linear(input_size, 1, bias=False)
            self.semantic_projection = nn.Linear(input_size, input_size, bias=False)
            self.semantic_query_layernorm = nn.LayerNorm(
                input_size, elementwise_affine=False
            )
            self.semantic_alpha = nn.Parameter(
                torch.tensor(float(semantic_query_alpha))
            )
            self.semantic_query_frequency_gate = bool(semantic_query_frequency_gate)
            self.register_buffer(
                "semantic_query_alpha_gate",
                torch.ones(num_classes),
                persistent=True,
            )
        self._init_weights(mean=0.0, std=0.03)

    @torch.no_grad()
    def set_semantic_query_frequency_gate(
        self,
        frequencies: torch.Tensor,
        tail_alpha: float = 0.2,
        torso_alpha: float = 0.1,
        head_alpha: float = 0.05,
        tail_max_frequency: int = 10,
        torso_max_frequency: int = 100,
    ) -> None:
        """Set a fixed frequency-dependent multiplier for semantic residuals."""
        if not self.semantic_query_adapter:
            raise RuntimeError("Semantic query adapter is disabled")
        if not self.semantic_query_frequency_gate:
            raise RuntimeError("Semantic query frequency gate is disabled")
        if frequencies.numel() != self.num_classes:
            raise ValueError("Frequency vector does not match number of labels")
        if tail_alpha <= 0 or torso_alpha <= 0 or head_alpha <= 0:
            raise ValueError("Semantic alpha values must be positive")
        frequencies = frequencies.to(
            device=self.semantic_query_alpha_gate.device, dtype=torch.long
        )
        alpha = torch.full_like(frequencies, float(head_alpha), dtype=torch.float32)
        alpha = torch.where(
            frequencies <= int(torso_max_frequency),
            torch.full_like(alpha, float(torso_alpha)),
            alpha,
        )
        alpha = torch.where(
            frequencies <= int(tail_max_frequency),
            torch.full_like(alpha, float(tail_alpha)),
            alpha,
        )
        base = self.semantic_alpha.detach().abs().clamp_min(1e-6)
        self.semantic_query_alpha_gate.copy_(alpha / base)

    @torch.no_grad()
    def set_semantic_fragments(
        self,
        embeddings: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        """Install frozen label-knowledge fragments in tokenizer label order."""
        if not self.semantic_query_adapter:
            raise RuntimeError("Semantic query adapter is disabled")
        if embeddings.shape != self.semantic_fragment_embeddings.shape:
            raise ValueError(
                "Semantic fragment shape mismatch: "
                f"{tuple(embeddings.shape)} != "
                f"{tuple(self.semantic_fragment_embeddings.shape)}"
            )
        if mask.shape != self.semantic_fragment_mask.shape:
            raise ValueError("Semantic fragment mask shape mismatch")
        self.semantic_fragment_embeddings.copy_(
            embeddings.to(
                device=self.semantic_fragment_embeddings.device,
                dtype=self.semantic_fragment_embeddings.dtype,
            )
        )
        self.semantic_fragment_mask.copy_(
            mask.to(device=self.semantic_fragment_mask.device, dtype=torch.bool)
        )

    def effective_label_queries(self) -> torch.Tensor:
        """Fuse trainable LAAT queries with pooled semantic label knowledge."""
        queries = self.label_representations
        if not self.semantic_query_adapter:
            return queries

        fragment_mask = self.semantic_fragment_mask
        has_semantics = fragment_mask.any(dim=1)
        if not bool(has_semantics.any()):
            return queries

        fragments = self.semantic_fragment_embeddings.to(dtype=queries.dtype)
        pooling_logits = self.semantic_pool_scorer(fragments).squeeze(-1)
        pooling_logits = pooling_logits.masked_fill(
            ~fragment_mask, torch.finfo(pooling_logits.dtype).min
        )
        pooling_weights = torch.softmax(pooling_logits, dim=1)
        pooling_weights = torch.where(
            has_semantics.unsqueeze(1),
            pooling_weights,
            torch.zeros_like(pooling_weights),
        )
        semantic_embedding = (
            pooling_weights.unsqueeze(-1) * fragments
        ).sum(dim=1)
        if self.semantic_query_frequency_gate:
            semantic_alpha = self.semantic_alpha * self.semantic_query_alpha_gate
        else:
            semantic_alpha = self.semantic_alpha
        fused = queries + semantic_alpha.unsqueeze(-1) * self.semantic_projection(
            semantic_embedding
        )

        # LayerNorm is followed by norm restoration because PLM-ICD initializes
        # LAAT queries at std=0.03; raw LayerNorm would otherwise increase their
        # scale by roughly 30x and confound the semantic ablation.
        normalized = self.semantic_query_layernorm(fused)
        normalized = normalized / normalized.norm(dim=1, keepdim=True).clamp_min(1e-8)
        normalized = normalized * queries.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.where(has_semantics.unsqueeze(1), normalized, queries)

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        output_attention: bool = False,
        attn_grad_hook_fn: Optional[Callable] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Label Cross Attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """

        V = self.weights_v(x)
        K = self.weights_k(x)
        Q = self.effective_label_queries()

        att_weights = Q.matmul(K.transpose(1, 2))

        # replace nan with max value of float16
        # att_weights = torch.where(
        #     torch.isnan(att_weights), torch.tensor(30000), att_weights
        # )
        att_weights, _ = mask_attention_scores(
            att_weights,
            attention_masks,
            token_dim=2,
        )
        attention = masked_softmax(
            att_weights / self.scale,
            attention_masks,
            token_dim=2,
        )  # [batch_size, num_classes, seq_len]
        if attn_grad_hook_fn is not None:
            attention.register_hook(attn_grad_hook_fn)

        y = attention @ V  # [batch_size, num_classes, input_size]

        y = self.layernorm(y)

        output = (
            self.output_linear.weight.mul(y)
            .sum(dim=2)
            .add(self.output_linear.bias)  # [batch_size, num_classes]
        )

        if output_attention:
            return output, att_weights

        return output

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        self.weights_k.weight = torch.nn.init.normal_(self.weights_k.weight, mean, std)
        self.weights_v.weight = torch.nn.init.normal_(self.weights_v.weight, mean, std)
        self.label_representations = torch.nn.init.normal_(
            self.label_representations, mean, std
        )
        self.output_linear.weight = torch.nn.init.normal_(
            self.output_linear.weight, mean, std
        )
        if self.semantic_query_adapter:
            torch.nn.init.zeros_(self.semantic_pool_scorer.weight)
            torch.nn.init.normal_(self.semantic_projection.weight, mean, std)


class TailResidualQueryHead(nn.Module):
    """A bounded label-query residual that only changes selected Tail logits.

    The frozen anchor makes the random and semantic variants parameter matched.
    A zero-initialized per-label gate guarantees that enabling the module does
    not change the parent checkpoint before residual training starts.
    """

    def __init__(
        self,
        num_classes: int,
        tail_indices: torch.Tensor,
        query_anchor: torch.Tensor,
        max_scale: float = 0.5,
    ) -> None:
        super().__init__()
        tail_indices = tail_indices.detach().to(dtype=torch.long, device="cpu")
        query_anchor = query_anchor.detach().to(dtype=torch.float32, device="cpu")
        if tail_indices.ndim != 1 or tail_indices.numel() == 0:
            raise ValueError("tail_indices must be a non-empty one-dimensional tensor")
        if query_anchor.ndim != 2 or query_anchor.size(0) != tail_indices.numel():
            raise ValueError("query_anchor must have one row per Tail label")
        if max_scale <= 0:
            raise ValueError("max_scale must be positive")

        self.num_classes = int(num_classes)
        self.max_scale = float(max_scale)
        self.register_buffer("tail_indices", tail_indices, persistent=True)
        self.register_buffer("query_anchor", query_anchor, persistent=True)
        self.query_delta = nn.Parameter(torch.zeros_like(query_anchor))
        self.raw_scale = nn.Parameter(torch.zeros(tail_indices.numel()))

    def effective_queries(self) -> torch.Tensor:
        anchor = self.query_anchor.to(dtype=self.query_delta.dtype)
        query = anchor + self.query_delta
        target_norm = anchor.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return query / query.norm(dim=1, keepdim=True).clamp_min(1e-8) * target_norm

    def forward(
        self,
        x: torch.Tensor,
        base_attention: LabelCrossAttention,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if not isinstance(base_attention, LabelCrossAttention):
            raise TypeError("TailResidualQueryHead requires LabelCrossAttention")

        keys = base_attention.weights_k(x)
        values = base_attention.weights_v(x)
        scores = self.effective_queries().matmul(keys.transpose(1, 2))
        attention = masked_softmax(
            scores / base_attention.scale,
            attention_masks,
            token_dim=2,
        )
        representations = base_attention.layernorm(attention @ values)
        raw_logits = (
            base_attention.output_linear.weight.mul(representations)
            .sum(dim=2)
            .add(base_attention.output_linear.bias)
        )
        scales = self.max_scale * torch.tanh(self.raw_scale)
        tail_delta = raw_logits * scales.unsqueeze(0)
        output = x.new_zeros((x.size(0), self.num_classes))
        output[:, self.tail_indices] = tail_delta
        return output


class DifferentialLabelCrossAttention(nn.Module):
    """Label attention with separate supporting and opposing evidence streams.

    The positive stream preserves the public attention interface used by the
    existing explainers. The negative stream includes a learned null position,
    allowing a label to express that no opposing evidence is present instead of
    forcing a probability distribution over unrelated tokens.
    """

    def __init__(
        self,
        input_size: int,
        num_classes: int,
        scale: float = 1.0,
        negative_evidence_scale: float = 1.0,
        negative_null_logit_bias: float = 14.0,
        negative_evidence_warmup_steps: int = 0,
    ):
        super().__init__()
        if scale <= 0:
            raise ValueError("scale must be positive")
        if negative_evidence_scale <= 0:
            raise ValueError("negative_evidence_scale must be positive")
        if negative_evidence_warmup_steps < 0:
            raise ValueError("negative_evidence_warmup_steps cannot be negative")

        self.weights_k = nn.Linear(input_size, input_size, bias=False)
        self.weights_v = nn.Linear(input_size, input_size)
        self.positive_label_representations = nn.Parameter(
            torch.empty(num_classes, input_size)
        )
        self.negative_label_representations = nn.Parameter(
            torch.empty(num_classes, input_size)
        )
        self.negative_null_key = nn.Parameter(torch.empty(input_size))
        self.negative_null_logit_bias = nn.Parameter(
            torch.tensor(float(negative_null_logit_bias))
        )
        unconstrained_scale = math.log(math.expm1(negative_evidence_scale))
        self.negative_evidence_scale_unconstrained = nn.Parameter(
            torch.tensor(unconstrained_scale)
        )

        self.positive_output_linear = nn.Linear(input_size, 1)
        self.negative_output_linear = nn.Linear(input_size, 1)
        self.positive_layernorm = nn.LayerNorm(input_size)
        self.negative_layernorm = nn.LayerNorm(input_size)
        self.num_classes = num_classes
        self.scale = scale
        self.negative_evidence_warmup_steps = negative_evidence_warmup_steps
        self.register_buffer(
            "_negative_evidence_training_steps",
            torch.zeros((), dtype=torch.long),
        )
        self._init_weights(mean=0.0, std=0.03)

    @property
    def negative_evidence_scale(self) -> torch.Tensor:
        """Return a strictly positive global negative-evidence multiplier."""
        return torch.nn.functional.softplus(self.negative_evidence_scale_unconstrained)

    @property
    def negative_evidence_warmup_multiplier(self) -> torch.Tensor:
        """Return the linear ramp applied to the subtractive evidence stream."""
        if self.negative_evidence_warmup_steps == 0:
            return self._negative_evidence_training_steps.new_tensor(1.0)
        return (
            self._negative_evidence_training_steps.float()
            / self.negative_evidence_warmup_steps
        ).clamp(max=1.0)

    def compute_attention_components(
        self,
        x: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute evidence attentions, contexts, and the negative null gate."""
        values = self.weights_v(x)
        keys = self.weights_k(x)

        positive_logits = self.positive_label_representations.matmul(
            keys.transpose(1, 2)
        )
        negative_logits = self.negative_label_representations.matmul(
            keys.transpose(1, 2)
        )

        positive_logits, _ = mask_attention_scores(
            positive_logits,
            attention_masks,
            token_dim=2,
        )
        negative_logits, _ = mask_attention_scores(
            negative_logits,
            attention_masks,
            token_dim=2,
        )

        positive_attention = masked_softmax(
            positive_logits / self.scale,
            attention_masks,
            token_dim=2,
        )

        null_logits = self.negative_label_representations.matmul(self.negative_null_key)
        null_logits = null_logits + self.negative_null_logit_bias
        null_logits = null_logits.view(1, self.num_classes, 1).expand(x.size(0), -1, -1)
        negative_attention_with_null = torch.softmax(
            torch.cat((negative_logits, null_logits), dim=2) / self.scale,
            dim=2,
        )
        negative_token_attention = negative_attention_with_null[:, :, :-1]
        negative_null_probability = negative_attention_with_null[:, :, -1]
        negative_gate = 1.0 - negative_null_probability
        conditional_negative_attention = (
            negative_token_attention
            / negative_gate.clamp_min(
                torch.finfo(negative_token_attention.dtype).eps
            ).unsqueeze(2)
        )

        return {
            "positive_logits": positive_logits,
            "negative_logits": negative_logits,
            "positive_attention": positive_attention,
            "negative_attention": conditional_negative_attention,
            "negative_gate": negative_gate,
            "negative_null_probability": negative_null_probability,
            "positive_context": positive_attention @ values,
            "negative_context": conditional_negative_attention @ values,
        }

    def forward_with_components(
        self,
        x: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        attn_grad_hook_fn: Optional[Callable] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return logits plus the evidence components used to produce them."""
        components = self.compute_attention_components(x, attention_masks)
        positive_attention = components["positive_attention"]
        if attn_grad_hook_fn is not None:
            positive_attention.register_hook(attn_grad_hook_fn)

        positive_context = self.positive_layernorm(components["positive_context"])
        negative_context = self.negative_layernorm(components["negative_context"])
        positive_score = self.positive_output_linear(positive_context).squeeze(2)
        negative_magnitude = torch.nn.functional.softplus(
            self.negative_output_linear(negative_context).squeeze(2)
        )
        warmup_multiplier = self.negative_evidence_warmup_multiplier.to(
            dtype=positive_score.dtype
        )
        output = positive_score - (
            self.negative_evidence_scale
            * warmup_multiplier
            * components["negative_gate"]
            * negative_magnitude
        )
        if self.training and self.negative_evidence_warmup_steps > 0:
            with torch.no_grad():
                self._negative_evidence_training_steps.add_(1)

        return output, components

    def forward(
        self,
        x: torch.Tensor,
        output_attention: bool = False,
        attention_masks: Optional[torch.Tensor] = None,
        attn_grad_hook_fn: Optional[Callable] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        output, components = self.forward_with_components(
            x,
            attention_masks=attention_masks,
            attn_grad_hook_fn=attn_grad_hook_fn,
        )

        if output_attention:
            return output, components["positive_logits"]
        return output

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        training_steps_key = f"{prefix}_negative_evidence_training_steps"
        if training_steps_key not in state_dict:
            state_dict[training_steps_key] = self._negative_evidence_training_steps
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        torch.nn.init.normal_(self.weights_k.weight, mean, std)
        torch.nn.init.normal_(self.weights_v.weight, mean, std)
        torch.nn.init.zeros_(self.weights_v.bias)
        torch.nn.init.normal_(self.positive_label_representations, mean, std)
        torch.nn.init.normal_(self.negative_label_representations, mean, std)
        torch.nn.init.normal_(self.negative_null_key, mean, std)
        torch.nn.init.normal_(self.positive_output_linear.weight, mean, std)
        torch.nn.init.zeros_(self.positive_output_linear.bias)
        torch.nn.init.normal_(self.negative_output_linear.weight, mean, std)
        torch.nn.init.zeros_(self.negative_output_linear.bias)


class InputMasker(nn.Module):
    def __init__(self, input_size: int, scale: float = 1.0):
        super().__init__()
        self.weights_k = nn.Linear(input_size, input_size)
        self.weights_q = nn.Linear(input_size, input_size)
        self.weights_v = nn.Linear(input_size, input_size)
        self.output_linear = nn.Linear(input_size, 1)
        self.layernorm = nn.LayerNorm(input_size)
        self.scale = scale
        self._init_weights(mean=0.0, std=0.03)

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        output_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Label Cross Attention mechanism

        Args:
            x (torch.Tensor): [batch_size, seq_len, input_size]

        Returns:
            torch.Tensor: [batch_size, num_classes]
        """

        V = self.weights_v(x)
        K = self.weights_k(x)
        Q = self.weights_q(x)

        att_weights = Q.matmul(K.transpose(1, 2))

        # replace nan with max value of float16
        # att_weights = torch.where(
        #     torch.isnan(att_weights), torch.tensor(30000), att_weights
        # )
        att_weights, _ = mask_attention_scores(
            att_weights,
            attention_masks,
            token_dim=2,
        )
        attention = masked_softmax(
            att_weights / self.scale,
            attention_masks,
            token_dim=2,
        )  # [batch_size, seq_len, seq_len]

        y = attention @ V  # [batch_size, num_classes, input_size]

        y = self.layernorm(y)

        output = (
            self.output_linear.weight.mul(y)
            .sum(dim=2)
            .add(self.output_linear.bias)  # [batch_size, num_classes]
        )

        if output_attention:
            return output, att_weights

        return output

    def _init_weights(self, mean: float = 0.0, std: float = 0.03) -> None:
        """
        Initialise the weights

        Args:
            mean (float, optional): Mean of the normal distribution. Defaults to 0.0.
            std (float, optional): Standard deviation of the normal distribution. Defaults to 0.03.
        """

        self.weights_k.weight = torch.nn.init.normal_(self.weights_k.weight, mean, std)
        self.weights_v.weight = torch.nn.init.normal_(self.weights_v.weight, mean, std)
        self.weights_q.weight = torch.nn.init.normal_(self.weights_q.weight, mean, std)
        self.output_linear.weight = torch.nn.init.normal_(
            self.output_linear.weight, mean, std
        )

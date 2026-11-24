from typing import Callable, Optional

import torch
import torch.utils.checkpoint
from pydantic import BaseModel
from torch import nn
from transformers import AutoConfig, AutoModel

from explainable_medical_coding.explainability.helper_functions import (
    create_baseline_input,
)
from explainable_medical_coding.models.modules.attention import (
    DifferentialLabelCrossAttention,
    InputMasker,
    LabelAttention,
    LabelCrossAttention,
    TailResidualQueryHead,
    masked_softmax,
)


class ModuleNames(BaseModel):
    ln_1: str
    ln_2: str
    dense_values: str
    dense_heads: str
    model_layer_name: str


class RobertaModuleNames(ModuleNames):
    ln_1: str = "attention.output.LayerNorm"
    ln_2: str = "output.LayerNorm"
    dense_values: str = "attention.self.value"
    dense_heads: str = "attention.output.dense"
    model_layer_name: str = "roberta_encoder.encoder.layer"


class PLMICD(nn.Module):
    def __init__(
        self,
        num_classes: int,
        model_path: str,
        chunk_size: int,
        pad_token_id: int,
        cross_attention: bool = True,
        scale: float = 1.0,
        mask_input: bool = False,
        add_chunk_position_embeddings: bool = False,
        max_chunks: int = 64,
        differential_attention: bool = False,
        negative_evidence_scale: float = 1.0,
        negative_null_logit_bias: float = 14.0,
        negative_evidence_warmup_steps: int = 0,
        use_evidence_module: bool = False,
        dropout: Optional[float] = None,
        semantic_query_adapter: bool = False,
        semantic_query_fragments: int = 3,
        semantic_query_alpha: float = 0.1,
        semantic_query_frequency_gate: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.chunk_size = chunk_size
        self.pad_token_id = pad_token_id
        self.module_names = RobertaModuleNames()
        self.gradient = None
        self.add_chunk_position_embeddings = add_chunk_position_embeddings
        self.max_chunks = max_chunks
        self.use_evidence_module = use_evidence_module
        self.tail_residual: Optional[TailResidualQueryHead] = None

        config_overrides = {}
        if dropout is not None:
            config_overrides = {
                "hidden_dropout_prob": float(dropout),
                "attention_probs_dropout_prob": float(dropout),
            }
        self.config = AutoConfig.from_pretrained(
            model_path,
            num_labels=num_classes,
            finetuning_task=None,
            **config_overrides,
        )

        self.roberta_encoder = AutoModel.from_pretrained(
            model_path, config=self.config, add_pooling_layer=False
        )
        if self.add_chunk_position_embeddings:
            if self.max_chunks <= 0:
                raise ValueError("max_chunks must be positive")
            self.chunk_position_embeddings = nn.Embedding(
                self.max_chunks,
                self.config.hidden_size,
            )
            nn.init.normal_(
                self.chunk_position_embeddings.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )

        if differential_attention and not cross_attention:
            raise ValueError("differential_attention requires cross_attention")
        if differential_attention:
            self.label_wise_attention = DifferentialLabelCrossAttention(
                input_size=self.config.hidden_size,
                num_classes=num_classes,
                scale=scale,
                negative_evidence_scale=negative_evidence_scale,
                negative_null_logit_bias=negative_null_logit_bias,
                negative_evidence_warmup_steps=negative_evidence_warmup_steps,
            )
        elif cross_attention:
            self.label_wise_attention = LabelCrossAttention(
                input_size=self.config.hidden_size,
                num_classes=num_classes,
                scale=scale,
                semantic_query_adapter=semantic_query_adapter,
                semantic_query_fragments=semantic_query_fragments,
                semantic_query_alpha=semantic_query_alpha,
                semantic_query_frequency_gate=semantic_query_frequency_gate,
            )
        else:
            self.label_wise_attention = LabelAttention(
                input_size=self.config.hidden_size,
                projection_size=self.config.hidden_size,
                num_classes=num_classes,
            )
        self.mask_input = mask_input
        if self.mask_input:
            self.input_masker = InputMasker(
                input_size=self.config.hidden_size, scale=scale
            )

    @torch.no_grad()
    def predict(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        return torch.sigmoid(
            self.forward(input_ids=input_ids, attention_masks=attention_mask)
        )

    def split_input_into_chunks(
        self, input_sequence: torch.Tensor, pad_index: int
    ) -> torch.Tensor:
        """Split input into chunks of chunk_size.

        Args:
            input_sequence (torch.Tensor): input sequence to split (batch_size, seq_len)
            pad_index (int): padding index

        Returns:
            torch.Tensor: reshaped input (batch_size, num_chunks, chunk_size)
        """
        batch_size = input_sequence.size(0)
        # pad input to be divisible by chunk_size
        input_sequence = nn.functional.pad(
            input_sequence,
            (0, self.chunk_size - input_sequence.size(1) % self.chunk_size),
            value=pad_index,
        )
        return input_sequence.view(batch_size, -1, self.chunk_size)

    def roberta_encode_embedding_input(self, embedding, attention_masks):
        input_shape = embedding.size()[:-1]
        extended_attention_mask = self.roberta_encoder.get_extended_attention_mask(
            attention_masks, input_shape
        )
        head_mask = self.roberta_encoder.get_head_mask(
            None, self.roberta_encoder.config.num_hidden_layers
        )
        encoder_outputs = self.roberta_encoder.encoder(
            embedding,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            return_dict=False,
        )
        sequence_output = encoder_outputs[0]
        return sequence_output

    def get_chunked_attention_masks(
        self, attention_masks: torch.Tensor
    ) -> torch.Tensor:
        return self.split_input_into_chunks(attention_masks, 0)

    def add_global_chunk_positions(
        self,
        hidden_output: torch.Tensor,
        num_chunks: int,
    ) -> torch.Tensor:
        """Add one learned global position vector to every token in a chunk."""
        if not self.add_chunk_position_embeddings:
            return hidden_output
        if num_chunks > self.max_chunks:
            raise ValueError(
                f"Document has {num_chunks} chunks, exceeding max_chunks={self.max_chunks}"
            )
        if hidden_output.size(1) != num_chunks * self.chunk_size:
            raise ValueError("Hidden sequence length is incompatible with chunk_size")

        batch_size, _, hidden_size = hidden_output.size()
        chunk_positions = self.chunk_position_embeddings(
            torch.arange(num_chunks, device=hidden_output.device)
        )
        chunk_positions = (
            chunk_positions[:, None, :]
            .expand(num_chunks, self.chunk_size, hidden_size)
            .reshape(1, num_chunks * self.chunk_size, hidden_size)
        )
        return hidden_output + chunk_positions.expand(batch_size, -1, -1)

    def get_input_embeddings(self):
        return self.roberta_encoder.embeddings

    def get_chunked_embedding(self, input_ids):
        input_ids = self.split_input_into_chunks(input_ids, self.pad_token_id)
        chunk_size = input_ids.size(-1)
        embedding = self.roberta_encoder.embeddings(input_ids.view(-1, chunk_size))
        return embedding

    def get_token_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get token embeddings. Huggingface Roberta model can't return more than 512 token embeddings at once.

        Args:
            input_ids (torch.Tensor): input ids

        Returns:
            torch.Tensor: token embeddings
        """
        sequence_length = input_ids.size(1)
        batch_size = input_ids.size(0)
        if sequence_length <= 512:
            return self.roberta_encoder.embeddings(input_ids)
        input_ids = self.split_input_into_chunks(input_ids, self.pad_token_id)
        chunk_size = input_ids.size(-1)
        chunked_embeddings = self.roberta_encoder.embeddings(
            input_ids.view(-1, chunk_size)
        )
        embeddings = chunked_embeddings.view(
            batch_size, -1, chunked_embeddings.size(-1)
        )
        return embeddings[:, :sequence_length]

    def encoder(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        input_ids = self.split_input_into_chunks(input_ids, self.pad_token_id)
        if attention_masks is not None:
            attention_masks = self.get_chunked_attention_masks(attention_masks)
        batch_size, num_chunks, chunk_size = input_ids.size()
        outputs = self.roberta_encoder(
            input_ids=input_ids.view(-1, chunk_size),
            attention_mask=attention_masks.view(-1, chunk_size)
            if attention_masks is not None
            else None,
            return_dict=False,
        )
        hidden_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)
        return self.add_global_chunk_positions(hidden_output, num_chunks)

    def forward_with_input_masking(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_mask: bool = False,
        baseline_token_id: int = 500001,
    ):
        baseline = create_baseline_input(
            input_ids,
            baseline_token_id=baseline_token_id,
            cls_token_id=baseline_token_id,
            eos_token_id=baseline_token_id,
        )
        baseline = baseline.requires_grad_(False)
        baseline_embeddings = self.get_chunked_embedding(baseline).detach()
        chunked_input_embeddings = self.get_chunked_embedding(input_ids)
        with torch.no_grad():
            token_representations = (
                self.get_token_representations_from_chunked_embeddings(
                    chunked_input_embeddings.detach(), attention_masks
                )
            )
        input_mask = self.input_masker(
            token_representations, attention_masks=attention_masks
        )

        input_mask_sigmoid = torch.sigmoid(input_mask)
        input_mask_sigmoid = input_mask_sigmoid.view(-1, self.chunk_size, 1)
        masked_chunked_input_embeddings = (
            chunked_input_embeddings * input_mask_sigmoid
            + baseline_embeddings * (1 - input_mask_sigmoid)
        )
        masked_token_representations = (
            self.get_token_representations_from_chunked_embeddings(
                masked_chunked_input_embeddings, attention_masks
            )
        )
        if output_mask:
            return self.label_wise_attention(
                masked_token_representations,
                attention_masks=attention_masks,
                output_attention=output_attentions,
            ), input_mask

        return self.label_wise_attention(
            masked_token_representations,
            attention_masks=attention_masks,
            output_attention=output_attentions,
        )

    def get_token_representations_from_chunked_embeddings(
        self,
        chunked_embedding: torch.Tensor,
        attention_masks: torch.Tensor,
    ) -> torch.Tensor:
        """Get token representations from chunked embeddings.

        Args:
            chunked_embedding (torch.Tensor): Chunked embedding of shape [batch_size*num_chunks, chunk_size, embedding_size]
            attention_masks (torch.Tensor): Attention masks of shape [batch_size, sequence_length]

        Returns:
            torch.Tensor: Token representations of shape [batch_size, sequence_length, hidden_size]
        """
        (
            num_chunks_times_batch_size,
            chunk_size,
            embedding_size,
        ) = chunked_embedding.size()
        batch_size = attention_masks.size(0)
        num_chunks = num_chunks_times_batch_size // batch_size
        chunked_attention_masks = self.get_chunked_attention_masks(attention_masks)
        hidden_outputs = self.roberta_encode_embedding_input(
            embedding=chunked_embedding.view(-1, chunk_size, embedding_size),
            attention_masks=chunked_attention_masks.view(-1, chunk_size),
        )
        hidden_outputs = hidden_outputs.view(
            batch_size,
            num_chunks * chunk_size,
            -1,
        )
        return self.add_global_chunk_positions(hidden_outputs, num_chunks)

    def forward_embedding_input(
        self,
        chunked_embedding: torch.Tensor,
        attention_masks: torch.Tensor,
        output_attention: bool = False,
    ) -> torch.Tensor:
        """Forward pass for the model with chunked embedding input.

        Args:
            chunked_embedding (torch.Tensor): Chunked embedding of shape [batch_size*num_chunks, chunk_size, embedding_size]
            attention_masks (torch.Tensor): Attention masks of shape [batch_size, num_chunks, chunk_size]

        Returns:
            torch.Tensor:
        """
        token_representations = self.get_token_representations_from_chunked_embeddings(
            chunked_embedding, attention_masks
        )
        return self.label_wise_attention(
            token_representations,
            attention_masks=attention_masks,
            output_attention=output_attention,
        )

    @torch.no_grad()
    def de_chunk_attention(
        self,
        attentions_chunked: torch.Tensor,
        batch_size: int,
        num_layers: int,
        num_chunks: int,
        chunk_size: int,
    ) -> torch.Tensor:
        """De-chunk attention.

        Args:
            attentions_chunked (torch.Tensor): Attention matrix of shape [batch_size, num_chunks, num_layers, chunk_size, chunk_size]
            batch_size (int): Batch size
            num_layers (int): Number of layers
            num_chunks (int): Number of chunks
            chunk_size (int): Chunk size

        Returns:
            torch.Tensor: Attention matrix of shape [batch_size, num_layers, num_chunks*chunk_size, num_chunks*chunk_size]
        """
        attentions = torch.zeros(
            batch_size,
            num_layers,
            num_chunks * chunk_size,
            num_chunks * chunk_size,
            device=attentions_chunked.device,
            dtype=torch.float16,
        )
        for chunk_idx in range(num_chunks):
            attentions[
                :,
                :,
                chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size,
                chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size,
            ] = attentions_chunked[:, chunk_idx]

        return attentions

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        attn_grad_hook_fn: Optional[Callable] = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.mask_input:
            return self.forward_with_input_masking(
                input_ids, attention_masks, output_attentions, False
            )
        hidden_output = self.encoder(input_ids, attention_masks)
        logits = self.label_wise_attention(
            hidden_output,
            attention_masks=attention_masks,
            output_attention=output_attentions,
            attn_grad_hook_fn=attn_grad_hook_fn,
        )
        if output_attentions:
            base_logits, attention = logits
            if self.tail_residual is not None:
                base_logits = base_logits + self.tail_residual(
                    hidden_output, self.label_wise_attention, attention_masks
                )
            return base_logits, attention
        if self.tail_residual is not None:
            logits = logits + self.tail_residual(
                hidden_output, self.label_wise_attention, attention_masks
            )
        return logits

    @torch.no_grad()
    def configure_tail_residual(
        self,
        frequencies: torch.Tensor,
        mode: str,
        tail_max_frequency: int = 10,
        max_scale: float = 0.5,
    ) -> None:
        """Attach a random- or semantic-anchor Tail residual query branch."""
        if not isinstance(self.label_wise_attention, LabelCrossAttention):
            raise TypeError("Tail residuals require LabelCrossAttention")
        frequencies = frequencies.detach().to(dtype=torch.long, device="cpu")
        if frequencies.numel() != self.num_classes:
            raise ValueError("Frequency vector does not match model label count")
        tail_indices = torch.nonzero(
            frequencies <= int(tail_max_frequency), as_tuple=False
        ).flatten()
        if tail_indices.numel() == 0:
            raise ValueError("Tail definition selected no labels")

        base_queries = self.label_wise_attention.label_representations.detach().cpu()
        if mode == "random":
            anchor = torch.empty_like(base_queries[tail_indices])
            torch.nn.init.normal_(anchor, mean=0.0, std=0.03)
        elif mode == "semantic":
            attention = self.label_wise_attention
            if not attention.semantic_query_adapter:
                raise ValueError("Semantic residual requires semantic fragments")
            fragments = attention.semantic_fragment_embeddings.detach().cpu()
            fragment_mask = attention.semantic_fragment_mask.detach().cpu()
            selected_fragments = fragments[tail_indices]
            selected_mask = fragment_mask[tail_indices]
            denominator = selected_mask.sum(dim=1, keepdim=True).clamp_min(1)
            anchor = (
                selected_fragments * selected_mask.unsqueeze(-1)
            ).sum(dim=1) / denominator
            has_semantics = selected_mask.any(dim=1)
            fallback = base_queries[tail_indices]
            anchor = torch.where(has_semantics.unsqueeze(1), anchor, fallback)
            target_norm = fallback.norm(dim=1, keepdim=True).clamp_min(1e-8)
            anchor = anchor / anchor.norm(dim=1, keepdim=True).clamp_min(1e-8)
            anchor = anchor * target_norm
        else:
            raise ValueError("Tail residual mode must be 'random' or 'semantic'")

        self.tail_residual = TailResidualQueryHead(
            num_classes=self.num_classes,
            tail_indices=tail_indices,
            query_anchor=anchor,
            max_scale=max_scale,
        ).to(self.label_wise_attention.label_representations.device)

    @torch.no_grad()
    def configure_tail_residual_from_state(
        self,
        tail_indices: torch.Tensor,
        query_anchor: torch.Tensor,
        max_scale: float = 0.5,
    ) -> None:
        """Recreate a saved residual module before loading its state dict."""
        self.tail_residual = TailResidualQueryHead(
            num_classes=self.num_classes,
            tail_indices=tail_indices,
            query_anchor=query_anchor,
            max_scale=max_scale,
        )

    def freeze_for_tail_residual(self) -> None:
        """Freeze E9 and leave only the newly attached residual trainable."""
        if self.tail_residual is None:
            raise RuntimeError("Tail residual has not been configured")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        for parameter in self.tail_residual.parameters():
            parameter.requires_grad_(True)
        self.roberta_encoder.eval()
        self.label_wise_attention.eval()

    def train(self, mode: bool = True) -> "PLMICD":
        super().train(mode)
        if self.tail_residual is not None and any(
            parameter.requires_grad for parameter in self.tail_residual.parameters()
        ):
            # A frozen teacher must not drift through dropout statistics/noise.
            self.roberta_encoder.eval()
            self.label_wise_attention.eval()
            self.tail_residual.train(mode)
        return self

    def forward_baseline(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """Return stable baseline logits and normalized [B, T, L] attention."""
        if self.mask_input:
            raise ValueError("forward_baseline is unavailable with input masking")
        hidden_output = self.encoder(input_ids, attention_masks)
        baseline_logits, legacy_attention = self.label_wise_attention(
            hidden_output,
            attention_masks=attention_masks,
            output_attention=True,
        )
        if isinstance(self.label_wise_attention, LabelAttention):
            attention = legacy_attention
        else:
            attention = masked_softmax(
                legacy_attention / self.label_wise_attention.scale,
                attention_masks,
                token_dim=2,
            )
        return {
            "baseline_logits": baseline_logits,
            "baseline_attention": attention.transpose(1, 2),
        }

    def forward_with_components(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Expose DLEA components for training-only auxiliary objectives."""
        if self.mask_input:
            raise ValueError("DLEA components are unavailable with input masking")
        if not isinstance(self.label_wise_attention, DifferentialLabelCrossAttention):
            raise ValueError("forward_with_components requires differential attention")
        hidden_output = self.encoder(input_ids, attention_masks)
        return self.label_wise_attention.forward_with_components(
            hidden_output,
            attention_masks=attention_masks,
        )

    @torch.no_grad()
    def get_encoder_attention_and_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_ids = self.split_input_into_chunks(input_ids, self.pad_token_id)
        if attention_masks is not None:
            attention_masks_chunks = self.split_input_into_chunks(attention_masks, 0)
        batch_size, num_chunks, chunk_size = input_ids.size()
        outputs = self.roberta_encoder(
            input_ids=input_ids.view(-1, chunk_size),
            attention_mask=attention_masks_chunks.view(-1, chunk_size)
            if attention_masks_chunks is not None
            else None,
            return_dict=False,
            output_attentions=True,
            output_hidden_states=True,
        )

        hidden_output = outputs[0].view(batch_size, num_chunks * chunk_size, -1)
        hidden_output = self.add_global_chunk_positions(hidden_output, num_chunks)
        _, label_wise_attention = self.label_wise_attention(
            hidden_output, attention_masks, True
        )
        return outputs[2], outputs[3], label_wise_attention

    @torch.no_grad()
    def attention_rollout(
        self,
        input_ids: torch.Tensor,
        attention_masks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        (
            _,
            attentions,
            label_wise_attention,
        ) = self.get_encoder_attention_and_hidden_states(input_ids, attention_masks)

        label_wise_attention = torch.softmax(label_wise_attention, dim=1)
        attentions = (
            torch.stack(attentions).to(torch.float16).transpose(1, 0)
        )  # [batch_size*num_chunks, num_layers, num_heads, chunk_size, chunk_size]

        batch_size = input_ids.size(0)
        num_chunks = attentions.size(0) // batch_size
        num_layers = attentions.size(1)
        num_heads = attentions.size(2)
        chunk_size = attentions.size(3)

        attentions = attentions.view(
            batch_size, num_chunks, num_layers, num_heads, chunk_size, chunk_size
        )
        attentions = torch.mean(
            attentions, dim=3
        )  # [batch_size, num_chunks, num_layers, chunk_size, chunk_size]
        attentions = self.de_chunk_attention(
            attentions, batch_size, num_layers, num_chunks, chunk_size
        )  # [batch_size, num_layers, num_chunks*chunk_size, num_chunks*chunk_size]

        attentions = (
            attentions
            + torch.eye(chunk_size * num_chunks, device=attentions.device)
            .unsqueeze(0)
            .unsqueeze(0)
            / 2
        )  # add skip connection

        attention_rollout = attentions[:, 0]
        for hidden_layer_idx in range(1, num_layers):
            attention_rollout = attentions[:, hidden_layer_idx] @ attention_rollout

        return label_wise_attention @ attention_rollout

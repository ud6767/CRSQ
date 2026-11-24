from copy import deepcopy
import csv
import math
from functools import lru_cache
from pathlib import Path
from typing import Optional, Sequence

import torch

from explainable_medical_coding.explainability.helper_functions import (
    create_baseline_input,
)
from explainable_medical_coding.utils.datatypes import Batch

diet_gradient_scaler = torch.cuda.amp.GradScaler()
advesarial_noise_gradient_scaler = torch.cuda.amp.GradScaler()


def forward_embedding_noise(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    noise: torch.Tensor,
) -> torch.Tensor:
    """Forward pass of the model when adding noise to the input token embeddings.

    Args:
        model (torch.nn.Module): Model to be used
        input_ids (torch.Tensor): Input token ids
        attention_mask (torch.Tensor): Attention mask
        noise (torch.Tensor): Noise to be added to the input token embeddings

    Returns:
        torch.Tensor: Logits
    """
    chunked_embeddings = model.get_chunked_embedding(input_ids)
    logits = model.forward_embedding_input(
        chunked_embeddings + noise, attention_masks=attention_mask
    )
    return logits


def kl_divergence(
    attention: torch.Tensor, one_hot_evidence: torch.Tensor
) -> torch.Tensor:
    """Calculate the KL divergence loss between the attention and the one hot encoded evidence

    Args:
        attention (torch.Tensor): Attention tensor [number_of_targets, sequence_length]
        one_hot_evidence (torch.Tensor): One hot encoded evidence tensor [number_of_targets, sequence_length]

    Returns:
        torch.Tensor: KL divergence lossPatie
    """
    # sum along the sequence length
    attention = torch.log_softmax(attention, dim=1)  # [num_classes, seq_len]
    one_hot_evidence = torch.nn.functional.normalize(one_hot_evidence, p=1.0, dim=-1)
    kl = torch.nn.functional.kl_div(attention, one_hot_evidence, reduction="mean")
    if torch.isnan(kl):
        raise ValueError("KL loss is NaN")

    return kl


def one_hot_encode_evidence_token_ids(
    evidence_token_ids: Sequence[Sequence[int]], max_length: int
) -> torch.Tensor:
    output_tensor = torch.zeros((len(evidence_token_ids), max_length))
    for target_idx, target_evidence_token_ids in enumerate(evidence_token_ids):
        if len(target_evidence_token_ids) > 0:
            output_tensor[target_idx, target_evidence_token_ids] = 1
    return output_tensor.float()


def binary_cross_entropy_loss(
    batch: Batch,
    model: torch.nn.Module,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, targets, attention_masks = (
        batch.input_ids,
        batch.targets,
        batch.attention_masks,
    )
    logits = model(input_ids, attention_masks)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return torch.sigmoid(logits), targets, loss


@lru_cache(maxsize=8)
def _load_label_frequencies(path: str) -> torch.Tensor:
    """Load frequencies in target-tokenizer order for frequency-aware B4/B5."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or "train_frequency" not in rows[0]:
        raise ValueError(f"Invalid label frequency file: {path}")
    return torch.tensor([int(row["train_frequency"]) for row in rows], dtype=torch.float32)


def _paired_consistency_coefficients(
    clean_probabilities: torch.Tensor,
    augmented_probabilities: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    frequency_file: str,
    frequency_gamma: float,
    frequency_weight_max: float,
    hard_negative_k: int,
    frequency_weight_ratio_max: Optional[float] = None,
) -> torch.Tensor:
    """Return the per-cell coefficients of the selected consistency objective."""
    coefficients = torch.zeros_like(clean_probabilities)
    positive = targets.bool()
    negative = ~positive

    if mode == "uniform":
        return torch.full_like(coefficients, 1.0 / coefficients.numel())

    positive_count = positive.sum().clamp_min(1)
    negative_count = negative.sum().clamp_min(1)
    coefficients[positive] = 0.5 / positive_count
    coefficients[negative] = 0.5 / negative_count

    if mode in {"split_normalized", "frequency_positive", "topk_hard_negative"}:
        if mode in {"frequency_positive", "topk_hard_negative"}:
            frequencies = _load_label_frequencies(frequency_file).to(
                device=coefficients.device, dtype=coefficients.dtype
            )
            if frequencies.numel() != coefficients.size(1):
                raise ValueError(
                    "frequency file length does not match model label count: "
                    f"{frequencies.numel()} != {coefficients.size(1)}"
                )
            raw_weights = frequencies.clamp_min(1).pow(-frequency_gamma)
            raw_weights = raw_weights.clamp_max(frequency_weight_max)
            if frequency_weight_ratio_max is not None:
                if frequency_weight_ratio_max < 1:
                    raise ValueError("frequency_weight_ratio_max must be at least 1")
                relative_floor = raw_weights.max() / frequency_weight_ratio_max
                raw_weights = torch.maximum(raw_weights, relative_floor)
            positive_weights = raw_weights.unsqueeze(0).expand_as(coefficients)
            positive_weight_sum = positive_weights[positive].sum().clamp_min(1e-8)
            coefficients[positive] = (
                0.5 * positive_weights[positive] / positive_weight_sum
            )

        if mode == "topk_hard_negative":
            scores = torch.maximum(clean_probabilities, augmented_probabilities)
            scores = scores.masked_fill(positive, float("-inf"))
            k = min(max(int(hard_negative_k), 1), scores.size(1))
            selected = torch.zeros_like(negative)
            top_indices = scores.topk(k=k, dim=1).indices
            selected.scatter_(1, top_indices, True)
            selected &= negative
            selected_count = selected.sum().clamp_min(1)
            coefficients[negative] = 0
            coefficients[selected] = 0.5 / selected_count

        return coefficients

    raise ValueError(
        "Unknown paired consistency mode: "
        f"{mode!r}; expected uniform, split_normalized, frequency_positive, "
        "or topk_hard_negative"
    )


def _paired_consistency_term(
    clean_probabilities: torch.Tensor,
    augmented_probabilities: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    frequency_file: str,
    frequency_gamma: float,
    frequency_weight_max: float,
    hard_negative_k: int,
    frequency_weight_ratio_max: Optional[float] = None,
    divergence: str = "mse",
) -> torch.Tensor:
    """Compute B2-B5 consistency variants with a stable overall scale."""
    elementwise_divergence = _paired_consistency_divergence(
        clean_probabilities,
        augmented_probabilities,
        divergence=divergence,
    )
    coefficients = _paired_consistency_coefficients(
        clean_probabilities,
        augmented_probabilities,
        targets,
        mode=mode,
        frequency_file=frequency_file,
        frequency_gamma=frequency_gamma,
        frequency_weight_max=frequency_weight_max,
        hard_negative_k=hard_negative_k,
        frequency_weight_ratio_max=frequency_weight_ratio_max,
    )
    return (elementwise_divergence * coefficients).sum()


def _paired_consistency_divergence(
    clean_probabilities: torch.Tensor,
    augmented_probabilities: torch.Tensor,
    divergence: str,
) -> torch.Tensor:
    """Return a per-label divergence between two Bernoulli predictions."""
    if divergence == "mse":
        return (clean_probabilities - augmented_probabilities).square()
    if divergence == "symmetric_kl":
        eps = max(torch.finfo(clean_probabilities.dtype).eps, 1e-6)
        clean = clean_probabilities.clamp(min=eps, max=1.0 - eps)
        augmented = augmented_probabilities.clamp(min=eps, max=1.0 - eps)
        clean_to_augmented = clean * torch.log(clean / augmented) + (
            1.0 - clean
        ) * torch.log((1.0 - clean) / (1.0 - augmented))
        augmented_to_clean = augmented * torch.log(augmented / clean) + (
            1.0 - augmented
        ) * torch.log((1.0 - augmented) / (1.0 - clean))
        return 0.5 * (clean_to_augmented + augmented_to_clean)
    raise ValueError(
        f"Unknown paired consistency divergence: {divergence!r}; "
        "expected mse or symmetric_kl"
    )


def paired_supervised_consistency_loss(
    batch: Batch,
    model: torch.nn.Module,
    consistency_lambda: float = 0.0,
    consistency_mode: str = "uniform",
    consistency_divergence: str = "mse",
    frequency_file: str = "artifacts/frequency/label_frequency.csv",
    frequency_gamma: float = 0.5,
    frequency_weight_max: float = 5.0,
    frequency_weight_ratio_max: Optional[float] = None,
    hard_negative_k: int = 32,
    retention_lambda: float = 0.0,
    retention_threshold: float = 0.3333333333,
    retention_margin: float = 0.2,
    retention_warmup_epochs: int = 1,
    semantic_anchor_lambda: float = 0.0,
    epoch: int = 0,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[dict[str, torch.Tensor]]]:
    """Paired supervised objective used by B1-B5."""
    clean_logits = model(batch.input_ids, batch.attention_masks)
    clean_probabilities = torch.sigmoid(clean_logits)

    # Validation and inference splits intentionally contain no augmented view.
    if batch.augmented_input_ids is None:
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            clean_logits, batch.targets
        )
        return clean_probabilities, batch.targets, loss

    augmented_logits = model(
        batch.augmented_input_ids, batch.augmented_attention_masks
    )
    augmented_probabilities = torch.sigmoid(augmented_logits)
    clean_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        clean_logits, batch.targets
    )
    augmented_bce = torch.nn.functional.binary_cross_entropy_with_logits(
        augmented_logits, batch.targets
    )
    supervised_loss = 0.5 * (clean_bce + augmented_bce)
    consistency_coefficients = _paired_consistency_coefficients(
        clean_probabilities,
        augmented_probabilities,
        batch.targets,
        mode=consistency_mode,
        frequency_file=frequency_file,
        frequency_gamma=frequency_gamma,
        frequency_weight_max=frequency_weight_max,
        hard_negative_k=hard_negative_k,
        frequency_weight_ratio_max=frequency_weight_ratio_max,
    )
    consistency_loss = (
        _paired_consistency_divergence(
            clean_probabilities,
            augmented_probabilities,
            divergence=consistency_divergence,
        )
        * consistency_coefficients
    ).sum()

    # Protect clean true positives from falling below the positive boundary
    # on the augmented view. The clean branch is a detached teacher.
    retention_loss = augmented_logits.sum() * 0.0
    retention_selected = torch.zeros(
        (), device=augmented_logits.device, dtype=torch.long
    )
    if retention_lambda > 0 and epoch >= retention_warmup_epochs:
        if not 0 < retention_threshold < 1:
            raise ValueError("retention_threshold must lie in (0, 1)")
        clean_tp_mask = batch.targets.bool() & (
            clean_probabilities.detach() >= retention_threshold
        )
        retention_selected = clean_tp_mask.sum()
        if bool(clean_tp_mask.any()):
            boundary_logit = torch.as_tensor(
                math.log(retention_threshold / (1.0 - retention_threshold)),
                device=augmented_logits.device,
                dtype=augmented_logits.dtype,
            )
            violation = torch.relu(
                boundary_logit + retention_margin - augmented_logits
            )
            retention_loss = violation[clean_tp_mask].mean()

    effective_retention_lambda = (
        retention_lambda if epoch >= retention_warmup_epochs else 0.0
    )
    semantic_anchor = augmented_logits.sum() * 0.0
    if semantic_anchor_lambda > 0:
        from explainable_medical_coding.utils.semantic_queries import (
            semantic_query_anchor_loss,
        )

        semantic_anchor = semantic_query_anchor_loss(model)
    loss = (
        supervised_loss
        + consistency_lambda * consistency_loss
        + effective_retention_lambda * retention_loss
        + semantic_anchor_lambda * semantic_anchor
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("paired training loss is NaN or Inf")

    diagnostics = {
        "clean_probabilities": clean_probabilities.detach(),
        "augmented_probabilities": augmented_probabilities.detach(),
        "targets": batch.targets.detach(),
        "supervised_loss": supervised_loss.detach(),
        "consistency_loss": consistency_loss.detach(),
        "weighted_consistency_loss": (
            consistency_lambda * consistency_loss
        ).detach(),
        "retention_loss": retention_loss.detach(),
        "retention_selected": retention_selected.detach(),
        "semantic_anchor_loss": semantic_anchor.detach(),
        "weighted_semantic_anchor_loss": (
            semantic_anchor_lambda * semantic_anchor
        ).detach(),
        "objective_coefficients": (
            consistency_lambda * consistency_coefficients
        ).detach(),
        "total_loss": loss.detach(),
    }
    return clean_probabilities, batch.targets, loss, diagnostics


def tail_residual_supervised_ranking_loss(
    batch: Batch,
    model: torch.nn.Module,
    frequency_file: str,
    tail_max_frequency: int = 10,
    hard_negative_k: int = 32,
    negative_weight: float = 0.25,
    ranking_lambda: float = 0.0,
    ranking_margin: float = 0.2,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tail-positive/hard-negative supervision with optional cross-note ranking.

    This objective deliberately has no paired-view consistency term. With the
    E9 teacher frozen, gradients can only update the Tail residual branch.
    """
    if hard_negative_k <= 0:
        raise ValueError("hard_negative_k must be positive")
    if negative_weight < 0 or ranking_lambda < 0:
        raise ValueError("loss weights must be non-negative")

    logits = model(batch.input_ids, batch.attention_masks)
    probabilities = torch.sigmoid(logits)
    frequencies = _load_label_frequencies(frequency_file).to(logits.device)
    if frequencies.numel() != logits.size(1):
        raise ValueError(
            "frequency file length does not match model label count: "
            f"{frequencies.numel()} != {logits.size(1)}"
        )
    tail_mask = frequencies <= int(tail_max_frequency)
    tail_logits = logits[:, tail_mask]
    tail_targets = batch.targets[:, tail_mask]
    positive = tail_targets.bool()
    negative = ~positive
    elementwise = torch.nn.functional.binary_cross_entropy_with_logits(
        tail_logits, tail_targets, reduction="none"
    )

    zero = tail_logits.sum() * 0.0
    positive_loss = elementwise[positive].mean() if bool(positive.any()) else zero

    k = min(int(hard_negative_k), tail_logits.size(1))
    negative_scores = tail_logits.detach().masked_fill(~negative, float("-inf"))
    selected = torch.zeros_like(negative)
    selected.scatter_(1, negative_scores.topk(k=k, dim=1).indices, True)
    selected &= negative
    negative_loss = elementwise[selected].mean() if bool(selected.any()) else zero

    ranking_loss = zero
    if ranking_lambda > 0 and tail_logits.size(0) > 1 and bool(positive.any()):
        pair_losses = []
        positive_labels = torch.nonzero(
            positive.any(dim=0) & negative.any(dim=0), as_tuple=False
        ).flatten()
        for label_idx in positive_labels:
            positive_logits = tail_logits[positive[:, label_idx], label_idx]
            hardest_negative = tail_logits[negative[:, label_idx], label_idx].max()
            pair_losses.append(
                torch.nn.functional.softplus(
                    ranking_margin - positive_logits + hardest_negative
                )
            )
        if pair_losses:
            ranking_loss = torch.cat(pair_losses).mean()

    loss = (
        positive_loss
        + float(negative_weight) * negative_loss
        + float(ranking_lambda) * ranking_loss
    )
    if not torch.isfinite(loss):
        raise FloatingPointError("Tail residual loss is NaN or Inf")
    return probabilities, batch.targets, loss


def polarity_attention_loss(
    negative_attention: torch.Tensor,
    negative_gate: torch.Tensor,
    polarity_label_ids: Sequence[Sequence[int]],
    polarity_token_ids: Sequence[Sequence[Sequence[int]]],
) -> Optional[torch.Tensor]:
    """Penalize low opposing-evidence mass inside weakly supervised spans."""
    pair_losses = []
    sequence_length = negative_attention.size(2)
    for batch_idx, (label_ids, document_token_ids) in enumerate(
        zip(polarity_label_ids, polarity_token_ids)
    ):
        for label_id, token_ids in zip(label_ids, document_token_ids):
            label_id = int(label_id)
            valid_token_ids = [
                int(token_id)
                for token_id in token_ids
                if 0 <= int(token_id) < sequence_length
            ]
            if not valid_token_ids:
                continue
            evidence_mass = negative_gate[batch_idx, label_id] * negative_attention[
                batch_idx, label_id, valid_token_ids
            ].sum()
            pair_losses.append(-torch.log(evidence_mass.clamp_min(1e-8)))

    if not pair_losses:
        return None
    return torch.stack(pair_losses).mean()


def polarity_guided_binary_cross_entropy_loss(
    batch: Batch,
    model: torch.nn.Module,
    lambda_polarity: float = 5e-5,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Train DLEA with BCE plus explicit-negation attention guidance."""
    logits, components = model.forward_with_components(
        batch.input_ids,
        batch.attention_masks,
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        logits, batch.targets
    )

    if (
        model.training
        and lambda_polarity > 0
        and batch.polarity_label_ids is not None
        and batch.polarity_token_ids is not None
    ):
        auxiliary_loss = polarity_attention_loss(
            negative_attention=components["negative_attention"],
            negative_gate=components["negative_gate"],
            polarity_label_ids=batch.polarity_label_ids,
            polarity_token_ids=batch.polarity_token_ids,
        )
        if auxiliary_loss is not None:
            loss = loss + lambda_polarity * auxiliary_loss

    return torch.sigmoid(logits), batch.targets, loss


def kl_attention_loss(
    batch: Batch,
    model: torch.nn.Module,
    lambda_1: float = 2.5,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, targets, attention_masks, evidence_token_ids = (
        batch.input_ids,
        batch.targets,
        batch.attention_masks,
        batch.evidence_input_ids,
    )
    logits, label_wise_attentions = model(
        input_ids, attention_masks, output_attentions=True
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    if torch.isnan(loss):
        raise ValueError("Loss is NaN")

    if evidence_token_ids is None:
        return torch.sigmoid(logits), targets, loss

    sequence_length = label_wise_attentions.shape[2]
    batch_size = label_wise_attentions.shape[0]

    kl_loss = 0
    num_elements_with_evidence = 0
    for element_idx in range(batch_size):
        # If element has no evidence tokens, skip it
        if evidence_token_ids[element_idx] is None:
            continue

        target_ids = torch.where(targets[element_idx])[0]
        one_hot_evidence = one_hot_encode_evidence_token_ids(
            evidence_token_ids[element_idx], sequence_length
        ).to(input_ids.device)
        kl_loss += kl_divergence(
            label_wise_attentions[element_idx, target_ids], one_hot_evidence
        )
        num_elements_with_evidence += 1

    if num_elements_with_evidence > 0:
        loss += lambda_1 * kl_loss / (num_elements_with_evidence)
    return torch.sigmoid(logits), targets, loss


def double_backpropagation_loss(
    batch: Batch,
    model: torch.nn.Module,
    lambda_1: float = 1.0,
    p: int = 1,
    scale: int = 1024,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scale = 1024
    input_ids, targets, attention_masks = (
        batch.input_ids,
        batch.targets,
        batch.attention_masks,
    )
    logits = model(input_ids, attention_masks)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    if torch.isnan(loss):
        import pdb

        pdb.set_trace()
        raise ValueError("Loss is NaN")

    if loss.requires_grad:
        scaled_loss = loss * scale
        input_grad = torch.autograd.grad(
            scaled_loss, model.parameters(), create_graph=True
        )[0]
        input_grad = input_grad / scale
        loss_dbp = torch.norm(input_grad, p=2, dim=1).sum()
        if torch.isnan(loss_dbp):
            print("Double back propagation loss is NaN. Skipping regularization.")
        else:
            loss += lambda_1 * loss_dbp

    return torch.sigmoid(logits), targets, loss


def advesarial_training_loss(
    batch: Batch,
    model: torch.nn.Module,
    epsilon=1e-5,
    alpha=1,
    num_iter=5,
    lambda_1: float = 0.01,
    lambda_2: float = 0.5,
    scale: int = 1024,
    attack_type: str = "pgd",
    norm: str = "softmax",
    adv_dist: str = "kl",
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    input_ids, targets, attention_masks = (
        batch.input_ids,
        batch.targets,
        batch.attention_masks,
    )
    advesarial_model = deepcopy(model)
    advesarial_model.zero_grad()
    advesarial_model.eval()
    if attack_type == "fgsm":
        advesarial_noise = fgsm(
            advesarial_model,
            input_ids,
            attention_masks,
            targets,
            epsilon=epsilon,
            scale=scale,
        )
    elif attack_type == "pgd":
        advesarial_noise = pgd(
            advesarial_model,
            input_ids,
            attention_masks,
            targets,
            epsilon=epsilon,
            alpha=alpha,
            num_iter=num_iter,
            scale=scale,
        )
    elif attack_type == "pgd_adam":
        advesarial_noise = pgd_adam(
            advesarial_model,
            input_ids,
            attention_masks,
            targets,
            epsilon=epsilon,
            alpha=alpha,
            scale=scale,
        )
        scale = advesarial_noise_gradient_scaler.get_scale()

    logits = model(input_ids, attention_masks)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    if torch.isnan(loss):
        import pdb

        pdb.set_trace()
        raise ValueError("Loss is NaN")

    logits_adv = forward_embedding_noise(
        model, input_ids, attention_masks, advesarial_noise
    )

    # symetric kl divergence
    if adv_dist == "kl":
        if norm == "softmax":
            advesarial_loss = torch.nn.functional.kl_div(
                torch.log_softmax(logits_adv, dim=-1, dtype=torch.float32),
                torch.softmax(logits.detach(), dim=-1, dtype=torch.float32),
                reduction="batchmean",
            ) + torch.nn.functional.kl_div(
                torch.log_softmax(logits, dim=-1, dtype=torch.float32),
                torch.softmax(logits_adv.detach(), dim=-1, dtype=torch.float32),
                reduction="batchmean",
            )
        elif norm == "l1":
            logits_adv_norm = torch.nn.functional.normalize(logits_adv, p=1.0, dim=-1)
            logits_norm = torch.nn.functional.normalize(logits, p=1.0, dim=-1)
            advesarial_loss = torch.nn.functional.kl_div(
                torch.log(logits_adv_norm), logits_norm.detach(), reduction="batchmean"
            ) + torch.nn.functional.kl_div(
                torch.log(logits_norm), logits_adv_norm.detach(), reduction="batchmean"
            )
        else:
            raise ValueError(f"Norm {norm} not supported")

    else:
        advesarial_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits_adv, targets
        )

    # input gradient regularization

    if loss.requires_grad and (lambda_1 > 0):
        scaled_loss = loss * scale
        input_grad = torch.autograd.grad(
            scaled_loss, model.parameters(), create_graph=True
        )[0]
        input_grad = input_grad / scale
        loss_dbp = torch.norm(input_grad, p=2, dim=1).sum()

        if torch.isnan(loss_dbp):
            print("Double back propagation loss is NaN. Skipping regularization.")
        else:
            loss += lambda_1 * loss_dbp

    if torch.isnan(advesarial_loss):
        print("Advesarial loss is NaN. Skipping regularization.")
    else:
        loss += lambda_2 * advesarial_loss

    return torch.sigmoid(logits), targets, loss


def masking_loss(
    batch: Batch,
    model: torch.nn.Module,
    epoch: int = 0,
    total_epochs: int = 5,
    lambda_1: float = 1,
    diet_constant: float = 1,
    baseline_token_id: int = 50001,
    cls_token_id: int = 0,
    eos_token_id: int = 2,
    pad_token_id: int = 1,
    lr: float = 0.1,
    rounding_schedule: bool = True,
    distillation: bool = True,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    advesarial_model = deepcopy(model)
    advesarial_model.zero_grad()
    advesarial_model.eval()
    input_ids, targets, attention_masks, teacher_logits = (
        batch.input_ids,
        batch.targets,
        batch.attention_masks,
        batch.teacher_logits,
    )
    baseline = create_baseline_input(
        input_ids,
        baseline_token_id=baseline_token_id,
        cls_token_id=baseline_token_id,
        eos_token_id=baseline_token_id,
    )
    baseline = baseline.requires_grad_(False)
    baseline_embeddings = model.get_chunked_embedding(baseline).detach()
    mask = token_masking(
        advesarial_model,
        input_ids,
        attention_masks,
        baseline_embeddings,
        diet_constant=diet_constant,
        lr=lr,
    )
    mask = mask.requires_grad_(False)
    mask = mask.view(input_ids.size(0), -1, 1)[:, : input_ids.size(1)].squeeze(-1)

    if rounding_schedule:
        rounding_constant = max(0.4 - epoch * (0.4 / total_epochs), 0)
    else:
        rounding_constant = 0

    mask = torch.round(mask + rounding_constant)
    masked_input_ids = (input_ids * mask + baseline_token_id * (1 - mask)).to(
        torch.long
    )
    masked_input_ids[:, 0] = cls_token_id
    masked_input_ids[torch.where(input_ids == eos_token_id)] = eos_token_id
    masked_input_ids[torch.where(input_ids == pad_token_id)] = pad_token_id

    student_logits = model(input_ids, attention_masks)
    student_probs = torch.sigmoid(student_logits)

    teacher_probs = torch.sigmoid(teacher_logits)

    masked_student_logits = model(masked_input_ids, attention_masks)
    masked_student_probs = torch.sigmoid(masked_student_logits)

    batch_size = input_ids.size(0)
    if distillation:
        model_loss = (
            torch.norm(student_probs - teacher_probs.detach(), p=1) / batch_size
        )
        masked_loss = (
            torch.norm(student_probs.detach() - masked_student_probs, p=1) / batch_size
        )
    else:
        model_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            student_logits, targets
        )
        masked_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            masked_student_logits, targets
        )

    if torch.isnan(masked_loss):
        print("Diet loss is NaN. Skipping regularization.")
        loss = model_loss
    else:
        loss = model_loss + lambda_1 * masked_loss

    if torch.isnan(loss):
        import pdb

        pdb.set_trace()
        raise ValueError("Loss is NaN")

    return student_probs, targets, loss


@torch.enable_grad()
def pgd(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    alpha: float,
    num_iter: int,
    scale: int = 1024,
):
    """Construct FGSM adversarial examples on the examples X"""
    chunked_embeddings = model.get_chunked_embedding(input_ids).detach()

    # delta = torch.zeros_like(chunked_embeddings, requires_grad=True)
    delta = torch.rand_like(chunked_embeddings, requires_grad=True)
    delta.data = delta.data * 2 * epsilon - epsilon

    logits = model.forward_embedding_input(
        chunked_embeddings, attention_masks=attention_mask
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    for t in range(num_iter):
        logits = model.forward_embedding_input(
            chunked_embeddings + torch.tanh(delta) * epsilon,
            attention_masks=attention_mask,
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        print(loss)
        loss = loss * scale
        loss.backward()
        delta.data = delta + chunked_embeddings.shape[0] * alpha * delta.grad.data / (
            scale * delta.grad.data.max(dim=-1, keepdim=True).values
        )
        model.zero_grad()
        delta.grad.zero_()
    return torch.tanh(delta.detach()) * epsilon


@torch.enable_grad()
def token_masking(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    baseline_embeddings: torch.Tensor,
    diet_constant: float = 0.1,
    lr: float = 0.1,
    max_iter: int = 20,
):
    """Construct FGSM adversarial examples on the examples X"""
    prev_loss, prev_prev_loss = float("inf"), float("inf")
    mask_converged = False
    is_training = model.training

    if is_training:
        model.eval()

    chunked_embeddings = model.get_chunked_embedding(input_ids).detach()

    original_logits = model.forward_embedding_input(
        chunked_embeddings, attention_masks=attention_mask
    )

    original_y_probs = torch.sigmoid(original_logits)
    # delta = torch.zeros_like(chunked_embeddings, requires_grad=True)
    mask = torch.ones(
        (chunked_embeddings.size(0), chunked_embeddings.size(1), 1),
        requires_grad=True,
        device=chunked_embeddings.device,
    )
    optimizer = torch.optim.AdamW([mask], lr=lr)
    mask.data = mask.data * 2

    for idx in range(max_iter):
        if mask_converged:
            break
        sigmoid_mask = torch.sigmoid(mask)
        input_embeddings = chunked_embeddings * sigmoid_mask + baseline_embeddings * (
            1 - sigmoid_mask
        )
        logits = model.forward_embedding_input(
            input_embeddings, attention_masks=attention_mask
        )

        y_probs = torch.sigmoid(logits)
        loss = torch.norm(sigmoid_mask, p=1) / (
            mask.size(0) * mask.size(1)
        ) + diet_constant * torch.norm(
            y_probs - original_y_probs.detach(), p=1
        ) / logits.size(0)
        mask_converged = (loss > 0.97 * prev_prev_loss) and (
            loss < 1.03 * prev_prev_loss
        )

        prev_prev_loss = prev_loss
        prev_loss = loss

        loss = diet_gradient_scaler.scale(loss)
        loss.backward()
        diet_gradient_scaler.step(optimizer)
        diet_gradient_scaler.update()

        optimizer.zero_grad()
        model.zero_grad()

    if is_training:
        model.train()

    return torch.sigmoid(mask.detach())


@torch.enable_grad()
def pgd_attack(
    input_ids,
    attention_mask,
    targets,
    model,
    epsilon=0.01,
    alpha=1,
    num_iter=5,
    scale=1024,
):
    noise = pgd(
        model,
        input_ids,
        attention_mask,
        targets,
        epsilon=epsilon,
        alpha=alpha,
        num_iter=num_iter,
        scale=scale,
    )
    chunked_embeddings = model.get_chunked_embedding(input_ids).detach()
    logits = model.forward_embedding_input(
        chunked_embeddings + noise, attention_masks=attention_mask
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    return torch.sigmoid(logits).detach(), loss.detach()


@torch.enable_grad()
def pgd_adam(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    alpha: float,
    scale: int = 1024,
):
    is_training = model.training
    if is_training:
        model.eval()
    """Construct FGSM adversarial examples on the examples X"""
    chunked_embeddings = model.get_chunked_embedding(input_ids).detach()

    # delta = torch.zeros_like(chunked_embeddings, requires_grad=True)
    delta = torch.rand_like(chunked_embeddings, requires_grad=True)
    delta.data = delta.data * 2 * epsilon - epsilon
    optimizer = torch.optim.AdamW([delta], lr=alpha)
    prev_loss, prev_prev_loss = float("inf"), float("inf")
    mask_converged = False
    for _ in range(100):
        if mask_converged:
            break
        optimizer.zero_grad()
        logits = model.forward_embedding_input(
            chunked_embeddings + torch.tanh(delta) * epsilon,
            attention_masks=attention_mask,
        )
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        mask_converged = (loss > 0.995 * prev_prev_loss) and (
            loss < 1.005 * prev_prev_loss
        )
        prev_prev_loss = prev_loss
        prev_loss = loss

        loss = advesarial_noise_gradient_scaler.scale(-loss)
        loss.backward()
        advesarial_noise_gradient_scaler.step(optimizer)
        advesarial_noise_gradient_scaler.update()
        model.zero_grad()
    if is_training:
        model.train()
    return torch.tanh(delta.detach()) * epsilon


@torch.enable_grad()
def fgsm(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    targets: torch.Tensor,
    epsilon: float,
    scale: int = 1024,
):
    """Construct FGSM adversarial examples on the examples X"""
    chunked_embeddings = model.get_chunked_embedding(input_ids).detach()

    delta = torch.zeros_like(chunked_embeddings, requires_grad=True)
    # delta = torch.rand_like(chunked_embeddings, requires_grad=True)
    # delta.data = delta.data * 2 * epsilon - epsilon
    logits = model.forward_embedding_input(
        chunked_embeddings + delta, attention_masks=attention_mask
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
    loss = loss * scale
    loss.backward()
    delta = epsilon * delta.grad.detach().sign()
    logits = model.forward_embedding_input(
        chunked_embeddings + delta, attention_masks=attention_mask
    )
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)

    return delta

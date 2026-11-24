import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SemanticQueryReport:
    total_labels: int
    description_labels: int
    synonym_labels: int
    initialized_labels: int
    preserved_labels: int
    query_norm_before: float
    query_norm_after: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


@dataclass(frozen=True)
class SemanticAdapterReport:
    total_labels: int
    description_labels: int
    synonym_labels: int
    hierarchy_labels: int
    initialized_labels: int
    missing_labels: int

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


def _core_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def get_label_queries(model: torch.nn.Module) -> torch.nn.Parameter:
    attention = getattr(_core_model(model), "label_wise_attention", None)
    queries = getattr(attention, "label_representations", None)
    if not isinstance(queries, torch.nn.Parameter):
        raise TypeError(
            "Semantic queries require LabelCrossAttention.label_representations"
        )
    return queries


def _load_descriptions(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8") as handle:
        return {
            row["code"].strip(): row["desc"].strip()
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("code") and row.get("desc") and row["desc"].strip()
        }


def _load_synonyms(path: Optional[Path]) -> dict[str, list[str]]:
    if path is None:
        return {}
    with path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    return {
        str(code): [str(item) for item in values]
        for code, values in raw.items()
        if isinstance(values, list)
    }


def _load_hierarchy_descriptions(
    relation_path: Path,
    group_path: Path,
    extra_description_path: Path,
) -> dict[str, str]:
    """Reproduce GKI-ICD's group/parent hierarchy text for each ICD-9 code."""
    with group_path.open(encoding="utf-8") as handle:
        group_names = {
            row["group"].strip(): row["name"].strip()
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("group") and row.get("name")
        }
    with extra_description_path.open(encoding="utf-8") as handle:
        extra_descriptions = {
            row["code"].strip(): row["desc"].strip()
            for row in csv.DictReader(handle, delimiter="\t")
            if row.get("code") and row.get("desc")
        }

    hierarchy = {}
    with relation_path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            code = (row.get("code") or "").strip()
            if not code:
                continue
            code_before_decimal = code.split(".")[0]
            group_keys = (
                [(row.get("proc") or "").strip()]
                if len(code_before_decimal) == 2
                else [
                    (row.get("diag1") or "").strip(),
                    (row.get("diag2") or "").strip(),
                ]
            )
            parts = []
            for key in group_keys:
                value = group_names.get(key)
                if value and value not in parts:
                    parts.append(value)
            if code != code_before_decimal:
                parent_description = extra_descriptions.get(code_before_decimal)
                if parent_description and parent_description not in parts:
                    parts.append(parent_description)
            if parts:
                hierarchy[code] = ", ".join(parts)
    return hierarchy


def _select_synonyms(
    values: Iterable[str], description: str, max_synonyms: int
) -> list[str]:
    selected = []
    seen = {description.strip().casefold()}
    for value in values:
        synonym = " ".join(value.split())
        normalized = synonym.casefold()
        if not synonym or normalized in seen:
            continue
        if normalized.startswith("synonym deleted refer to"):
            continue
        seen.add(normalized)
        selected.append(synonym)
        if len(selected) >= max_synonyms:
            break
    return selected


@torch.no_grad()
def _encode_texts(
    encoder: torch.nn.Module,
    tokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    if not texts:
        hidden_size = int(encoder.config.hidden_size)
        return torch.empty((0, hidden_size), device=device, dtype=torch.float32)

    was_training = encoder.training
    encoder.eval()
    pooled_batches = []
    for start in range(0, len(texts), batch_size):
        tokenized = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        hidden = encoder(**tokenized).last_hidden_state
        mask = tokenized["attention_mask"].bool().unsqueeze(-1)
        pooled = hidden.masked_fill(~mask, torch.finfo(hidden.dtype).min).max(dim=1)[0]
        pooled_batches.append(pooled.float())
    encoder.train(was_training)
    return torch.cat(pooled_batches, dim=0)


@torch.no_grad()
def _encode_texts_mean(
    encoder: torch.nn.Module,
    tokenizer,
    texts: list[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    """Encode one semantic fragment at a time with masked token mean pooling."""
    if not texts:
        hidden_size = int(encoder.config.hidden_size)
        return torch.empty((0, hidden_size), device=device, dtype=torch.float32)

    was_training = encoder.training
    encoder.eval()
    pooled_batches = []
    for start in range(0, len(texts), batch_size):
        tokenized = tokenizer(
            texts[start : start + batch_size],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        tokenized = {key: value.to(device) for key, value in tokenized.items()}
        hidden = encoder(**tokenized).last_hidden_state.float()
        mask = tokenized["attention_mask"].float().unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled_batches.append(pooled)
    encoder.train(was_training)
    return torch.cat(pooled_batches, dim=0)


@torch.no_grad()
def initialize_semantic_query_adapter(
    model: torch.nn.Module,
    tokenizer,
    target_names: list[str],
    description_path: str | Path,
    synonym_path: str | Path,
    hierarchy_relation_path: str | Path,
    hierarchy_group_path: str | Path,
    hierarchy_extra_description_path: str | Path,
    max_synonyms: int = 5,
    batch_size: int = 128,
    max_length: int = 128,
) -> SemanticAdapterReport:
    """Encode description, synonym and hierarchy fragments for the query adapter."""
    if max_synonyms <= 0 or batch_size <= 0 or max_length <= 0:
        raise ValueError("max_synonyms, batch_size, and max_length must be positive")

    core = _core_model(model)
    attention = getattr(core, "label_wise_attention", None)
    if attention is None or not getattr(attention, "semantic_query_adapter", False):
        raise TypeError("Model was not constructed with semantic_query_adapter=true")
    if attention.semantic_fragment_embeddings.size(0) != len(target_names):
        raise ValueError("Target tokenizer order does not match semantic adapter")
    if attention.semantic_fragment_embeddings.size(1) != 3:
        raise ValueError("Semantic adapter requires exactly three fragment types")

    descriptions = _load_descriptions(Path(description_path))
    synonyms = _load_synonyms(Path(synonym_path))
    hierarchy = _load_hierarchy_descriptions(
        Path(hierarchy_relation_path),
        Path(hierarchy_group_path),
        Path(hierarchy_extra_description_path),
    )
    fragment_maps = [
        descriptions,
        {
            code: " ; ".join(
                _select_synonyms(
                    values,
                    descriptions.get(code, ""),
                    max_synonyms,
                )
            )
            for code, values in synonyms.items()
        },
        hierarchy,
    ]

    device = attention.label_representations.device
    hidden_size = attention.label_representations.size(1)
    fragment_embeddings = torch.zeros(
        len(target_names), 3, hidden_size, device=device, dtype=torch.float32
    )
    fragment_mask = torch.zeros(
        len(target_names), 3, device=device, dtype=torch.bool
    )
    fragment_counts = []
    for fragment_index, fragment_map in enumerate(fragment_maps):
        indices = [
            index
            for index, code in enumerate(target_names)
            if fragment_map.get(code, "").strip()
        ]
        texts = [fragment_map[target_names[index]] for index in indices]
        encoded = _encode_texts_mean(
            core.roberta_encoder,
            tokenizer,
            texts,
            batch_size,
            max_length,
            device,
        )
        if indices:
            index_tensor = torch.tensor(indices, device=device, dtype=torch.long)
            fragment_embeddings[index_tensor, fragment_index] = encoded
            fragment_mask[index_tensor, fragment_index] = True
        fragment_counts.append(len(indices))

    attention.set_semantic_fragments(fragment_embeddings, fragment_mask)
    initialized = int(fragment_mask.any(dim=1).sum().item())
    return SemanticAdapterReport(
        total_labels=len(target_names),
        description_labels=fragment_counts[0],
        synonym_labels=fragment_counts[1],
        hierarchy_labels=fragment_counts[2],
        initialized_labels=initialized,
        missing_labels=len(target_names) - initialized,
    )


def _frequency_anchor_weights(
    frequencies: Optional[torch.Tensor],
    mask: torch.Tensor,
    gamma: float,
    weight_max: float,
) -> torch.Tensor:
    if not bool(mask.any()):
        raise ValueError("No ICD descriptions matched the target tokenizer")
    if frequencies is None or gamma == 0:
        return torch.ones(mask.numel(), device=mask.device, dtype=torch.float32)
    if frequencies.numel() != mask.numel():
        raise ValueError("Label frequencies and semantic query mask must align")
    frequency = frequencies.to(device=mask.device, dtype=torch.float32).clamp_min(1)
    weights = (frequency.max() / frequency).pow(gamma).clamp(max=weight_max)
    weights = weights / weights[mask].mean().clamp_min(1e-12)
    return weights


def _set_nonpersistent_buffer(
    model: torch.nn.Module, name: str, tensor: torch.Tensor
) -> None:
    core = _core_model(model)
    if name in core._buffers:
        core._buffers[name] = tensor
        core._non_persistent_buffers_set.add(name)
    else:
        core.register_buffer(name, tensor, persistent=False)


@torch.no_grad()
def initialize_semantic_label_queries(
    model: torch.nn.Module,
    tokenizer,
    target_names: list[str],
    description_path: str | Path,
    synonym_path: str | Path | None = None,
    use_synonyms: bool = False,
    synonym_weight: float = 0.3,
    max_synonyms: int = 5,
    batch_size: int = 128,
    max_length: int = 128,
    label_frequencies: Optional[torch.Tensor] = None,
    frequency_gamma: float = 0.5,
    frequency_weight_max: float = 5.0,
) -> SemanticQueryReport:
    if not 0 <= synonym_weight < 1:
        raise ValueError("synonym_weight must lie in [0, 1)")
    if max_synonyms <= 0 or batch_size <= 0 or max_length <= 0:
        raise ValueError("max_synonyms, batch_size, and max_length must be positive")

    queries = get_label_queries(model)
    if queries.size(0) != len(target_names):
        raise ValueError("Target tokenizer order does not match the label query matrix")
    device = queries.device
    encoder = _core_model(model).roberta_encoder
    descriptions = _load_descriptions(Path(description_path))
    synonyms = _load_synonyms(Path(synonym_path) if synonym_path else None)

    description_indices = [
        index for index, code in enumerate(target_names) if descriptions.get(code)
    ]
    description_texts = [descriptions[target_names[index]] for index in description_indices]
    description_features = _encode_texts(
        encoder, tokenizer, description_texts, batch_size, max_length, device
    )

    semantic_directions = F.normalize(queries.detach().float(), dim=1)
    semantic_mask = torch.zeros(len(target_names), device=device, dtype=torch.bool)
    if description_indices:
        index_tensor = torch.tensor(description_indices, device=device, dtype=torch.long)
        semantic_directions[index_tensor] = F.normalize(description_features, dim=1)
        semantic_mask[index_tensor] = True

    synonym_label_indices = []
    synonym_texts = []
    synonym_label_coverage = set()
    if use_synonyms:
        for label_index in description_indices:
            code = target_names[label_index]
            selected = _select_synonyms(
                synonyms.get(code, []), descriptions[code], max_synonyms
            )
            if selected:
                synonym_label_coverage.add(label_index)
            synonym_label_indices.extend([label_index] * len(selected))
            synonym_texts.extend(selected)
        synonym_features = _encode_texts(
            encoder, tokenizer, synonym_texts, batch_size, max_length, device
        )
        if synonym_texts:
            synonym_sums = torch.zeros_like(semantic_directions)
            synonym_counts = torch.zeros(
                len(target_names), device=device, dtype=torch.float32
            )
            synonym_index_tensor = torch.tensor(
                synonym_label_indices, device=device, dtype=torch.long
            )
            synonym_sums.index_add_(
                0, synonym_index_tensor, F.normalize(synonym_features, dim=1)
            )
            synonym_counts.index_add_(
                0,
                synonym_index_tensor,
                torch.ones(len(synonym_texts), device=device),
            )
            synonym_mask = synonym_counts > 0
            synonym_directions = F.normalize(
                synonym_sums / synonym_counts.clamp_min(1).unsqueeze(1), dim=1
            )
            semantic_directions[synonym_mask] = F.normalize(
                (1.0 - synonym_weight) * semantic_directions[synonym_mask]
                + synonym_weight * synonym_directions[synonym_mask],
                dim=1,
            )

    norms_before = queries.detach().float().norm(dim=1).clamp_min(1e-6)
    scaled_semantics = semantic_directions * norms_before.unsqueeze(1)
    queries[semantic_mask] = scaled_semantics[semantic_mask].to(dtype=queries.dtype)
    norms_after = queries.detach().float().norm(dim=1)

    anchor_weights = _frequency_anchor_weights(
        label_frequencies,
        semantic_mask,
        gamma=frequency_gamma,
        weight_max=frequency_weight_max,
    )
    _set_nonpersistent_buffer(
        model, "semantic_query_targets", semantic_directions.detach()
    )
    _set_nonpersistent_buffer(model, "semantic_query_mask", semantic_mask)
    _set_nonpersistent_buffer(model, "semantic_query_weights", anchor_weights)

    return SemanticQueryReport(
        total_labels=len(target_names),
        description_labels=len(description_indices),
        synonym_labels=len(synonym_label_coverage),
        initialized_labels=int(semantic_mask.sum().item()),
        preserved_labels=int((~semantic_mask).sum().item()),
        query_norm_before=float(norms_before.mean().item()),
        query_norm_after=float(norms_after.mean().item()),
    )


def semantic_query_anchor_loss(model: torch.nn.Module) -> torch.Tensor:
    core = _core_model(model)
    for name in (
        "semantic_query_targets",
        "semantic_query_mask",
        "semantic_query_weights",
    ):
        if not hasattr(core, name):
            raise RuntimeError("Semantic query anchoring was not initialized")
    queries = get_label_queries(model).float()
    cosine_distance = 1.0 - F.cosine_similarity(
        queries, core.semantic_query_targets.float(), dim=1
    )
    mask = core.semantic_query_mask
    weights = core.semantic_query_weights.float()
    return (cosine_distance[mask] * weights[mask]).sum() / weights[mask].sum()

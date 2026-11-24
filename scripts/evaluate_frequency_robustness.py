#!/usr/bin/env python3
"""Evaluate PLM-ICD robustness by training-label frequency.

The script deliberately keeps perturbation generation, inference, and statistical
analysis in one place so the experiment can be reproduced from a checkpoint and
the processed parquet files alone.
"""

from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
import polars as pl
import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer

import explainable_medical_coding.config.factories as factories
from explainable_medical_coding.evidence.section_parser import parse_sections
from explainable_medical_coding.utils.tokenizer import (
    TargetTokenizer,
    tokenize_texts_with_chunk_special_tokens,
)


CONDITIONS = (
    "clean",
    "spelling",
    "format",
    "section_reorder",
    "protected_delete",
    "reorder",
    "delete",
)
DEFAULT_SEVERITIES = {
    "clean": 0.0,
    "spelling": 0.05,
    "format": 0.10,
    "section_reorder": 1.0,
    "protected_delete": 0.10,
    "reorder": 1.0,
    "delete": 0.10,
}
FORMAL_SEVERITY_LEVELS = {
    "spelling": (0.01, 0.03, 0.05),
    "format": (0.05, 0.10, 0.20),
    "section_reorder": (1 / 3, 2 / 3, 1.0),
    "protected_delete": (0.05, 0.10, 0.20),
}
PROTECTED_SECTION_NAMES = {
    "DISCHARGE_DIAGNOSIS",
    "FINAL_DIAGNOSIS",
    "PRINCIPAL_DIAGNOSIS",
    "PROCEDURES",
    "MAJOR_PROCEDURES",
    "MAJOR_SURGICAL_OR_INVASIVE_PROCEDURE",
    "PRINCIPAL_PROCEDURE",
}
WORD_RE = re.compile(r"\b[A-Za-z]{4,}\b")
SEPARATOR_RE = re.compile(r"((?:\r?\n)+|(?<=[.!?])(?:[ \t]+))")
WHITESPACE_RE = re.compile(r"(?:\r?\n)+|[ \t]+")
PARAGRAPH_SEPARATOR_RE = re.compile(r"((?:\r?\n[ \t]*){2,})")
STRUCTURED_LINE_RE = re.compile(r"^\s*(?:[-*#]\s+|\d+[.)]\s+)", re.MULTILINE)
SENTENCE_END_RE = re.compile(r"[.!?]+(?:[\"')\]]*)")
SENTENCE_WORD_RE = re.compile(r"\b[A-Za-z]+\b")
NON_TERMINAL_ABBREVIATIONS = {
    "approx",
    "dr",
    "e.g",
    "etc",
    "fig",
    "i.e",
    "mr",
    "mrs",
    "ms",
    "no",
    "prof",
    "st",
    "vs",
}


def stable_rng(seed: int, condition: str, record_id: object) -> random.Random:
    payload = f"{seed}|{condition}|{record_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "little"))


def split_sentence_like_units(text: str) -> tuple[list[str], list[str]]:
    """Split text while retaining the exact separators between units."""
    parts = SEPARATOR_RE.split(text)
    units: list[str] = []
    separators: list[str] = []
    for index in range(0, len(parts), 2):
        units.append(parts[index])
        separators.append(parts[index + 1] if index + 1 < len(parts) else "")
    nonempty = [idx for idx, unit in enumerate(units) if unit.strip()]
    if len(nonempty) < 2:
        # Fall back to coarse clauses for notes with no punctuation/newlines.
        clauses = re.split(r"(?<=[;:])\s+", text)
        if len(clauses) >= 2:
            return clauses, [" "] * (len(clauses) - 1) + [""]
    return units, separators


def misspell_word(word: str, rng: random.Random) -> str:
    chars = list(word)
    operation = rng.choice(("swap", "delete", "duplicate", "substitute"))
    # Prefer internal characters so the corruption resembles a typo rather than
    # changing token boundaries or capitalization conventions.
    position = rng.randrange(1, len(chars) - 1)
    if operation == "swap":
        chars[position], chars[position + 1] = chars[position + 1], chars[position]
    elif operation == "delete":
        del chars[position]
    elif operation == "duplicate":
        chars.insert(position, chars[position])
    else:
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        replacement = rng.choice(alphabet)
        if chars[position].isupper():
            replacement = replacement.upper()
        chars[position] = replacement
    return "".join(chars)


def perturb_spelling(text: str, severity: float, rng: random.Random) -> str:
    matches = list(WORD_RE.finditer(text))
    if not matches or severity <= 0:
        return text
    count = min(len(matches), max(1, int(round(severity * len(matches)))))
    chosen = set(rng.sample(range(len(matches)), count))
    output: list[str] = []
    cursor = 0
    for index, match in enumerate(matches):
        output.append(text[cursor : match.start()])
        word = match.group(0)
        output.append(misspell_word(word, rng) if index in chosen else word)
        cursor = match.end()
    output.append(text[cursor:])
    return "".join(output)


def perturb_format(text: str, severity: float, rng: random.Random) -> str:
    """Corrupt only whitespace, leaving every non-whitespace character intact."""
    matches = list(WHITESPACE_RE.finditer(text))
    if not matches or severity <= 0:
        return text
    count = min(len(matches), max(1, int(round(severity * len(matches)))))
    chosen = set(rng.sample(range(len(matches)), count))
    output: list[str] = []
    cursor = 0
    for index, match in enumerate(matches):
        output.append(text[cursor : match.start()])
        span = match.group(0)
        if index not in chosen:
            replacement = span
        elif "\n" in span or "\r" in span:
            replacement = rng.choice((" ", "\n\n"))
        else:
            replacement = rng.choice(("\t", "  "))
        output.append(replacement)
        cursor = match.end()
    output.append(text[cursor:])
    return "".join(output)


def perturb_reorder(text: str, severity: float, rng: random.Random) -> str:
    units, separators = split_sentence_like_units(text)
    candidates = [idx for idx, unit in enumerate(units) if unit.strip()]
    if len(candidates) < 2 or severity <= 0:
        return text
    count = min(len(candidates), max(2, int(math.ceil(severity * len(candidates)))))
    selected = rng.sample(candidates, count)
    shuffled = [units[idx] for idx in selected]
    # Avoid a no-op permutation, which matters for very short notes.
    for _ in range(8):
        rng.shuffle(shuffled)
        if any(units[idx] != value for idx, value in zip(selected, shuffled)):
            break
    reordered = list(units)
    for idx, value in zip(selected, shuffled):
        reordered[idx] = value
    return "".join(unit + separator for unit, separator in zip(reordered, separators))


def _is_complete_sentence_boundary(text: str, start: int, end: int) -> bool:
    punctuation_start = end - 1
    while punctuation_start >= start and text[punctuation_start] in "\"')].!?":
        punctuation_start -= 1
    token_match = re.search(r"([A-Za-z]+)\W*$", text[start : punctuation_start + 1])
    token = token_match.group(1).lower() if token_match else ""
    if token in NON_TERMINAL_ABBREVIATIONS or (len(token) == 1 and token.isalpha()):
        return False
    if punctuation_start >= 0 and text[punctuation_start].isdigit():
        return False
    cursor = end
    while cursor < len(text) and text[cursor].isspace():
        cursor += 1
    if cursor == end:
        return cursor == len(text)
    if cursor == len(text):
        return True
    return text[cursor].isupper() or text[cursor].isdigit() or text[cursor] in "[#"


def complete_sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return conservative complete-sentence spans without splitting hard wraps."""
    if STRUCTURED_LINE_RE.search(text):
        return []
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text) and text[start].isspace():
        start += 1
    for match in SENTENCE_END_RE.finditer(text):
        end = match.end()
        if not _is_complete_sentence_boundary(text, start, end):
            continue
        candidate = text[start:end]
        if len(SENTENCE_WORD_RE.findall(candidate)) >= 3:
            spans.append((start, end))
        start = end
        while start < len(text) and text[start].isspace():
            start += 1
    return spans


def _complete_sentence_groups(body: str) -> list[list[tuple[int, int]]]:
    groups: list[list[tuple[int, int]]] = []
    offset = 0
    for part in PARAGRAPH_SEPARATOR_RE.split(body):
        if not part or PARAGRAPH_SEPARATOR_RE.fullmatch(part):
            offset += len(part)
            continue
        spans = [
            (offset + start, offset + end)
            for start, end in complete_sentence_spans(part)
        ]
        if len(spans) >= 4:
            groups.append(spans)
        offset += len(part)
    return groups


def _reorder_section_body(body: str, rng: random.Random) -> str:
    replacements: list[tuple[int, int, str]] = []
    for spans in _complete_sentence_groups(body):
        sentences = [body[start:end] for start, end in spans]
        shuffled = list(sentences)
        for _ in range(8):
            rng.shuffle(shuffled)
            if shuffled != sentences:
                break
        replacements.extend(
            (start, end, value) for (start, end), value in zip(spans, shuffled)
        )
    return _replace_spans(body, replacements) if replacements else body


def _replace_spans(text: str, replacements: list[tuple[int, int, str]]) -> str:
    output: list[str] = []
    cursor = 0
    for start, end, replacement in sorted(replacements):
        if start < cursor:
            raise ValueError("replacement spans overlap")
        output.append(text[cursor:start])
        output.append(replacement)
        cursor = end
    output.append(text[cursor:])
    return "".join(output)


def perturb_section_reorder(text: str, severity: float, rng: random.Random) -> str:
    """Reorder sentences only inside eligible sections, preserving all headers."""
    eligible = []
    for section in parse_sections(text):
        if section.name == "UNKNOWN":
            continue
        body = text[section.start : section.end]
        if _complete_sentence_groups(body):
            eligible.append(section)
    if not eligible or severity <= 0:
        return text
    selected_count = min(
        len(eligible), max(1, int(math.ceil(severity * len(eligible))))
    )
    selected = rng.sample(eligible, selected_count)
    replacements = []
    for section in selected:
        body = text[section.start : section.end]
        replacements.append(
            (section.start, section.end, _reorder_section_body(body, rng))
        )
    return _replace_spans(text, replacements)


def perturb_delete(text: str, severity: float, rng: random.Random) -> str:
    units, separators = split_sentence_like_units(text)
    candidates = [idx for idx, unit in enumerate(units) if unit.strip()]
    if len(candidates) < 2 or severity <= 0:
        return text
    count = min(len(candidates) - 1, max(1, int(round(severity * len(candidates)))))
    deleted = set(rng.sample(candidates, count))
    return "".join(
        unit + separator
        for idx, (unit, separator) in enumerate(zip(units, separators))
        if idx not in deleted
    )


def perturb_protected_delete(text: str, severity: float, rng: random.Random) -> str:
    """Delete sentence-like units outside diagnosis and procedure sections."""
    candidates: list[tuple[int, int]] = []
    sections = parse_sections(text)
    for section in sections:
        if section.name in PROTECTED_SECTION_NAMES:
            continue
        body = text[section.start : section.end]
        units, _ = split_sentence_like_units(body)
        candidates.extend(
            (section.start, idx) for idx, unit in enumerate(units) if unit.strip()
        )
    if len(candidates) < 2 or severity <= 0:
        return text
    delete_count = min(
        len(candidates) - 1, max(1, int(round(severity * len(candidates))))
    )
    deleted = set(rng.sample(candidates, delete_count))
    replacements = []
    for section in sections:
        if section.name in PROTECTED_SECTION_NAMES:
            continue
        body = text[section.start : section.end]
        units, separators = split_sentence_like_units(body)
        if not any((section.start, idx) in deleted for idx in range(len(units))):
            continue
        replacement = "".join(
            unit + separators[idx]
            for idx, unit in enumerate(units)
            if (section.start, idx) not in deleted
        )
        replacements.append((section.start, section.end, replacement))
    return _replace_spans(text, replacements)


PERTURBATIONS: dict[str, Callable[[str, float, random.Random], str]] = {
    "spelling": perturb_spelling,
    "format": perturb_format,
    "section_reorder": perturb_section_reorder,
    "protected_delete": perturb_protected_delete,
    "reorder": perturb_reorder,
    "delete": perturb_delete,
}


def perturb_text(
    text: str,
    condition: str,
    severity: float,
    seed: int,
    record_id: object,
) -> str:
    if condition == "clean":
        return text
    return PERTURBATIONS[condition](
        text, severity, stable_rng(seed, condition, record_id)
    )


def load_model_on_device(
    checkpoint_dir: Path,
    config: OmegaConf,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.nn.Module, float]:
    """Load optimizer-heavy checkpoints on CPU before moving only the model."""
    checkpoint = torch.load(
        checkpoint_dir / "best_model.pt", map_location="cpu", weights_only=False
    )
    model = factories.get_model(
        config.model,
        {"num_classes": checkpoint["num_classes"], "pad_token_id": pad_token_id},
    )
    state = checkpoint["model"]
    if state and next(iter(state)).startswith("_orig_mod."):
        state = {key.removeprefix("_orig_mod."): value for key, value in state.items()}
    # E5/E6 predate the persistent frequency-gate buffer.  Their semantic
    # adapter definition used no frequency gate, which is exactly an all-one
    # multiplier; install that explicit identity value for backward-compatible
    # evaluation without relaxing strict loading for any other parameter.
    gate_key = "label_wise_attention.semantic_query_alpha_gate"
    if gate_key in model.state_dict() and gate_key not in state:
        if bool(
            OmegaConf.select(
                config,
                "model.configs.semantic_query_frequency_gate",
                default=False,
            )
        ):
            raise RuntimeError(
                "frequency-gated checkpoint is missing semantic_query_alpha_gate"
            )
        state[gate_key] = torch.ones_like(model.state_dict()[gate_key])
    model.load_state_dict(state)
    decision_boundary = float(checkpoint["db"])
    del checkpoint, state
    model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, decision_boundary


def batched(values: list[int], batch_size: int) -> Iterable[list[int]]:
    for start in range(0, len(values), batch_size):
        yield values[start : start + batch_size]


def tokenize_documents(
    texts: list[str],
    tokenizer,
    max_length: int,
    add_chunk_special_tokens: bool,
    chunk_size: int,
    tokenizer_batch_size: int,
) -> list[list[int]]:
    input_ids: list[list[int]] = []
    for start in range(0, len(texts), tokenizer_batch_size):
        batch_texts = texts[start : start + tokenizer_batch_size]
        if add_chunk_special_tokens:
            encoded = tokenize_texts_with_chunk_special_tokens(
                texts=batch_texts,
                text_tokenizer=tokenizer,
                max_length=max_length,
                chunk_size=chunk_size,
            )
        else:
            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=max_length,
                add_special_tokens=True,
            )
        input_ids.extend(encoded["input_ids"])
    return input_ids


def command_predict(args: argparse.Namespace) -> None:
    checkpoint_dir = Path(args.checkpoint_dir)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.load(checkpoint_dir / "config.yaml")
    severity = (
        DEFAULT_SEVERITIES[args.condition]
        if args.severity is None
        else float(args.severity)
    )
    if not 0 <= severity <= 1:
        raise ValueError("severity must be in [0, 1]")

    split_path = Path(args.data_dir) / f"{args.split}.parquet"
    frame = pl.read_parquet(split_path).select(["_id", "note_id", "text"])
    if args.limit is not None:
        frame = frame.head(args.limit)
    ids = frame["_id"].to_numpy()
    # Store identifiers as plain Unicode so the NPZ remains readable without pickle.
    note_ids = np.asarray(frame["note_id"].to_list(), dtype=np.str_)
    raw_texts = frame["text"].to_list()
    texts = [
        perturb_text(text, args.condition, severity, args.seed, record_id)
        for text, record_id in zip(raw_texts, note_ids)
    ]
    changed = sum(before != after for before, after in zip(raw_texts, texts))
    character_ratios = [
        len(after) / max(len(before), 1) for before, after in zip(raw_texts, texts)
    ]
    print(
        json.dumps(
            {
                "condition": args.condition,
                "severity": severity,
                "documents": len(ids),
                "changed_documents": changed,
                "mean_length_ratio": float(np.mean(character_ratios)),
            }
        ),
        flush=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.configs.model_path, use_fast=True
    )
    input_ids = tokenize_documents(
        texts=texts,
        tokenizer=tokenizer,
        max_length=int(config.data.max_length),
        add_chunk_special_tokens=bool(
            OmegaConf.select(config, "data.add_chunk_special_tokens", default=False)
        ),
        chunk_size=int(config.model.configs.chunk_size),
        tokenizer_batch_size=args.tokenizer_batch_size,
    )
    del texts, raw_texts
    order = sorted(range(len(input_ids)), key=lambda index: len(input_ids[index]))
    device = torch.device(args.device)
    model, decision_boundary = load_model_on_device(
        checkpoint_dir, config, tokenizer.pad_token_id, device
    )
    target_tokenizer = TargetTokenizer()
    target_tokenizer.load(checkpoint_dir / "target_tokenizer.json")
    probabilities = np.empty((len(ids), len(target_tokenizer)), dtype=np.float32)

    use_amp = device.type == "cuda"
    completed = 0
    with torch.inference_mode():
        for indices in batched(order, args.batch_size):
            sequences = [torch.tensor(input_ids[index], dtype=torch.long) for index in indices]
            padded = torch.nn.utils.rnn.pad_sequence(
                sequences, batch_first=True, padding_value=tokenizer.pad_token_id
            ).to(device)
            attention_mask = padded.ne(tokenizer.pad_token_id).long()
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                logits = model(padded, attention_mask)
            batch_probabilities = torch.sigmoid(logits).float().cpu().numpy()
            probabilities[np.asarray(indices)] = batch_probabilities
            completed += len(indices)
            if completed % 100 == 0 or completed == len(ids):
                print(f"predicted {completed}/{len(ids)}", flush=True)
            del padded, attention_mask, logits, batch_probabilities

    metadata = {
        "condition": args.condition,
        "severity": severity,
        "seed": args.seed,
        "split": args.split,
        "checkpoint_dir": str(checkpoint_dir),
        "decision_boundary": decision_boundary,
        "codes": target_tokenizer.id2target,
        "documents": len(ids),
        "changed_documents": changed,
        "mean_length_ratio": float(np.mean(character_ratios)),
        "rng_key": "note_id",
    }
    np.savez(output, ids=ids, note_ids=note_ids, probabilities=probabilities)
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"saved {output}", flush=True)


def parse_prediction(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("prediction must be NAME=PATH")
    name, path = value.split("=", 1)
    return name, Path(path)


def build_targets(
    target_lists: Iterable[Iterable[str]], code_to_index: dict[str, int]
) -> np.ndarray:
    targets = np.zeros((len(target_lists), len(code_to_index)), dtype=np.bool_)
    for row, codes in enumerate(target_lists):
        for code in codes:
            index = code_to_index.get(str(code))
            if index is not None:
                targets[row, index] = True
    return targets


def count_training_codes(train_path: Path) -> Counter[str]:
    counter: Counter[str] = Counter()
    frame = pl.read_parquet(train_path).select(
        ["diagnosis_codes", "procedure_codes"]
    )
    for diagnosis, procedure in frame.iter_rows():
        counter.update(set((diagnosis or []) + (procedure or [])))
    return counter


def micro_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    tp = int(np.count_nonzero(targets & predictions))
    fp = int(np.count_nonzero(~targets & predictions))
    fn = int(np.count_nonzero(targets & ~predictions))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan")
    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}


def bootstrap_disappearance_difference(
    tail_clean_tp: np.ndarray,
    tail_disappeared: np.ndarray,
    head_clean_tp: np.ndarray,
    head_disappeared: np.ndarray,
    samples: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(tail_clean_tp)
    differences = np.empty(samples, dtype=np.float64)
    for start in range(0, samples, 100):
        width = min(100, samples - start)
        indices = rng.integers(0, n, size=(width, n))
        tail_denominator = tail_clean_tp[indices].sum(axis=1)
        head_denominator = head_clean_tp[indices].sum(axis=1)
        tail_rate = tail_disappeared[indices].sum(axis=1) / np.maximum(
            tail_denominator, 1
        )
        head_rate = head_disappeared[indices].sum(axis=1) / np.maximum(
            head_denominator, 1
        )
        differences[start : start + width] = tail_rate - head_rate
    low, high = np.quantile(differences, [0.025, 0.975])
    one_sided_p = (np.count_nonzero(differences <= 0) + 1) / (samples + 1)
    return float(low), float(high), float(one_sided_p)


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm family-wise error correction with monotonic adjusted p-values."""
    values = np.asarray(p_values, dtype=np.float64)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running_max = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running_max = max(running_max, (total - rank) * values[index])
        adjusted[index] = min(running_max, 1.0)
    return adjusted.tolist()


def load_noisy_predictions(
    path: Path,
    clean_ids: np.ndarray,
    clean_note_ids: np.ndarray,
    expected_codes: list[str],
) -> tuple[np.ndarray, dict]:
    archive = np.load(path)
    noisy_ids = archive["ids"]
    noisy_note_ids = load_prediction_note_ids(path, archive)
    probabilities = archive["probabilities"]
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if metadata["codes"] != expected_codes:
        raise ValueError(f"label order differs for {path}")
    if not np.array_equal(noisy_ids, clean_ids) or not np.array_equal(
        noisy_note_ids, clean_note_ids
    ):
        raise ValueError(f"row order, admission IDs, or note IDs differ for {path}")
    return probabilities, metadata


def load_prediction_note_ids(path: Path, archive: np.lib.npyio.NpzFile) -> np.ndarray:
    if "note_ids" not in archive.files:
        raise ValueError(f"formal analysis requires note_ids in {path}")
    try:
        return archive["note_ids"]
    except ValueError as error:
        if "Object arrays cannot be loaded" not in str(error):
            raise
        with np.load(path, allow_pickle=True) as legacy_archive:
            return legacy_archive["note_ids"].astype(np.str_)


def command_analyze(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    clean_path = Path(args.clean_predictions)
    clean_archive = np.load(clean_path)
    clean_ids = clean_archive["ids"]
    clean_note_ids = load_prediction_note_ids(clean_path, clean_archive)
    clean_probabilities = clean_archive["probabilities"]
    clean_metadata = json.loads(
        clean_path.with_suffix(".json").read_text(encoding="utf-8")
    )
    codes = clean_metadata["codes"]
    test_frame = pl.read_parquet(args.test_data).select(
        ["_id", "note_id", "diagnosis_codes", "procedure_codes"]
    )
    test_ids = test_frame["_id"].to_numpy()
    test_note_ids = np.asarray(test_frame["note_id"].to_list(), dtype=np.str_)
    if not np.array_equal(test_ids, clean_ids) or not np.array_equal(
        test_note_ids, clean_note_ids
    ):
        raise ValueError(
            "clean predictions do not follow test parquet admission/note row order"
        )
    target_lists = [
        list(diagnosis or []) + list(procedure or [])
        for diagnosis, procedure in test_frame.select(
            ["diagnosis_codes", "procedure_codes"]
        ).iter_rows()
    ]
    targets = build_targets(
        target_lists, {code: idx for idx, code in enumerate(codes)}
    )
    frequencies = count_training_codes(Path(args.train_data))
    frequency_array = np.asarray([frequencies.get(code, 0) for code in codes])
    groups = {
        "tail": np.flatnonzero((frequency_array >= 1) & (frequency_array <= args.tail_max)),
        "torso": np.flatnonzero((frequency_array > args.tail_max) & (frequency_array <= args.head_min)),
        "head": np.flatnonzero(frequency_array > args.head_min),
    }

    checkpoint = torch.load(
        Path(args.checkpoint_dir) / "best_model.pt",
        map_location="cpu",
        weights_only=False,
    )
    decision_boundary = float(checkpoint["db"])
    del checkpoint
    clean_predictions = clean_probabilities > decision_boundary
    prediction_specs = [parse_prediction(value) for value in args.prediction]
    rows: list[dict] = []
    retention_rows: list[dict] = []
    hypothesis_rows: list[dict] = []
    retention_groups = {
        "rare": np.flatnonzero(frequency_array <= args.tail_max),
        "head": groups["head"],
    }

    for condition_index, (condition, path) in enumerate(prediction_specs):
        noisy_probabilities, metadata = load_noisy_predictions(
            path, clean_ids, clean_note_ids, codes
        )
        noisy_predictions = noisy_probabilities > decision_boundary
        for group, indices in retention_groups.items():
            group_clean = clean_predictions[:, indices]
            eligible = group_clean.sum(axis=1) > 0
            group_clean = group_clean[eligible]
            group_noisy = noisy_predictions[np.ix_(eligible, indices)]
            retention = (group_clean & group_noisy).sum(axis=1) / group_clean.sum(axis=1)
            retention_rows.append({
                "condition": condition,
                "severity": metadata["severity"],
                "group": group,
                "eligible_documents": int(eligible.sum()),
                "label_retention_rate": float(retention.mean()) if retention.size else None,
            })
        document_counts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for group, indices in groups.items():
            group_targets = targets[:, indices]
            group_clean = clean_predictions[:, indices]
            group_noisy = noisy_predictions[:, indices]
            clean_metrics = micro_metrics(group_targets, group_clean)
            noisy_metrics = micro_metrics(group_targets, group_noisy)
            clean_tp_mask = group_targets & group_clean
            disappeared_mask = clean_tp_mask & ~group_noisy
            clean_tp = int(clean_tp_mask.sum())
            disappeared = int(disappeared_mask.sum())
            disappearance_rate = disappeared / clean_tp if clean_tp else float("nan")
            probability_drop = (
                float(
                    (clean_probabilities[:, indices] - noisy_probabilities[:, indices])[
                        clean_tp_mask
                    ].mean()
                )
                if clean_tp
                else float("nan")
            )
            rows.append(
                {
                    "condition": condition,
                    "severity": metadata["severity"],
                    "group": group,
                    "code_count": len(indices),
                    "test_positive_instances": int(group_targets.sum()),
                    "clean_tp": clean_tp,
                    "disappeared_clean_tp": disappeared,
                    "disappearance_rate": disappearance_rate,
                    "mean_probability_drop_on_clean_tp": probability_drop,
                    "clean_precision": clean_metrics["precision"],
                    "noisy_precision": noisy_metrics["precision"],
                    "precision_drop": clean_metrics["precision"] - noisy_metrics["precision"],
                    "clean_recall": clean_metrics["recall"],
                    "noisy_recall": noisy_metrics["recall"],
                    "recall_drop": clean_metrics["recall"] - noisy_metrics["recall"],
                    "relative_recall_drop": (
                        (clean_metrics["recall"] - noisy_metrics["recall"])
                        / clean_metrics["recall"]
                        if clean_metrics["recall"]
                        else float("nan")
                    ),
                    "clean_f1": clean_metrics["f1"],
                    "noisy_f1": noisy_metrics["f1"],
                    "f1_drop": clean_metrics["f1"] - noisy_metrics["f1"],
                    "relative_f1_drop": (
                        (clean_metrics["f1"] - noisy_metrics["f1"])
                        / clean_metrics["f1"]
                        if clean_metrics["f1"]
                        else float("nan")
                    ),
                }
            )
            document_counts[group] = (
                clean_tp_mask.sum(axis=1).astype(np.int32),
                disappeared_mask.sum(axis=1).astype(np.int32),
            )

        tail_rate = rows[-3]["disappearance_rate"]
        head_rate = rows[-1]["disappearance_rate"]
        low, high, p_value = bootstrap_disappearance_difference(
            document_counts["tail"][0],
            document_counts["tail"][1],
            document_counts["head"][0],
            document_counts["head"][1],
            samples=args.bootstrap_samples,
            seed=args.bootstrap_seed + condition_index,
        )
        hypothesis_rows.append(
            {
                "condition": condition,
                "severity": metadata["severity"],
                "tail_disappearance_rate": tail_rate,
                "head_disappearance_rate": head_rate,
                "tail_minus_head": tail_rate - head_rate,
                "ci95_low": low,
                "ci95_high": high,
                "one_sided_bootstrap_p": p_value,
                "supports_tail_more_sensitive": bool(low > 0),
            }
        )
        del noisy_probabilities, noisy_predictions

    adjusted_p_values = holm_adjust(
        [row["one_sided_bootstrap_p"] for row in hypothesis_rows]
    )
    for row, adjusted_p in zip(hypothesis_rows, adjusted_p_values):
        row["holm_adjusted_p"] = adjusted_p
        row["supports_after_holm"] = bool(
            row["tail_minus_head"] > 0 and adjusted_p < 0.05
        )

    summary = pd.DataFrame(rows)
    hypothesis = pd.DataFrame(hypothesis_rows)
    summary.to_csv(output_dir / "group_metrics.csv", index=False)
    pd.DataFrame(retention_rows).to_csv(output_dir / "label_retention.csv", index=False)
    hypothesis.to_csv(output_dir / "hypothesis_tests.csv", index=False)
    result = {
        "decision_boundary": decision_boundary,
        "frequency_groups": {
            name: {
                "code_count": int(len(indices)),
                "definition": (
                    f"1 <= train_frequency <= {args.tail_max}"
                    if name == "tail"
                    else (
                        f"{args.tail_max} < train_frequency <= {args.head_min}"
                        if name == "torso"
                        else f"train_frequency > {args.head_min}"
                    )
                ),
            }
            for name, indices in groups.items()
        },
        "hypothesis_tests": hypothesis_rows,
        "direction_consistent": bool((hypothesis["tail_minus_head"] > 0).all()),
        "significant_conditions_after_holm": int(
            hypothesis["supports_after_holm"].sum()
        ),
        "all_conditions_support_after_holm": bool(
            hypothesis["supports_after_holm"].all()
        ),
    }
    (output_dir / "results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )

    lines = [
        "# PLM-ICD frequency robustness",
        "",
        f"Decision boundary: `{decision_boundary:.6f}`. Primary endpoint: the fraction of clean true positives that disappear after perturbation.",
        "",
        "| Perturbation | Tail disappear | Head disappear | Tail - Head (95% bootstrap CI) | Holm p | Significant |",
        "|---|---:|---:|---:|---:|:---:|",
    ]
    for row in hypothesis_rows:
        lines.append(
            f"| {row['condition']} | {row['tail_disappearance_rate']:.1%} | "
            f"{row['head_disappearance_rate']:.1%} | {row['tail_minus_head']:.1%} "
            f"({row['ci95_low']:.2%}, {row['ci95_high']:.2%}) | "
            f"{row['holm_adjusted_p']:.4f} | "
            f"{'yes' if row['supports_after_holm'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "The confidence intervals are individual document-bootstrap intervals. Significance uses one-sided bootstrap p-values with Holm correction across the four perturbations.",
            "",
            "## Conventional performance metrics",
            "",
            "Relative changes below are normalized by each group's clean score. Negative values mean the metric improved after perturbation.",
            "",
            "| Perturbation | Group | Relative Recall drop | Relative F1 drop |",
            "|---|---|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            f"| {row['condition']} | {row['group']} | "
            f"{row['relative_recall_drop']:.1%} | {row['relative_f1_drop']:.1%} |"
        )
    tail_clean_tp = next(row["clean_tp"] for row in rows if row["group"] == "tail")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "The primary endpoint supports greater Tail prediction instability, but it does not imply that every conventional aggregate metric drops more for Tail. Recall and F1 also include newly gained true positives and false-positive changes.",
            "",
            f"Only {tail_clean_tp} clean Tail true-positive instances are available, so the Tail estimate is aggregate and comparatively uncertain. This is a one-checkpoint, one-seed, one-severity experiment. Sentence deletion is information-removing rather than label-preserving and should be interpreted separately from spelling, formatting, and reordering.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(hypothesis.to_string(index=False), flush=True)
    print(f"saved analysis to {output_dir}", flush=True)


def command_preview(args: argparse.Namespace) -> None:
    frame = pl.read_parquet(args.data).select(["_id", "note_id", "text"]).head(args.count)
    for record_id, note_id, text in frame.iter_rows():
        print(f"\n===== id={record_id} note_id={note_id} =====")
        print(text[: args.characters])
        for condition in CONDITIONS[1:]:
            severity = DEFAULT_SEVERITIES[condition]
            perturbed = perturb_text(text, condition, severity, args.seed, note_id)
            print(f"\n--- {condition} severity={severity} ---")
            print(perturbed[: args.characters])


def _section_names(text: str) -> list[str]:
    return [section.name for section in parse_sections(text)]


def _literal_section_headers(text: str) -> list[str]:
    headers = []
    for section in parse_sections(text):
        if section.name == "UNKNOWN" or section.start == 0:
            continue
        header_start = text.rfind("\n", 0, section.start - 1) + 1
        headers.append(text[header_start : section.start])
    return headers


def _headers_preserved(original: str, perturbed: str) -> bool:
    cursor = 0
    for header in _literal_section_headers(original):
        position = perturbed.find(header, cursor)
        if position < 0:
            return False
        cursor = position + len(header)
    return True


def _is_subsequence(candidate: str, source: str) -> bool:
    source_iterator = iter(source)
    return all(character in source_iterator for character in candidate)


def _protected_section_bodies(text: str) -> list[tuple[str, str]]:
    return [
        (section.name, text[section.start : section.end])
        for section in parse_sections(text)
        if section.name in PROTECTED_SECTION_NAMES
    ]


def _section_membership_preserved(original: str, perturbed: str) -> bool:
    return all(
        sorted(original[section.start : section.end])
        == sorted(perturbed[section.start : section.end])
        for section in parse_sections(original)
    )


def _unknown_section_bodies(text: str) -> list[str]:
    return [
        text[section.start : section.end]
        for section in parse_sections(text)
        if section.name == "UNKNOWN"
    ]


def _unknown_sections_preserved(original: str, perturbed: str) -> bool:
    return all(
        original[section.start : section.end]
        == perturbed[section.start : section.end]
        for section in parse_sections(original)
        if section.name == "UNKNOWN"
    )


def _complete_sentence_units_preserved(original: str, perturbed: str) -> bool:
    original_units: Counter[str] = Counter()
    for section in parse_sections(original):
        if section.name == "UNKNOWN":
            continue
        body = original[section.start : section.end]
        for group in _complete_sentence_groups(body):
            original_units.update(body[start:end] for start, end in group)
    return all(perturbed.count(unit) >= count for unit, count in original_units.items())


def _audit_checks(condition: str, original: str, perturbed: str) -> dict[str, bool]:
    checks = {"changed": original != perturbed}
    if condition == "spelling":
        checks["layout_preserved"] = original.count("\n") == perturbed.count("\n")
    elif condition == "format":
        checks["non_whitespace_preserved"] = (
            re.sub(r"\s+", "", original) == re.sub(r"\s+", "", perturbed)
        )
    elif condition == "section_reorder":
        checks["headers_preserved"] = _headers_preserved(original, perturbed)
        checks["length_preserved"] = len(original) == len(perturbed)
        checks["character_multiset_preserved"] = sorted(original) == sorted(perturbed)
        checks["section_membership_preserved"] = _section_membership_preserved(
            original, perturbed
        )
        checks["unknown_sections_preserved"] = _unknown_sections_preserved(
            original, perturbed
        )
        checks["complete_sentence_units_preserved"] = (
            _complete_sentence_units_preserved(original, perturbed)
        )
    elif condition == "protected_delete":
        checks["headers_preserved"] = _headers_preserved(original, perturbed)
        checks["only_deletes"] = len(perturbed) < len(original)
        checks["perturbed_is_subsequence"] = _is_subsequence(perturbed, original)
        checks["protected_sections_preserved"] = (
            _protected_section_bodies(original) == _protected_section_bodies(perturbed)
        )
    return checks


def command_audit(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pl.read_parquet(args.data).select(
        ["_id", "note_id", "text", "diagnosis_codes", "procedure_codes"]
    )
    conditions = list(FORMAL_SEVERITY_LEVELS)
    rng = random.Random(args.seed)
    candidate_indices = list(range(frame.height))
    rng.shuffle(candidate_indices)
    used: set[int] = set()
    records: list[dict] = []
    full_records: list[dict] = []
    for audit_index in range(args.count):
        condition = conditions[audit_index % len(conditions)]
        levels = FORMAL_SEVERITY_LEVELS[condition]
        severity = levels[(audit_index // len(conditions)) % len(levels)]
        selected_index = None
        original = perturbed = ""
        row = None
        for candidate_index in candidate_indices:
            if candidate_index in used:
                continue
            candidate = frame.row(candidate_index, named=True)
            candidate_original = candidate["text"]
            candidate_perturbed = perturb_text(
                candidate_original,
                condition,
                severity,
                args.seed,
                candidate["note_id"],
            )
            if candidate_original != candidate_perturbed:
                selected_index = candidate_index
                row = candidate
                original = candidate_original
                perturbed = candidate_perturbed
                break
        if selected_index is None or row is None:
            raise RuntimeError(f"could not find changed audit sample for {condition}")
        used.add(selected_index)
        checks = _audit_checks(condition, original, perturbed)
        automatic_pass = all(checks.values())
        labels = list(row["diagnosis_codes"] or []) + list(row["procedure_codes"] or [])
        summary = {
            "audit_index": audit_index,
            "row_index": selected_index,
            "_id": row["_id"],
            "note_id": row["note_id"],
            "condition": condition,
            "severity": severity,
            "labels": "|".join(labels),
            "original_characters": len(original),
            "perturbed_characters": len(perturbed),
            "length_ratio": len(perturbed) / max(len(original), 1),
            "sequence_similarity": difflib.SequenceMatcher(
                None, original, perturbed
            ).ratio(),
            "automatic_pass": automatic_pass,
            "automatic_checks": json.dumps(checks, sort_keys=True),
            "human_review_status": "pending",
            "original_excerpt": original[: args.excerpt_characters].replace("\r", ""),
            "perturbed_excerpt": perturbed[: args.excerpt_characters].replace("\r", ""),
        }
        records.append(summary)
        full_records.append(
            {
                **{key: value for key, value in summary.items() if not key.endswith("excerpt")},
                "original_text": original,
                "perturbed_text": perturbed,
            }
        )
    csv_path = output_dir / "perturbation_audit.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    jsonl_path = output_dir / "perturbation_audit_full.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as output_file:
        for record in full_records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "source": str(args.data),
        "seed": args.seed,
        "samples": len(records),
        "automatic_passes": sum(record["automatic_pass"] for record in records),
        "human_review_status": "pending",
        "severity_levels": FORMAL_SEVERITY_LEVELS,
        "condition_counts": dict(Counter(record["condition"] for record in records)),
        "note": "Automated invariants passed; clinical/human sign-off is intentionally not fabricated.",
    }
    (output_dir / "perturbation_audit_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2), flush=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict", help="run one perturbation condition")
    predict.add_argument("--checkpoint-dir", required=True)
    predict.add_argument(
        "--data-dir", default="data/processed/mimiciii_full"
    )
    predict.add_argument("--split", choices=("val", "test"), default="test")
    predict.add_argument("--condition", choices=CONDITIONS, required=True)
    predict.add_argument("--severity", type=float)
    predict.add_argument("--seed", type=int, default=1337)
    predict.add_argument("--device", default="cuda:0")
    predict.add_argument("--batch-size", type=int, default=2)
    predict.add_argument("--tokenizer-batch-size", type=int, default=64)
    predict.add_argument("--limit", type=int)
    predict.add_argument("--output", required=True)
    predict.set_defaults(function=command_predict)

    analyze = subparsers.add_parser("analyze", help="compare predictions by frequency")
    analyze.add_argument("--checkpoint-dir", required=True)
    analyze.add_argument(
        "--clean-predictions",
        required=True,
        help="clean-condition npz generated by the predict command",
    )
    analyze.add_argument(
        "--test-data",
        default="data/processed/mimiciii_full/test.parquet",
    )
    analyze.add_argument(
        "--train-data",
        default="data/processed/mimiciii_full/train.parquet",
    )
    analyze.add_argument(
        "--prediction",
        action="append",
        required=True,
        help="condition and npz path as NAME=PATH; repeat for each condition",
    )
    analyze.add_argument("--tail-max", type=int, default=10)
    analyze.add_argument("--head-min", type=int, default=100)
    analyze.add_argument("--bootstrap-samples", type=int, default=2000)
    analyze.add_argument("--bootstrap-seed", type=int, default=20260731)
    analyze.add_argument("--output-dir", required=True)
    analyze.set_defaults(function=command_analyze)

    preview = subparsers.add_parser("preview", help="visually inspect perturbations")
    preview.add_argument(
        "--data", default="data/processed/mimiciii_full/test.parquet"
    )
    preview.add_argument("--count", type=int, default=1)
    preview.add_argument("--characters", type=int, default=1200)
    preview.add_argument("--seed", type=int, default=1337)
    preview.set_defaults(function=command_preview)

    audit = subparsers.add_parser(
        "audit", help="create a stratified perturbation audit packet"
    )
    audit.add_argument(
        "--data", default="data/processed/mimiciii_full/val.parquet"
    )
    audit.add_argument("--output-dir", required=True)
    audit.add_argument("--count", type=int, default=100)
    audit.add_argument("--excerpt-characters", type=int, default=800)
    audit.add_argument("--seed", type=int, default=20260801)
    audit.set_defaults(function=command_audit)
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if hasattr(args, "batch_size") and args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    args.function(args)


if __name__ == "__main__":
    main()

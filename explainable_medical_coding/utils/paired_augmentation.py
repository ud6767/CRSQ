"""Deterministic label-preserving training perturbations for paired views."""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass

from explainable_medical_coding.evidence.section_parser import parse_sections


WORD_RE = re.compile(r"\b[A-Za-z]{4,}\b")
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


@dataclass(frozen=True)
class AugmentationChoice:
    condition: str
    severity: float


DEFAULT_TRAINING_CHOICES = (
    AugmentationChoice("spelling", 0.01),
    AugmentationChoice("spelling", 0.03),
    AugmentationChoice("format", 0.05),
    AugmentationChoice("format", 0.10),
    AugmentationChoice("section_reorder", 1 / 3),
    AugmentationChoice("section_reorder", 2 / 3),
)


def stable_rng(seed: int, namespace: str, record_key: object) -> random.Random:
    payload = f"{seed}|{namespace}|{record_key}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return random.Random(int.from_bytes(digest, "little"))


def misspell_word(word: str, rng: random.Random) -> str:
    chars = list(word)
    operation = rng.choice(("swap", "delete", "duplicate", "substitute"))
    position = rng.randrange(1, len(chars) - 1)
    if operation == "swap":
        chars[position], chars[position + 1] = chars[position + 1], chars[position]
    elif operation == "delete":
        del chars[position]
    elif operation == "duplicate":
        chars.insert(position, chars[position])
    else:
        replacement = rng.choice("abcdefghijklmnopqrstuvwxyz")
        chars[position] = replacement.upper() if chars[position].isupper() else replacement
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


def _is_complete_sentence_boundary(
    text: str, start: int, end: int
) -> bool:
    punctuation_start = end - 1
    while punctuation_start >= start and text[punctuation_start] in "\"')].!?":
        punctuation_start -= 1
    token_match = re.search(
        r"([A-Za-z]+)\W*$", text[start : punctuation_start + 1]
    )
    token = token_match.group(1).lower() if token_match else ""
    if token in NON_TERMINAL_ABBREVIATIONS or (
        len(token) == 1 and token.isalpha()
    ):
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
    return (
        text[cursor].isupper()
        or text[cursor].isdigit()
        or text[cursor] in "[#"
    )


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


def _replace_spans(
    text: str, replacements: list[tuple[int, int, str]]
) -> str:
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
            (start, end, value)
            for (start, end), value in zip(spans, shuffled)
        )
    return _replace_spans(body, replacements) if replacements else body


def perturb_section_reorder(
    text: str, severity: float, rng: random.Random
) -> str:
    """Reorder complete sentences inside eligible non-UNKNOWN sections."""
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


def augment_training_text(
    text: str,
    record_key: object,
    seed: int,
    choices: tuple[AugmentationChoice, ...] = DEFAULT_TRAINING_CHOICES,
) -> dict[str, str | float]:
    """Choose and apply exactly one reproducible training perturbation."""
    if not choices:
        raise ValueError("at least one augmentation choice is required")
    choice_rng = stable_rng(seed, "paired_training_choice", record_key)
    choice = choices[choice_rng.randrange(len(choices))]
    perturb_rng = stable_rng(
        seed,
        f"paired_training_{choice.condition}_{choice.severity:g}",
        record_key,
    )
    output_condition = choice.condition
    output_severity = choice.severity
    if choice.condition == "spelling":
        augmented_text = perturb_spelling(text, choice.severity, perturb_rng)
    elif choice.condition == "format":
        augmented_text = perturb_format(text, choice.severity, perturb_rng)
    elif choice.condition == "section_reorder":
        augmented_text = perturb_section_reorder(
            text, choice.severity, perturb_rng
        )
        if augmented_text == text:
            output_condition = "spelling"
            output_severity = 0.01
            fallback_rng = stable_rng(
                seed, "paired_training_reorder_fallback_spelling_0.01", record_key
            )
            augmented_text = perturb_spelling(text, output_severity, fallback_rng)
    else:
        raise ValueError(f"unsupported training perturbation: {choice.condition}")
    if augmented_text == text:
        output_condition = "format"
        output_severity = 0.05
        final_fallback_rng = stable_rng(
            seed, "paired_training_final_fallback_format_0.05", record_key
        )
        augmented_text = perturb_format(
            text, output_severity, final_fallback_rng
        )
    return {
        "augmented_text": augmented_text,
        "augmentation_condition": output_condition,
        "augmentation_severity": output_severity,
    }

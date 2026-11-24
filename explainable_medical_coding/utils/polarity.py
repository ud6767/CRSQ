import csv
import gzip
import re
from collections import defaultdict
from pathlib import Path
from typing import Sequence

from explainable_medical_coding.utils.data_helper_functions import (
    reformat_icd9cm_code,
)
from explainable_medical_coding.utils.tokenizer import TargetTokenizer


CUE_PATTERN = re.compile(
    r"\b(?:no evidence of|negative for|denies|denied|ruled out|rule out|without)"
    r"\b.{2,120}?(?=\b(?:positive for|but|however|reports?|states?|endorses?)\b|"
    r"[\n.;:!?]|$)",
    re.IGNORECASE,
)
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "classified",
    "elsewhere",
    "family",
    "history",
    "in",
    "of",
    "other",
    "or",
    "specified",
    "the",
    "to",
    "unspecified",
    "with",
    "without",
}


def normalize_token(token: str) -> str:
    if len(token) > 5 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def normalize_words(text: str) -> list[str]:
    return [
        normalized
        for token in TOKEN_PATTERN.findall(text.lower())
        if (normalized := normalize_token(token)) not in STOPWORDS
    ]


def has_ordered_bigram(
    description_words: tuple[str, ...], scope_words: list[str]
) -> bool:
    description_bigrams = set(zip(description_words, description_words[1:]))
    return any(
        pair in description_bigrams for pair in zip(scope_words, scope_words[1:])
    )


def load_mimiciii_diagnosis_descriptions(path: str | Path) -> dict[str, str]:
    """Load diagnosis descriptions without admitting procedure labels."""
    path = Path(path)
    open_file = gzip.open if path.suffix == ".gz" else open
    with open_file(path, mode="rt", encoding="utf-8", newline="") as file:
        rows = csv.DictReader(file)
        return {
            reformat_icd9cm_code(row["ICD9_CODE"]): row["LONG_TITLE"]
            for row in rows
            if row.get("ICD9_CODE") and row.get("LONG_TITLE")
        }


class PolarityGuidanceGenerator:
    """Create conservative label/span pairs from explicit negation scopes."""

    def __init__(
        self,
        target_tokenizer: TargetTokenizer,
        code_to_description: dict[str, str],
        max_labels_per_scope: int = 4,
        max_pairs_per_document: int = 32,
    ) -> None:
        self.target_tokenizer = target_tokenizer
        self.max_labels_per_scope = max_labels_per_scope
        self.max_pairs_per_document = max_pairs_per_document
        self.signatures: dict[int, tuple[str, ...]] = {}
        self.inverted_index: dict[str, set[int]] = defaultdict(set)

        for code, label_id in target_tokenizer.target2id.items():
            if code.startswith(("V", "E")):
                continue
            description = code_to_description.get(code)
            if not description:
                continue
            signature = tuple(dict.fromkeys(normalize_words(description)))
            if not signature:
                continue
            self.signatures[label_id] = signature
            for word in signature:
                self.inverted_index[word].add(label_id)

    @staticmethod
    def _scope_token_ids(
        scope_start: int,
        scope_end: int,
        offset_mapping: Sequence[Sequence[int]],
    ) -> list[int]:
        return [
            token_id
            for token_id, offset in enumerate(offset_mapping)
            if int(offset[1]) > int(offset[0])
            and int(offset[0]) < scope_end
            and int(offset[1]) > scope_start
        ]

    def __call__(
        self,
        text: str,
        assigned_codes: Sequence[str],
        offset_mapping: Sequence[Sequence[int]],
    ) -> dict[str, list]:
        assigned_codes = set(assigned_codes)
        polarity_label_ids: list[int] = []
        polarity_token_ids: list[list[int]] = []

        for scope_match in CUE_PATTERN.finditer(text):
            scope_words = normalize_words(scope_match.group(0))
            candidate_ids: set[int] = set()
            for word in set(scope_words):
                candidate_ids.update(self.inverted_index.get(word, ()))

            accepted = []
            for label_id in candidate_ids:
                code = self.target_tokenizer.id2target[label_id]
                if code in assigned_codes:
                    continue
                signature = self.signatures[label_id]
                overlap = len(set(signature).intersection(scope_words))
                coverage = overlap / len(signature)
                if len(signature) == 1:
                    matched = coverage == 1.0 and len(signature[0]) >= 7
                else:
                    matched = (
                        overlap >= 2
                        and coverage >= 2 / 3
                        and has_ordered_bigram(signature, scope_words)
                    )
                if matched:
                    accepted.append((coverage, overlap, label_id))

            accepted.sort(reverse=True)
            token_ids = self._scope_token_ids(
                scope_match.start(), scope_match.end(), offset_mapping
            )
            if not token_ids:
                continue
            for _, _, label_id in accepted[: self.max_labels_per_scope]:
                polarity_label_ids.append(label_id)
                polarity_token_ids.append(token_ids)
                if len(polarity_label_ids) >= self.max_pairs_per_document:
                    return {
                        "polarity_label_ids": polarity_label_ids,
                        "polarity_token_ids": polarity_token_ids,
                    }

        return {
            "polarity_label_ids": polarity_label_ids,
            "polarity_token_ids": polarity_token_ids,
        }

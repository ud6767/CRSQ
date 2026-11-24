import re
from dataclasses import dataclass

from explainable_medical_coding.evidence.section_parser import parse_sections


SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])|\n+")


@dataclass(frozen=True)
class SentenceSpan:
    text: str
    start: int
    end: int
    section: str


def split_sentences(text: str) -> list[SentenceSpan]:
    """Split clinical text into sentences while preserving document offsets."""
    sentences: list[SentenceSpan] = []
    for section in parse_sections(text):
        section_text = text[section.start : section.end]
        cursor = 0
        for boundary in SENTENCE_BOUNDARY.finditer(section_text):
            _append_sentence(
                sentences,
                section_text[cursor : boundary.start()],
                section.start + cursor,
                section.name,
            )
            cursor = boundary.end()
        _append_sentence(
            sentences,
            section_text[cursor:],
            section.start + cursor,
            section.name,
        )
    return sentences


def _append_sentence(
    output: list[SentenceSpan],
    raw_sentence: str,
    raw_start: int,
    section: str,
) -> None:
    left_trimmed = len(raw_sentence) - len(raw_sentence.lstrip())
    sentence = raw_sentence.strip()
    if not sentence:
        return
    start = raw_start + left_trimmed
    output.append(
        SentenceSpan(
            text=sentence,
            start=start,
            end=start + len(sentence),
            section=section,
        )
    )

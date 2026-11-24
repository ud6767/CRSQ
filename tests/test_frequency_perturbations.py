import random
import re

from scripts.evaluate_frequency_robustness import (
    _complete_sentence_units_preserved,
    _protected_section_bodies,
    _unknown_section_bodies,
    complete_sentence_spans,
    perturb_format,
    perturb_protected_delete,
    perturb_section_reorder,
)


SAMPLE_NOTE = """HISTORY OF PRESENT ILLNESS:
First sentence.
Second sentence.
Third sentence.
Fourth sentence.
Fifth sentence.

DISCHARGE DIAGNOSIS:
Acute myocardial infarction.
Hypertension.

HOSPITAL COURSE:
Course sentence one.
Course sentence two.
Course sentence three.
Course sentence four.
Course sentence five.
"""


def test_section_reorder_is_deterministic_and_content_preserving() -> None:
    first = perturb_section_reorder(SAMPLE_NOTE, 1.0, random.Random(17))
    second = perturb_section_reorder(SAMPLE_NOTE, 1.0, random.Random(17))

    assert first == second
    assert first != SAMPLE_NOTE
    assert len(first) == len(SAMPLE_NOTE)
    assert sorted(first) == sorted(SAMPLE_NOTE)
    assert _protected_section_bodies(first) == _protected_section_bodies(SAMPLE_NOTE)


def test_protected_delete_keeps_diagnosis_section() -> None:
    perturbed = perturb_protected_delete(SAMPLE_NOTE, 0.5, random.Random(23))

    assert len(perturbed) < len(SAMPLE_NOTE)
    assert _protected_section_bodies(perturbed) == _protected_section_bodies(SAMPLE_NOTE)


def test_format_noise_changes_only_whitespace() -> None:
    perturbed = perturb_format(SAMPLE_NOTE, 0.2, random.Random(31))

    assert perturbed != SAMPLE_NOTE
    assert re.sub(r"\s+", "", perturbed) == re.sub(r"\s+", "", SAMPLE_NOTE)


def test_complete_sentences_keep_hard_wrapped_lines_together() -> None:
    paragraph = """The patient presented with severe chest
pain after exercise. She denied nausea or
vomiting at presentation. Testing showed an elevated
troponin level. Cardiology recommended urgent
catheterization."""

    spans = complete_sentence_spans(paragraph)
    sentences = [paragraph[start:end] for start, end in spans]

    assert len(sentences) == 4
    assert any("severe chest\npain after exercise." in sentence for sentence in sentences)
    assert "denied nausea or\nvomiting" in sentences[1]


def test_section_reorder_preserves_unknown_preamble_and_numbered_lists() -> None:
    note = """Admission Date: 2100-01-01
Service: MEDICINE

History of Present Illness:
First complete sentence has enough words.
Second complete sentence has enough words.
Third complete sentence has enough words.
Fourth complete sentence has enough words.

Major Surgical or Invasive Procedure:
1. First numbered procedure was performed.
2. Second numbered procedure was performed.
3. Third numbered procedure was performed.
4. Fourth numbered procedure was performed.
"""
    perturbed = perturb_section_reorder(note, 1.0, random.Random(41))

    assert perturbed != note
    assert _unknown_section_bodies(perturbed) == _unknown_section_bodies(note)
    assert _complete_sentence_units_preserved(note, perturbed)
    assert "1. First numbered procedure was performed.\n2. Second" in perturbed


if __name__ == "__main__":
    test_section_reorder_is_deterministic_and_content_preserving()
    test_protected_delete_keeps_diagnosis_section()
    test_format_noise_changes_only_whitespace()
    test_complete_sentences_keep_hard_wrapped_lines_together()
    test_section_reorder_preserves_unknown_preamble_and_numbered_lists()
    print("5 perturbation tests passed")

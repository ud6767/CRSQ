import re
from dataclasses import dataclass


SECTION_ALIASES = {
    "history of present illness": "HPI",
    "hpi": "HPI",
    "past medical history": "PMH",
    "pmh": "PMH",
    "family history": "FAMILY_HISTORY",
    "social history": "SOCIAL_HISTORY",
    "hospital course": "HOSPITAL_COURSE",
    "discharge diagnosis": "DISCHARGE_DIAGNOSIS",
    "discharge diagnoses": "DISCHARGE_DIAGNOSIS",
    "admission diagnosis": "ADMISSION_DIAGNOSIS",
    "assessment and plan": "ASSESSMENT_PLAN",
    "assessment/plan": "ASSESSMENT_PLAN",
    "procedures": "PROCEDURES",
    "radiology": "RADIOLOGY",
}


@dataclass(frozen=True)
class SectionSpan:
    name: str
    start: int
    end: int


def normalize_section_name(header: str) -> str:
    cleaned = re.sub(r"\s+", " ", header.strip().rstrip(":")).lower()
    if cleaned in SECTION_ALIASES:
        return SECTION_ALIASES[cleaned]
    return re.sub(r"[^A-Z0-9]+", "_", cleaned.upper()).strip("_") or "UNKNOWN"


def _is_section_header(line: str) -> bool:
    stripped = line.strip()
    if not stripped or len(stripped) > 80:
        return False
    normalized = re.sub(r"\s+", " ", stripped.rstrip(":")).lower()
    if normalized in SECTION_ALIASES:
        return True
    if stripped.endswith(":") and len(stripped.split()) <= 8:
        return True
    letters = [character for character in stripped if character.isalpha()]
    return bool(letters) and all(character.isupper() for character in letters)


def parse_sections(text: str) -> list[SectionSpan]:
    """Return section body offsets without treating the word History as history."""
    headers: list[tuple[str, int]] = []
    offset = 0
    for line in text.splitlines(keepends=True):
        if _is_section_header(line):
            headers.append((normalize_section_name(line), offset + len(line)))
        offset += len(line)

    if not headers:
        return [SectionSpan(name="UNKNOWN", start=0, end=len(text))]

    sections: list[SectionSpan] = []
    if headers[0][1] > 0:
        header_line_start = text.rfind("\n", 0, headers[0][1] - 1) + 1
        if header_line_start > 0:
            sections.append(SectionSpan("UNKNOWN", 0, header_line_start))
    for index, (name, body_start) in enumerate(headers):
        if index + 1 < len(headers):
            next_body_start = headers[index + 1][1]
            next_header_start = text.rfind("\n", 0, next_body_start - 1) + 1
            body_end = next_header_start
        else:
            body_end = len(text)
        sections.append(SectionSpan(name=name, start=body_start, end=body_end))
    return sections

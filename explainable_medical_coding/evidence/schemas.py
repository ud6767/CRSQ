from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Optional


class EvidenceType(str, Enum):
    SUPPORT = "SUPPORT"
    CONTRADICTION = "CONTRADICTION"
    HISTORY = "HISTORY"
    UNCERTAIN = "UNCERTAIN"
    NONE = "NONE"


class Assertion(str, Enum):
    AFFIRMED = "affirmed"
    NEGATED = "negated"
    POSSIBLE = "possible"
    CONDITIONAL = "conditional"
    HYPOTHETICAL = "hypothetical"
    UNKNOWN = "unknown"


class EncounterRelation(str, Enum):
    CURRENT_EPISODE = "current_episode"
    REMOTE_HISTORY = "remote_history"
    FUTURE_OR_PLANNED = "future_or_planned"
    UNKNOWN = "unknown"


class Experiencer(str, Enum):
    PATIENT = "patient"
    FAMILY = "family"
    OTHER = "other"
    UNKNOWN = "unknown"


class ProcedureStatus(str, Enum):
    PERFORMED = "performed"
    PLANNED = "planned"
    CANCELLED = "cancelled"
    ABORTED = "aborted"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class CodeType(str, Enum):
    DIAGNOSIS = "diagnosis"
    PROCEDURE = "procedure"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class EvidenceAttributes:
    relevant: bool
    assertion: Assertion = Assertion.UNKNOWN
    encounter_relation: EncounterRelation = EncounterRelation.UNKNOWN
    experiencer: Experiencer = Experiencer.UNKNOWN
    procedure_status: ProcedureStatus = ProcedureStatus.NOT_APPLICABLE

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceAttributes":
        return cls(
            relevant=bool(value["relevant"]),
            assertion=Assertion(value.get("assertion", Assertion.UNKNOWN)),
            encounter_relation=EncounterRelation(
                value.get("encounter_relation", EncounterRelation.UNKNOWN)
            ),
            experiencer=Experiencer(value.get("experiencer", Experiencer.UNKNOWN)),
            procedure_status=ProcedureStatus(
                value.get("procedure_status", ProcedureStatus.NOT_APPLICABLE)
            ),
        )


@dataclass(frozen=True)
class CodeMetadata:
    code: str
    description: str
    code_type: CodeType = CodeType.DIAGNOSIS
    synonyms: tuple[str, ...] = field(default_factory=tuple)
    is_personal_history_code: bool = False
    is_family_history_code: bool = False
    is_status_code: bool = False
    family: Optional[str] = None
    sibling_codes: tuple[str, ...] = field(default_factory=tuple)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CodeMetadata":
        description = str(value.get("description", value.get("code_description", "")))
        lowered = description.lower()
        return cls(
            code=str(value["code"]),
            description=description,
            code_type=CodeType(value.get("code_type", CodeType.DIAGNOSIS)),
            synonyms=tuple(value.get("synonyms", ())),
            is_personal_history_code=bool(
                value.get("is_personal_history_code", "personal history" in lowered)
            ),
            is_family_history_code=bool(
                value.get("is_family_history_code", "family history" in lowered)
            ),
            is_status_code=bool(
                value.get(
                    "is_status_code",
                    "status" in lowered or "presence of" in lowered,
                )
            ),
            family=value.get("family"),
            sibling_codes=tuple(value.get("sibling_codes", ())),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["code_type"] = self.code_type.value
        value["synonyms"] = list(self.synonyms)
        value["sibling_codes"] = list(self.sibling_codes)
        return value


@dataclass(frozen=True)
class EvidenceDocument:
    hadm_id: str
    note_id: str
    text: str


@dataclass(frozen=True)
class ExistingEvidenceSpan:
    code: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass(frozen=True)
class AttentionSentenceCandidate:
    code: str
    sentence_start: int
    score: float


@dataclass(frozen=True)
class EvidenceExample:
    hadm_id: str
    note_id: str
    code: str
    code_description: str
    code_type: CodeType
    section: str
    sentence: str
    sentence_start: int
    sentence_end: int
    mention_start: Optional[int]
    mention_end: Optional[int]
    relevant: bool
    assertion: Assertion
    encounter_relation: EncounterRelation
    experiencer: Experiencer
    procedure_status: ProcedureStatus
    evidence_type: EvidenceType
    annotation_source: str
    confidence: float
    has_explicit_mention: bool = False

    def __post_init__(self) -> None:
        if self.sentence_start < 0 or self.sentence_end < self.sentence_start:
            raise ValueError("invalid sentence offsets")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if (self.mention_start is None) != (self.mention_end is None):
            raise ValueError("mention_start and mention_end must both be set or null")
        if self.mention_start is not None:
            if not 0 <= self.mention_start <= self.mention_end <= len(self.sentence):
                raise ValueError("mention offsets must be relative to the sentence")

    @property
    def attributes(self) -> EvidenceAttributes:
        return EvidenceAttributes(
            relevant=self.relevant,
            assertion=self.assertion,
            encounter_relation=self.encounter_relation,
            experiencer=self.experiencer,
            procedure_status=self.procedure_status,
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in (
            "code_type",
            "assertion",
            "encounter_relation",
            "experiencer",
            "procedure_status",
            "evidence_type",
        ):
            value[key] = getattr(self, key).value
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "EvidenceExample":
        return cls(
            **{
                **value,
                "code_type": CodeType(value["code_type"]),
                "assertion": Assertion(value["assertion"]),
                "encounter_relation": EncounterRelation(value["encounter_relation"]),
                "experiencer": Experiencer(value["experiencer"]),
                "procedure_status": ProcedureStatus(value["procedure_status"]),
                "evidence_type": EvidenceType(value["evidence_type"]),
            }
        )

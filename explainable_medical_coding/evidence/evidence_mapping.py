from explainable_medical_coding.evidence.schemas import (
    Assertion,
    CodeMetadata,
    EncounterRelation,
    EvidenceAttributes,
    EvidenceType,
    Experiencer,
    ProcedureStatus,
)


def derive_evidence_type(
    attrs: EvidenceAttributes,
    code_meta: CodeMetadata,
) -> EvidenceType:
    """Derive evidence polarity while honoring history and status code semantics."""
    if not attrs.relevant:
        return EvidenceType.NONE

    if code_meta.is_personal_history_code:
        if (
            attrs.assertion == Assertion.AFFIRMED
            and attrs.encounter_relation == EncounterRelation.REMOTE_HISTORY
            and attrs.experiencer == Experiencer.PATIENT
        ):
            return EvidenceType.SUPPORT

    if code_meta.is_family_history_code:
        if (
            attrs.assertion == Assertion.AFFIRMED
            and attrs.experiencer == Experiencer.FAMILY
        ):
            return EvidenceType.SUPPORT

    if code_meta.is_status_code:
        if attrs.procedure_status == ProcedureStatus.PERFORMED:
            return EvidenceType.SUPPORT

    if attrs.assertion == Assertion.NEGATED or attrs.procedure_status in {
        ProcedureStatus.CANCELLED,
        ProcedureStatus.ABORTED,
    }:
        return EvidenceType.CONTRADICTION

    if (
        attrs.assertion
        in {Assertion.POSSIBLE, Assertion.CONDITIONAL, Assertion.HYPOTHETICAL}
        or attrs.encounter_relation == EncounterRelation.FUTURE_OR_PLANNED
        or attrs.procedure_status == ProcedureStatus.PLANNED
    ):
        return EvidenceType.UNCERTAIN

    if (
        attrs.encounter_relation == EncounterRelation.REMOTE_HISTORY
        or attrs.experiencer in {Experiencer.FAMILY, Experiencer.OTHER}
    ):
        return EvidenceType.HISTORY

    if (
        attrs.assertion == Assertion.AFFIRMED
        and attrs.encounter_relation == EncounterRelation.CURRENT_EPISODE
    ):
        return EvidenceType.SUPPORT

    return EvidenceType.NONE

from typing import Sequence

import torch

from explainable_medical_coding.evidence.schemas import AttentionSentenceCandidate
from explainable_medical_coding.evidence.sentence_splitter import SentenceSpan


def aggregate_token_attention_to_sentences(
    token_attention: torch.Tensor,
    offset_mapping: Sequence[tuple[int, int]],
    sentences: Sequence[SentenceSpan],
    codes: Sequence[str],
    top_k: int = 8,
    pooling: str = "sum",
) -> list[AttentionSentenceCandidate]:
    """Aggregate one document's [T, L] baseline attention into top sentences."""
    if token_attention.ndim != 2:
        raise ValueError("token_attention must have shape [tokens, labels]")
    if token_attention.size(0) != len(offset_mapping):
        raise ValueError("offset_mapping length must match the token dimension")
    if token_attention.size(1) != len(codes):
        raise ValueError("codes length must match the label dimension")
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if pooling not in {"sum", "max"}:
        raise ValueError("pooling must be 'sum' or 'max'")

    attention = token_attention.detach().float().cpu()
    candidates: list[AttentionSentenceCandidate] = []
    for code_index, code in enumerate(codes):
        sentence_scores: list[tuple[float, SentenceSpan]] = []
        for sentence in sentences:
            token_indices = [
                token_index
                for token_index, (token_start, token_end) in enumerate(offset_mapping)
                if token_end > token_start
                and token_start < sentence.end
                and token_end > sentence.start
            ]
            if not token_indices:
                continue
            values = attention[token_indices, code_index]
            score = values.sum() if pooling == "sum" else values.max()
            sentence_scores.append((float(score.item()), sentence))
        for score, sentence in sorted(
            sentence_scores,
            key=lambda item: item[0],
            reverse=True,
        )[:top_k]:
            candidates.append(
                AttentionSentenceCandidate(
                    code=str(code),
                    sentence_start=sentence.start,
                    score=score,
                )
            )
    return candidates

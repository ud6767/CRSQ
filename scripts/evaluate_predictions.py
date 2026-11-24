"""Compute clean test metrics from a completed run and its frozen threshold."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch


def evaluate(predictions: pd.DataFrame, threshold: float) -> dict:
    scores = predictions.drop(columns=["_id", "target"])
    probabilities = scores.to_numpy(dtype=np.float32)
    code_indices = {code: index for index, code in enumerate(scores.columns)}
    targets = np.zeros(probabilities.shape, dtype=bool)
    for row, codes in enumerate(predictions["target"]):
        for code in codes:
            targets[row, code_indices[code]] = True

    positive = probabilities > threshold
    tp = np.count_nonzero(positive & targets, axis=0)
    denominator = positive.sum(axis=0) + targets.sum(axis=0)
    per_code_f1 = np.divide(
        2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0
    )
    micro_f1 = 2 * tp.sum() / denominator.sum() if denominator.sum() else 0.0

    supports = targets.sum(axis=0)
    auc_codes = np.flatnonzero((supports > 0) & (supports < len(targets)))
    metrics = {
        "documents": len(targets),
        "output_labels": len(code_indices),
        "decision_threshold": threshold,
        "macro_f1": float(per_code_f1.mean()),
        "micro_f1": float(micro_f1),
        "macro_auc": float(np.mean([
            roc_auc_score(targets[:, j], probabilities[:, j]) for j in auc_codes
        ])) if auc_codes.size else None,
        "micro_auc": (
            float(roc_auc_score(targets.ravel(), probabilities.ravel()))
            if targets.any() and not targets.all() else None
        ),
    }
    cutoffs = [k for k in (8, 15) if len(code_indices) >= k]
    if cutoffs:
        limit = max(cutoffs)
        top_indices = np.argpartition(-probabilities, limit - 1, axis=1)[:, :limit]
        top_scores = np.take_along_axis(probabilities, top_indices, axis=1)
        top_targets = np.take_along_axis(targets, top_indices, axis=1)
        ranked_targets = np.take_along_axis(top_targets, np.argsort(-top_scores, axis=1), axis=1)
        for k in cutoffs:
            metrics[f"p_at_{k}"] = float(ranked_targets[:, :k].mean())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    checkpoint = torch.load(
        args.run_dir / "best_model.pt", map_location="cpu", weights_only=False
    )
    threshold = float(checkpoint.pop("db"))
    del checkpoint
    predictions = pd.read_feather(args.run_dir / "predictions_test.feather")
    print(json.dumps(evaluate(predictions, threshold), indent=2))


if __name__ == "__main__":
    main()

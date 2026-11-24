"""Training-time decomposition of raw and actually optimized consistency."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch


CATEGORIES = (
    "positive_zero_shot",
    "positive_tail",
    "positive_torso",
    "positive_head",
    "negative_easy",
    "negative_medium",
    "negative_hard",
)


@dataclass
class CategoryAccumulator:
    cell_count: int = 0
    loss_sum: float = 0.0
    gradient_sum: float = 0.0
    objective_loss_sum: float = 0.0
    objective_gradient_sum: float = 0.0


@dataclass
class ConsistencyDiagnosticsTracker:
    label_frequencies: torch.Tensor
    total_steps: int
    decision_boundary: float
    easy_threshold: float = 0.01
    checkpoints: tuple[float, ...] = (0.1, 0.5, 1.0)
    next_checkpoint_index: int = 0
    phase_start_step: int = 0
    accumulators: dict[str, CategoryAccumulator] = field(default_factory=dict)
    supervised_loss_sum: float = 0.0
    consistency_loss_sum: float = 0.0
    weighted_consistency_loss_sum: float = 0.0
    total_loss_sum: float = 0.0
    batch_count: int = 0

    def __post_init__(self) -> None:
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if not 0 < self.decision_boundary < 1:
            raise ValueError("decision_boundary must lie in (0, 1)")
        if tuple(sorted(self.checkpoints)) != self.checkpoints:
            raise ValueError("diagnostic checkpoints must be sorted")
        if not self.accumulators:
            self.accumulators = {
                category: CategoryAccumulator() for category in CATEGORIES
            }
        self.label_frequencies = self.label_frequencies.to(dtype=torch.long, device="cpu")

    @torch.no_grad()
    def update(self, payload: dict[str, torch.Tensor]) -> None:
        clean = payload["clean_probabilities"]
        augmented = payload["augmented_probabilities"]
        targets = payload["targets"].bool()
        frequencies = self.label_frequencies.to(clean.device)
        delta = clean - augmented
        mse = delta.square()
        gradient = (
            (2 * delta * clean * (1 - clean)).abs()
            + (2 * delta * augmented * (1 - augmented)).abs()
        )
        objective_coefficients = payload.get("objective_coefficients")
        if objective_coefficients is None:
            objective_coefficients = torch.full_like(mse, 1.0 / mse.numel())
        objective_loss = mse * objective_coefficients
        objective_gradient = gradient * objective_coefficients
        maximum_probability = torch.maximum(clean, augmented)
        negative = ~targets
        masks = {
            "positive_zero_shot": targets & (frequencies[None, :] == 0),
            "positive_tail": targets
            & (frequencies[None, :] >= 1)
            & (frequencies[None, :] <= 10),
            "positive_torso": targets
            & (frequencies[None, :] > 10)
            & (frequencies[None, :] <= 100),
            "positive_head": targets & (frequencies[None, :] > 100),
            "negative_easy": negative & (maximum_probability < self.easy_threshold),
            "negative_medium": negative
            & (maximum_probability >= self.easy_threshold)
            & (maximum_probability < self.decision_boundary),
            "negative_hard": negative
            & (maximum_probability >= self.decision_boundary),
        }
        if sum(int(mask.sum().item()) for mask in masks.values()) != targets.numel():
            raise RuntimeError("consistency diagnostic partition does not close")
        for category, mask in masks.items():
            accumulator = self.accumulators[category]
            accumulator.cell_count += int(mask.sum().item())
            accumulator.loss_sum += float(mse[mask].sum().item())
            accumulator.gradient_sum += float(gradient[mask].sum().item())
            accumulator.objective_loss_sum += float(
                objective_loss[mask].sum().item()
            )
            accumulator.objective_gradient_sum += float(
                objective_gradient[mask].sum().item()
            )
        self.supervised_loss_sum += float(payload["supervised_loss"].item())
        self.consistency_loss_sum += float(payload["consistency_loss"].item())
        self.weighted_consistency_loss_sum += float(
            payload.get("weighted_consistency_loss", payload["consistency_loss"] * 0).item()
        )
        self.total_loss_sum += float(
            payload.get("total_loss", payload["supervised_loss"]).item()
        )
        self.batch_count += 1

    def pop_due_rows(self, global_step: int) -> list[dict[str, float | int | str]]:
        if self.next_checkpoint_index >= len(self.checkpoints):
            return []
        target_fraction = self.checkpoints[self.next_checkpoint_index]
        target_step = max(1, round(target_fraction * self.total_steps))
        if global_step < target_step:
            return []
        total_cells = sum(item.cell_count for item in self.accumulators.values())
        total_loss = sum(item.loss_sum for item in self.accumulators.values())
        total_gradient = sum(item.gradient_sum for item in self.accumulators.values())
        total_objective_loss = sum(
            item.objective_loss_sum for item in self.accumulators.values()
        )
        total_objective_gradient = sum(
            item.objective_gradient_sum for item in self.accumulators.values()
        )
        rows: list[dict[str, float | int | str]] = []
        for category, item in self.accumulators.items():
            rows.append(
                {
                    "progress": target_fraction,
                    "phase_start_step": self.phase_start_step,
                    "phase_end_step": global_step,
                    "category": category,
                    "cell_count": item.cell_count,
                    "cell_count_share": item.cell_count / total_cells if total_cells else 0.0,
                    "mse_loss_sum": item.loss_sum,
                    "mse_loss_share": item.loss_sum / total_loss if total_loss else 0.0,
                    "logit_gradient_sum": item.gradient_sum,
                    "logit_gradient_share": (
                        item.gradient_sum / total_gradient if total_gradient else 0.0
                    ),
                    "objective_loss_sum": item.objective_loss_sum,
                    "objective_loss_share": (
                        item.objective_loss_sum / total_objective_loss
                        if total_objective_loss
                        else 0.0
                    ),
                    "objective_logit_gradient_sum": item.objective_gradient_sum,
                    "objective_logit_gradient_share": (
                        item.objective_gradient_sum / total_objective_gradient
                        if total_objective_gradient
                        else 0.0
                    ),
                    "mean_supervised_loss": (
                        self.supervised_loss_sum / self.batch_count
                        if self.batch_count
                        else 0.0
                    ),
                    "mean_consistency_loss": (
                        self.consistency_loss_sum / self.batch_count
                        if self.batch_count
                        else 0.0
                    ),
                    "mean_weighted_consistency_loss": (
                        self.weighted_consistency_loss_sum / self.batch_count
                        if self.batch_count
                        else 0.0
                    ),
                    "mean_total_loss": (
                        self.total_loss_sum / self.batch_count
                        if self.batch_count
                        else 0.0
                    ),
                }
            )
        self.next_checkpoint_index += 1
        self.phase_start_step = global_step
        self.accumulators = {
            category: CategoryAccumulator() for category in CATEGORIES
        }
        self.supervised_loss_sum = 0.0
        self.consistency_loss_sum = 0.0
        self.weighted_consistency_loss_sum = 0.0
        self.total_loss_sum = 0.0
        self.batch_count = 0
        return rows

    def state_dict(self) -> dict:
        return {
            "next_checkpoint_index": self.next_checkpoint_index,
            "phase_start_step": self.phase_start_step,
            "accumulators": {
                name: vars(value).copy() for name, value in self.accumulators.items()
            },
            "supervised_loss_sum": self.supervised_loss_sum,
            "consistency_loss_sum": self.consistency_loss_sum,
            "weighted_consistency_loss_sum": self.weighted_consistency_loss_sum,
            "total_loss_sum": self.total_loss_sum,
            "batch_count": self.batch_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.next_checkpoint_index = int(state["next_checkpoint_index"])
        self.phase_start_step = int(state["phase_start_step"])
        self.accumulators = {
            name: CategoryAccumulator(**values)
            for name, values in state["accumulators"].items()
        }
        self.supervised_loss_sum = float(state["supervised_loss_sum"])
        self.consistency_loss_sum = float(state["consistency_loss_sum"])
        self.weighted_consistency_loss_sum = float(
            state.get("weighted_consistency_loss_sum", 0.0)
        )
        self.total_loss_sum = float(state.get("total_loss_sum", 0.0))
        self.batch_count = int(state["batch_count"])

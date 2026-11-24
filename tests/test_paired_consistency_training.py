import random

import numpy as np
import torch
from omegaconf import OmegaConf

from explainable_medical_coding.trainer.trainer import Trainer
from explainable_medical_coding.eval.metrics import deterministic_cumsum
from explainable_medical_coding.trainer.consistency import (
    CATEGORIES,
    ConsistencyDiagnosticsTracker,
)
from explainable_medical_coding.utils.datatypes import Batch, Lookups
from explainable_medical_coding.utils.loss_functions import (
    _paired_consistency_divergence,
    _paired_consistency_term,
    paired_supervised_consistency_loss,
)
from explainable_medical_coding.utils.paired_augmentation import (
    AugmentationChoice,
    augment_training_text,
    perturb_format,
    perturb_section_reorder,
    perturb_spelling,
)
from scripts.evaluate_frequency_robustness import (
    perturb_format as formal_perturb_format,
    perturb_section_reorder as formal_perturb_section_reorder,
    perturb_spelling as formal_perturb_spelling,
)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 4)
        self.head = torch.nn.Linear(4, 3)

    def forward(self, input_ids, attention_masks):
        hidden = self.embedding(input_ids)
        mask = attention_masks.unsqueeze(-1)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        return self.head(pooled)


def make_batch(with_augmented=True):
    kwargs = {}
    if with_augmented:
        kwargs = {
            "augmented_input_ids": torch.tensor([[1, 3, 2], [1, 5, 2]]),
            "augmented_attention_masks": torch.ones(2, 3, dtype=torch.long),
        }
    return Batch(
        input_ids=torch.tensor([[1, 2, 0], [1, 4, 2]]),
        attention_masks=torch.tensor([[1, 1, 0], [1, 1, 1]]),
        targets=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        target_names=[["a"], ["b"]],
        ids=[1, 2],
        **kwargs,
    )


def test_training_augmentation_is_deterministic_and_single_condition():
    text = (
        "Metadata preamble that must stay fixed.\n"
        "HISTORY OF PRESENT ILLNESS:\n"
        "The patient arrived at the hospital. "
        "She reported persistent abdominal pain. "
        "Blood cultures remained negative overnight. "
        "The clinical team continued close monitoring.\n"
    )
    first = augment_training_text(text, record_key="note-7", seed=13)
    second = augment_training_text(text, record_key="note-7", seed=13)
    assert first == second
    assert first["augmentation_condition"] in {
        "spelling",
        "format",
        "section_reorder",
    }
    assert first["augmentation_severity"] in {
        0.01,
        0.03,
        0.05,
        0.10,
        1 / 3,
        2 / 3,
    }
    assert first["augmented_text"] != text


def test_format_augmentation_preserves_non_whitespace_characters():
    text = "A clinical line.\n\nAnother clinical line with spaces."
    augmented = perturb_format(text, 0.10, random.Random(9))
    assert "".join(text.split()) == "".join(augmented.split())


def test_ineligible_section_reorder_choice_falls_back_to_spelling():
    text = "Clinical narrative contains several meaningful words but no section header."
    result = augment_training_text(
        text,
        record_key="fallback-note",
        seed=13,
        choices=(AugmentationChoice("section_reorder", 2 / 3),),
    )
    assert result["augmentation_condition"] == "spelling"
    assert result["augmentation_severity"] == 0.01
    assert result["augmented_text"] != text


def test_training_perturbations_match_frozen_evaluation_implementations():
    text = "Clinical history with multiple meaningful words.\n\nA second line."
    assert perturb_spelling(text, 0.03, random.Random(17)) == formal_perturb_spelling(
        text, 0.03, random.Random(17)
    )
    assert perturb_format(text, 0.10, random.Random(17)) == formal_perturb_format(
        text, 0.10, random.Random(17)
    )
    sectioned_text = (
        "Metadata preamble.\n"
        "HOSPITAL COURSE:\n"
        "The first complete sentence is here. "
        "The second complete sentence is here. "
        "The third complete sentence is here. "
        "The fourth complete sentence is here.\n"
    )
    assert perturb_section_reorder(
        sectioned_text, 2 / 3, random.Random(17)
    ) == formal_perturb_section_reorder(
        sectioned_text, 2 / 3, random.Random(17)
    )


def test_lambda_zero_is_exact_b1_objective():
    torch.manual_seed(4)
    model = TinyModel().eval()
    batch = make_batch()
    output = paired_supervised_consistency_loss(
        batch, model=model, consistency_lambda=0.0
    )
    clean_logits = model(batch.input_ids, batch.attention_masks)
    augmented_logits = model(
        batch.augmented_input_ids, batch.augmented_attention_masks
    )
    expected = 0.5 * (
        torch.nn.functional.binary_cross_entropy_with_logits(
            clean_logits, batch.targets
        )
        + torch.nn.functional.binary_cross_entropy_with_logits(
            augmented_logits, batch.targets
        )
    )
    torch.testing.assert_close(output[2], expected, rtol=0, atol=0)


def test_validation_falls_back_to_clean_bce():
    model = TinyModel().eval()
    batch = make_batch(with_augmented=False)
    output = paired_supervised_consistency_loss(
        batch, model=model, consistency_lambda=0.3
    )
    assert len(output) == 3
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        model(batch.input_ids, batch.attention_masks), batch.targets
    )
    torch.testing.assert_close(output[2], expected)


def test_objective_coefficients_reconstruct_weighted_consistency_loss():
    torch.manual_seed(5)
    model = TinyModel().eval()
    batch = make_batch()
    output = paired_supervised_consistency_loss(
        batch,
        model=model,
        consistency_lambda=0.3,
        consistency_mode="split_normalized",
    )
    payload = output[3]
    mse = (
        payload["clean_probabilities"] - payload["augmented_probabilities"]
    ).square()
    reconstructed = (mse * payload["objective_coefficients"]).sum()
    torch.testing.assert_close(
        reconstructed, payload["weighted_consistency_loss"]
    )


def test_split_normalized_consistency_balances_positive_and_negative_means(tmp_path):
    clean = torch.tensor([[0.9, 0.6, 0.4, 0.2]], requires_grad=True)
    augmented = torch.zeros_like(clean)
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    value = _paired_consistency_term(
        clean,
        augmented,
        targets,
        mode="split_normalized",
        frequency_file=str(tmp_path / "unused.csv"),
        frequency_gamma=0.5,
        frequency_weight_max=5.0,
        hard_negative_k=1,
    )
    expected = 0.5 * (0.81 + (0.36 + 0.16 + 0.04) / 3)
    torch.testing.assert_close(value, torch.tensor(expected))
    value.backward()
    assert clean.grad is not None


def test_symmetric_kl_is_symmetric_finite_and_zero_on_identical_predictions():
    clean = torch.tensor([[0.01, 0.25, 0.8, 0.999999]])
    augmented = torch.tensor([[0.05, 0.7, 0.4, 0.000001]])
    forward = _paired_consistency_divergence(
        clean, augmented, divergence="symmetric_kl"
    )
    reverse = _paired_consistency_divergence(
        augmented, clean, divergence="symmetric_kl"
    )
    torch.testing.assert_close(forward, reverse)
    assert torch.isfinite(forward).all()
    assert bool((forward >= 0).all())
    identical = _paired_consistency_divergence(
        clean, clean, divergence="symmetric_kl"
    )
    torch.testing.assert_close(identical, torch.zeros_like(identical))


def test_symmetric_kl_can_replace_mse_without_changing_uniform_label_weights(
    tmp_path,
):
    clean = torch.tensor([[0.9, 0.6, 0.4, 0.2]], requires_grad=True)
    augmented = torch.tensor([[0.8, 0.2, 0.3, 0.1]])
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    value = _paired_consistency_term(
        clean,
        augmented,
        targets,
        mode="uniform",
        frequency_file=str(tmp_path / "unused.csv"),
        frequency_gamma=0.5,
        frequency_weight_max=5.0,
        hard_negative_k=1,
        divergence="symmetric_kl",
    )
    expected = _paired_consistency_divergence(
        clean, augmented, divergence="symmetric_kl"
    ).mean()
    torch.testing.assert_close(value, expected)
    value.backward()
    assert clean.grad is not None


def test_frequency_positive_upweights_rare_positive(tmp_path):
    frequency_file = tmp_path / "frequencies.csv"
    frequency_file.write_text(
        "code,train_frequency\na,1\nb,100\nc,10\nd,10\n", encoding="utf-8"
    )
    clean = torch.tensor([[0.9, 0.6, 0.4, 0.2]])
    augmented = torch.zeros_like(clean)
    targets = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    value = _paired_consistency_term(
        clean,
        augmented,
        targets,
        mode="frequency_positive",
        frequency_file=str(frequency_file),
        frequency_gamma=0.5,
        frequency_weight_max=5.0,
        hard_negative_k=1,
    )
    weighted_positive = (0.81 * 1.0 + 0.36 * 0.1) / 1.1
    expected = 0.5 * (weighted_positive + (0.16 + 0.04) / 2)
    torch.testing.assert_close(value, torch.tensor(expected))


def test_frequency_positive_relative_cap_limits_weight_ratio(tmp_path):
    frequency_file = tmp_path / "frequencies.csv"
    frequency_file.write_text(
        "code,train_frequency\na,1\nb,100\nc,10\nd,10\n", encoding="utf-8"
    )
    clean = torch.tensor([[0.9, 0.6, 0.4, 0.2]])
    augmented = torch.zeros_like(clean)
    targets = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    value = _paired_consistency_term(
        clean,
        augmented,
        targets,
        mode="frequency_positive",
        frequency_file=str(frequency_file),
        frequency_gamma=0.5,
        frequency_weight_max=5.0,
        frequency_weight_ratio_max=5.0,
        hard_negative_k=1,
    )
    weighted_positive = (0.81 * 1.0 + 0.36 * 0.2) / 1.2
    expected = 0.5 * (weighted_positive + (0.16 + 0.04) / 2)
    torch.testing.assert_close(value, torch.tensor(expected))


def test_topk_hard_negative_uses_highest_probability_negative(tmp_path):
    frequency_file = tmp_path / "frequencies.csv"
    frequency_file.write_text(
        "code,train_frequency\na,1\nb,10\nc,10\nd,10\n", encoding="utf-8"
    )
    clean = torch.tensor([[0.9, 0.6, 0.4, 0.2]])
    augmented = torch.zeros_like(clean)
    targets = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    value = _paired_consistency_term(
        clean,
        augmented,
        targets,
        mode="topk_hard_negative",
        frequency_file=str(frequency_file),
        frequency_gamma=0.5,
        frequency_weight_max=5.0,
        hard_negative_k=1,
    )
    torch.testing.assert_close(value, torch.tensor(0.5 * (0.81 + 0.36)))


def test_seven_category_diagnostics_close_and_round_trip():
    tracker = ConsistencyDiagnosticsTracker(
        label_frequencies=torch.tensor([0, 5, 50, 500, 500, 500, 500]),
        total_steps=10,
        decision_boundary=0.4,
    )
    clean = torch.tensor([[0.2, 0.6, 0.4, 0.8, 0.001, 0.2, 0.7]])
    augmented = torch.tensor([[0.3, 0.5, 0.3, 0.7, 0.002, 0.3, 0.6]])
    targets = torch.tensor([[1, 1, 1, 1, 0, 0, 0]], dtype=torch.float32)
    tracker.update(
        {
            "clean_probabilities": clean,
            "augmented_probabilities": augmented,
            "targets": targets,
            "supervised_loss": torch.tensor(0.8),
            "consistency_loss": torch.tensor(0.01),
        }
    )
    rows = tracker.pop_due_rows(global_step=1)
    assert {row["category"] for row in rows} == set(CATEGORIES)
    assert sum(row["cell_count"] for row in rows) == targets.numel()
    assert abs(sum(row["cell_count_share"] for row in rows) - 1.0) < 1e-7
    assert abs(sum(row["mse_loss_share"] for row in rows) - 1.0) < 1e-6
    assert abs(sum(row["logit_gradient_share"] for row in rows) - 1.0) < 1e-6
    assert abs(sum(row["objective_loss_share"] for row in rows) - 1.0) < 1e-6
    assert (
        abs(sum(row["objective_logit_gradient_share"] for row in rows) - 1.0)
        < 1e-6
    )

    restored = ConsistencyDiagnosticsTracker(
        label_frequencies=tracker.label_frequencies,
        total_steps=10,
        decision_boundary=0.4,
    )
    restored.load_state_dict(tracker.state_dict())
    assert restored.state_dict() == tracker.state_dict()


def test_checkpoint_restores_optimizer_scheduler_rng_and_tracker(tmp_path):
    model = TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    config = OmegaConf.create(
        {
            "trainer": {
                "validate_on_training_data": False,
                "print_metrics": False,
                "epochs": 3,
                "use_amp": False,
                "threshold_tuning": False,
                "clip_grad_norm": 1,
                "max_optimizer_steps": None,
            }
        }
    )
    tracker = ConsistencyDiagnosticsTracker(
        label_frequencies=torch.tensor([1, 20, 200]),
        total_steps=3,
        decision_boundary=0.4,
    )
    trainer = Trainer(
        config=config,
        model=model,
        optimizer=optimizer,
        dataloaders={},
        metric_collections={},
        callbacks=[],
        lookups=Lookups(
            data_info={"num_classes": 3},
            target_tokenizer=None,
            split2code_indices={},
        ),
        loss_function=paired_supervised_consistency_loss,
        lr_scheduler=scheduler,
        consistency_tracker=tracker,
    )
    trainer.experiment_path = tmp_path
    trainer.epoch = 1
    trainer.global_step = 7
    torch.manual_seed(123)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(123)
    np.random.seed(123)
    random.seed(123)
    trainer.save_checkpoint("resume.pt")
    expected_torch = torch.rand(3)
    expected_cuda = (
        torch.rand(3, device="cuda").cpu() if torch.cuda.is_available() else None
    )
    expected_numpy = np.random.random(3)
    expected_python = [random.random() for _ in range(3)]

    trainer.epoch = 0
    trainer.global_step = 0
    torch.manual_seed(999)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(999)
        trainer.to("cuda")
    np.random.seed(999)
    random.seed(999)
    trainer.load_checkpoint(tmp_path / "resume.pt")

    assert trainer.epoch == 2
    assert trainer.global_step == 7
    torch.testing.assert_close(torch.rand(3), expected_torch)
    if expected_cuda is not None:
        torch.testing.assert_close(torch.rand(3, device="cuda").cpu(), expected_cuda)
    np.testing.assert_allclose(np.random.random(3), expected_numpy)
    assert [random.random() for _ in range(3)] == expected_python


def test_metric_cumsum_supports_deterministic_cuda():
    if not torch.cuda.is_available():
        return
    was_enabled = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        values = torch.tensor([[1.0, 0.0, 1.0]], device="cuda")
        result = deterministic_cumsum(values, dim=1)
        torch.testing.assert_close(
            result.cpu(), torch.tensor([[1.0, 1.0, 2.0]])
        )
    finally:
        torch.use_deterministic_algorithms(was_enabled)

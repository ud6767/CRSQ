import unittest

import torch

from explainable_medical_coding.models.modules.attention import (
    DifferentialLabelCrossAttention,
    LabelAttention,
    LabelCrossAttention,
    masked_softmax,
)


class AttentionMaskTest(unittest.TestCase):
    def test_padding_probability_is_zero_for_all_label_heads(self):
        torch.manual_seed(3)
        hidden = torch.randn(2, 5, 6)
        mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])

        label_attention = LabelAttention(6, 4, 3)
        output, probabilities = label_attention(
            hidden, attention_masks=mask, output_attention=True
        )
        self.assertTrue(torch.isfinite(output).all())
        padding_probabilities = probabilities.masked_select(~mask[:, None, :].bool())
        self.assertLess(padding_probabilities.max().item(), 1e-6)

        cross_attention = LabelCrossAttention(6, 3)
        output, scores = cross_attention(
            hidden, attention_masks=mask, output_attention=True
        )
        probabilities = masked_softmax(scores, mask, token_dim=2)
        self.assertTrue(torch.isfinite(output).all())
        padding_probabilities = probabilities.masked_select(~mask[:, None, :].bool())
        self.assertLess(padding_probabilities.max().item(), 1e-6)

        differential_attention = DifferentialLabelCrossAttention(6, 3)
        components = differential_attention.compute_attention_components(hidden, mask)
        padding_mask = ~mask[:, None, :].bool()
        self.assertLess(
            components["positive_attention"].masked_select(padding_mask).max().item(),
            1e-6,
        )
        self.assertLess(
            components["negative_attention"].masked_select(padding_mask).max().item(),
            1e-6,
        )

    def test_fully_masked_rows_are_zero_and_finite(self):
        scores = torch.randn(2, 4, 3)
        mask = torch.tensor([[0, 0, 0, 0], [1, 1, 0, 0]])
        probabilities = masked_softmax(scores, mask, token_dim=1)
        self.assertTrue(torch.isfinite(probabilities).all())
        self.assertTrue(
            torch.equal(probabilities[0], torch.zeros_like(probabilities[0]))
        )
        self.assertLess(probabilities[1, 2:].max().item(), 1e-6)

    def test_masked_softmax_supports_half_precision_dtypes(self):
        mask = torch.tensor([[1, 1, 0, 0]])
        for dtype in (torch.float16, torch.bfloat16):
            scores = torch.tensor(
                [[[1.0], [2.0], [100.0], [100.0]]],
                dtype=dtype,
            )
            probabilities = masked_softmax(scores, mask, token_dim=1)
            self.assertTrue(torch.isfinite(probabilities).all())
            self.assertLess(probabilities[:, 2:].max().float().item(), 1e-6)
            self.assertAlmostEqual(
                probabilities.sum(dim=1).float().item(),
                1.0,
                places=3,
            )


if __name__ == "__main__":
    unittest.main()

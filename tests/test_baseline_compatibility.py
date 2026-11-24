import types
import unittest
from unittest.mock import patch

import torch
from torch import nn

from explainable_medical_coding.models.models import PLMICD
from explainable_medical_coding.models.modules.attention import LabelCrossAttention


class BaselineCompatibilityTest(unittest.TestCase):
    @patch("explainable_medical_coding.models.models.AutoModel.from_pretrained")
    @patch("explainable_medical_coding.models.models.AutoConfig.from_pretrained")
    def test_default_and_explicitly_disabled_evidence_configs_match(
        self,
        config_loader,
        model_loader,
    ):
        config_loader.return_value = types.SimpleNamespace(
            hidden_size=6,
            initializer_range=0.02,
        )
        model_loader.return_value = nn.Identity()

        torch.manual_seed(11)
        default_model = PLMICD(3, "unused", 4, 0)
        torch.manual_seed(11)
        disabled_model = PLMICD(3, "unused", 4, 0, use_evidence_module=False)

        self.assertFalse(default_model.use_evidence_module)
        self.assertEqual(
            default_model.state_dict().keys(), disabled_model.state_dict().keys()
        )
        for name, value in default_model.state_dict().items():
            self.assertTrue(torch.equal(value, disabled_model.state_dict()[name]), name)
        disabled_model.load_state_dict(default_model.state_dict(), strict=True)

    def test_all_valid_mask_preserves_baseline_logits(self):
        torch.manual_seed(5)
        attention = LabelCrossAttention(input_size=6, num_classes=3)
        hidden = torch.randn(2, 4, 6)
        without_mask = attention(hidden)
        with_mask = attention(hidden, attention_masks=torch.ones(2, 4))
        self.assertTrue(torch.equal(without_mask, with_mask))

    @patch("explainable_medical_coding.models.models.AutoModel.from_pretrained")
    @patch("explainable_medical_coding.models.models.AutoConfig.from_pretrained")
    def test_forward_baseline_exposes_b_t_l_attention(
        self,
        config_loader,
        model_loader,
    ):
        config_loader.return_value = types.SimpleNamespace(
            hidden_size=6,
            initializer_range=0.02,
        )
        model_loader.return_value = nn.Identity()
        model = PLMICD(3, "unused", 4, 0)
        hidden = torch.randn(2, 4, 6)
        mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])

        with patch.object(model, "encoder", return_value=hidden):
            result = model.forward_baseline(torch.zeros(2, 4, dtype=torch.long), mask)
            legacy_logits = model.label_wise_attention(hidden, attention_masks=mask)

        self.assertTrue(torch.equal(result["baseline_logits"], legacy_logits))
        self.assertEqual(result["baseline_attention"].shape, (2, 4, 3))
        padding = ~mask[:, :, None].bool()
        self.assertLess(
            result["baseline_attention"].masked_select(padding).max().item(),
            1e-6,
        )


if __name__ == "__main__":
    unittest.main()

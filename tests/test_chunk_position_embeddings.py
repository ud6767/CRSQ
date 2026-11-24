import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from explainable_medical_coding.models.models import PLMICD


class ChunkPositionEmbeddingTest(unittest.TestCase):
    def make_model(self, enabled=True, max_chunks=4):
        with (
            patch(
                "explainable_medical_coding.models.models.AutoConfig.from_pretrained",
                return_value=SimpleNamespace(hidden_size=8, initializer_range=0.02),
            ),
            patch(
                "explainable_medical_coding.models.models.AutoModel.from_pretrained",
                return_value=torch.nn.Identity(),
            ),
        ):
            return PLMICD(
                num_classes=3,
                model_path="unused",
                chunk_size=4,
                pad_token_id=1,
                add_chunk_position_embeddings=enabled,
                max_chunks=max_chunks,
            )

    def test_disabled_positions_are_identity(self):
        model = self.make_model(enabled=False)
        hidden_output = torch.randn(2, 8, model.config.hidden_size)
        positioned = model.add_global_chunk_positions(hidden_output, num_chunks=2)
        self.assertIs(positioned, hidden_output)

    def test_each_chunk_receives_its_position_vector(self):
        model = self.make_model()
        with torch.no_grad():
            model.chunk_position_embeddings.weight.zero_()
            model.chunk_position_embeddings.weight[0].fill_(1.0)
            model.chunk_position_embeddings.weight[1].fill_(2.0)

        hidden_output = torch.zeros(1, 8, model.config.hidden_size)
        positioned = model.add_global_chunk_positions(hidden_output, num_chunks=2)

        self.assertTrue(torch.all(positioned[:, :4] == 1.0))
        self.assertTrue(torch.all(positioned[:, 4:] == 2.0))

    def test_too_many_chunks_raise_clear_error(self):
        model = self.make_model(max_chunks=1)
        hidden_output = torch.zeros(1, 8, model.config.hidden_size)
        with self.assertRaisesRegex(ValueError, "exceeding max_chunks"):
            model.add_global_chunk_positions(hidden_output, num_chunks=2)


if __name__ == "__main__":
    unittest.main()

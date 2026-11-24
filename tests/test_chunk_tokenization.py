import unittest

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing
from transformers import PreTrainedTokenizerFast

from explainable_medical_coding.utils.tokenizer import (
    get_chunked_content_max_length,
    tokenize_texts_with_chunk_special_tokens,
)


class WordTokenizer(PreTrainedTokenizerFast):
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        return [self.bos_token_id, *token_ids_0, self.eos_token_id]


class ChunkSpecialTokenizationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        words = "<unk> <s> </s> alpha beta gamma delta epsilon zeta eta theta"
        vocab = {word: index for index, word in enumerate(words.split())}
        tokenizer = Tokenizer(WordLevel(vocab, unk_token="<unk>"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer.post_processor = TemplateProcessing(
            single="<s> $A </s>", special_tokens=[("<s>", 1), ("</s>", 2)]
        )
        cls.tokenizer = WordTokenizer(
            tokenizer_object=tokenizer, unk_token="<unk>", bos_token="<s>", eos_token="</s>"
        )

    def test_content_budget_stays_within_model_budget(self):
        self.assertEqual(get_chunked_content_max_length(6000, 128, 2), 5906)
        self.assertEqual(get_chunked_content_max_length(512, 128, 2), 504)
        self.assertEqual(get_chunked_content_max_length(131, 128, 2), 127)

    def test_every_chunk_has_bos_and_eos(self):
        result = tokenize_texts_with_chunk_special_tokens(
            texts=["alpha beta gamma delta epsilon zeta eta theta " * 20],
            text_tokenizer=self.tokenizer,
            max_length=32,
            chunk_size=8,
        )
        input_ids = result["input_ids"][0]

        self.assertLessEqual(len(input_ids), 32)
        self.assertEqual(len(input_ids) % 8, 0)
        for start in range(0, len(input_ids), 8):
            chunk = input_ids[start : start + 8]
            self.assertEqual(chunk[0], self.tokenizer.bos_token_id)
            self.assertEqual(chunk[-1], self.tokenizer.eos_token_id)
        self.assertEqual(result["attention_mask"][0], [1] * len(input_ids))
        self.assertEqual(result["length"][0], len(input_ids))

    def test_empty_document_still_has_special_tokens(self):
        result = tokenize_texts_with_chunk_special_tokens(
            texts=[""],
            text_tokenizer=self.tokenizer,
            max_length=128,
            chunk_size=128,
        )
        self.assertEqual(
            result["input_ids"][0],
            [self.tokenizer.bos_token_id, self.tokenizer.eos_token_id],
        )
        self.assertEqual(result["attention_mask"][0], [1, 1])
        self.assertEqual(result["length"][0], 2)

    def test_offsets_survive_independent_chunk_wrapping(self):
        text = "alpha beta gamma delta epsilon zeta eta theta " * 4
        result = tokenize_texts_with_chunk_special_tokens(
            texts=[text],
            text_tokenizer=self.tokenizer,
            max_length=32,
            chunk_size=8,
            return_offsets_mapping=True,
        )
        input_ids = result["input_ids"][0]
        offsets = result["offset_mapping"][0]

        self.assertEqual(len(input_ids), len(offsets))
        for start in range(0, len(input_ids), 8):
            chunk_offsets = offsets[start : start + 8]
            self.assertEqual(tuple(chunk_offsets[0]), (0, 0))
            self.assertEqual(tuple(chunk_offsets[-1]), (0, 0))
            for token_start, token_end in chunk_offsets[1:-1]:
                self.assertLess(token_start, token_end)
                self.assertTrue(text[token_start:token_end])


if __name__ == "__main__":
    unittest.main()

import unittest
from typing import Any, cast

import torch

from looped_transformer.domains.text_classification.tokenizer import PairTokenizer


class _BackendTokenizer:
    def __call__(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        self.kwargs = kwargs
        return {
            "input_ids": torch.ones(2, 7, dtype=torch.long),
            "attention_mask": torch.ones(2, 7, dtype=torch.long),
            "token_type_ids": torch.zeros(2, 7, dtype=torch.long),
        }


class PairTokenizerTests(unittest.TestCase):
    def test_encode_dynamically_pads_to_longest_batch_item(self) -> None:
        tokenizer = PairTokenizer.__new__(PairTokenizer)
        backend = _BackendTokenizer()
        tokenizer._tokenizer = cast(Any, backend)
        tokenizer._max_sequence_length = 256

        result = tokenizer.encode(("short", "longer premise"), ("q1", "q2"))

        self.assertIs(backend.kwargs["padding"], True)
        self.assertNotIn("max_length", backend.kwargs)
        self.assertIs(backend.kwargs["truncation"], False)
        self.assertEqual(tuple(result.input_ids.shape), (2, 7))

    def test_encode_rejects_inputs_over_the_model_limit(self) -> None:
        tokenizer = PairTokenizer.__new__(PairTokenizer)
        tokenizer._tokenizer = cast(Any, _BackendTokenizer())
        tokenizer._max_sequence_length = 6

        with self.assertRaisesRegex(ValueError, "7 > 6"):
            tokenizer.encode(("premise", "premise"), ("q1", "q2"))


if __name__ == "__main__":
    unittest.main()

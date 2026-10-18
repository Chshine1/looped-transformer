import torch

from looped_transformer.tokenizer import MNLITokenizer
import unittest


class TokenizerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._premises = [
            "Conceptually cream skimming has two basic dimensions - product and geography.",
            "you know during the season and i guess at at your level uh you lose them to the next level if if they decide to recall the the parent team the Braves decide to call to recall a guy from triple A then a double A guy goes up to replace him and a single A guy goes up to replace him",
        ]
        cls._hypotheses = [
            "Product and geography are what make cream skimming work. ",
            "You lose the things to the following level if the people recall.",
        ]

    def test_tokenizer_encodes_correctly(self) -> None:
        batch_size = len(self._premises)
        max_sequence_length = 128

        tokenizer = MNLITokenizer(max_sequence_length=max_sequence_length)
        result = tokenizer.encode(self._premises, self._hypotheses)

        hf_tokenizer = tokenizer._tokenizer

        self.assertEqual(result.input_ids.shape, (batch_size, max_sequence_length))
        self.assertEqual(result.attention_mask.shape, (batch_size, max_sequence_length))
        self.assertFalse(result.input_ids.is_floating_point())
        self.assertFalse(result.attention_mask.is_floating_point())

        self.assertTrue(torch.isin(result.attention_mask, torch.tensor([0, 1])).all())
        valid_lengths = result.attention_mask.sum(dim=1)
        self.assertTrue((valid_lengths > 0).all())

        for i in range(batch_size):
            n = int(valid_lengths[i])
            if bool(result.attention_mask[i, 0] == 1):
                self.assertTrue((result.attention_mask[i, :n] == 1).all())
                self.assertTrue((result.attention_mask[i, n:] == 0).all())
            else:
                self.assertTrue((result.attention_mask[i, -n:] == 1).all())
                self.assertTrue((result.attention_mask[i, :-n] == 0).all())

        for i in range(batch_size):
            n = int(valid_lengths[i])
            tokens = hf_tokenizer.convert_ids_to_tokens(result.input_ids[i].tolist())
            real_tokens = (
                tokens[:n] if bool(result.attention_mask[i, 0] == 1) else tokens[-n:]
            )
            print(real_tokens)
            sep_count = sum(1 for t in real_tokens if t in ("[SEP]", "</s>"))
            self.assertEqual(sep_count, 2)

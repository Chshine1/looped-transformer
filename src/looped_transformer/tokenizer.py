import os
from collections.abc import Sequence
from typing import cast

from transformers import TensorType

# noinspection protected-member
from transformers.models.bert.tokenization_bert import BertTokenizerFast
from transformers.utils import PaddingStrategy

from looped_transformer.config import ModelOptions
from looped_transformer.types.tokenizer import TokenizeResult, TokenizerInfo


class MNLITokenizer:
    def __init__(self, model_options: ModelOptions):
        self._model_options = model_options
        offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in {
            "1",
            "true",
            "yes",
        }
        source = "local cache only" if offline else "local cache or Hugging Face Hub"
        print(f"Loading bert-base-uncased tokenizer ({source})...", flush=True)
        self._tokenizer = BertTokenizerFast.from_pretrained(
            "bert-base-uncased",
            local_files_only=offline,
        )
        print("Tokenizer loaded.", flush=True)

    def get_tokenizer_info(self) -> TokenizerInfo:
        return TokenizerInfo(
            vocab_size=self._tokenizer.vocab_size,
            pad_token_id=cast(int, self._tokenizer.pad_token_id),
            mask_token_id=cast(int, self._tokenizer.mask_token_id),
            special_token_ids=tuple(self._tokenizer.all_special_ids),
        )

    def encode(
        self, premises: Sequence[str], hypotheses: Sequence[str]
    ) -> TokenizeResult:
        # noinspection bad-argument-type
        encoding = self._tokenizer(
            text=premises,
            text_pair=hypotheses,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=True,
            max_length=self._model_options.max_sequence_length,
            return_tensors=TensorType.PYTORCH,
        )
        return TokenizeResult(
            input_ids=encoding["input_ids"],
            attention_mask=encoding["attention_mask"],
        )

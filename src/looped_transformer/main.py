from typing import cast

import torch
from transformers import TensorType
from transformers.models.bert.tokenization_bert import BertTokenizerFast
from transformers.utils import PaddingStrategy

from looped_transformer.config.model import ModelOptions
from looped_transformer.config.tokenizer import TokenizerOptions
from looped_transformer.model import LoopedMNLI
from looped_transformer.types.mnli import LogicRelation
from looped_transformer.types.model import ModelEvalOutput


class MNLIRunner:
    def __init__(self, model_options: ModelOptions):
        self._model_options = model_options

        tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(
            "bert-base-uncased"
        )
        tokenizer_options: TokenizerOptions = TokenizerOptions(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=cast(int, tokenizer.convert_tokens_to_ids("[PAD]")),
        )

        self._tokenizer = tokenizer
        self._model = LoopedMNLI(model_options, tokenizer_options)

    def train(self):
        pass

    def evaluate(
        self, premise_texts: list[str], hypothesis_texts: list[str]
    ) -> tuple[LogicRelation, ...]:
        encoding = self._tokenizer(
            text=premise_texts,
            text_pair=hypothesis_texts,
            padding=PaddingStrategy.MAX_LENGTH,
            truncation=True,
            max_length=self._model_options.max_sequence_length,
            return_tensors=TensorType.PYTORCH,
        )
        self._model.eval()
        with torch.no_grad():
            output: ModelEvalOutput = self._model(
                encoding["input_ids"], encoding["attention_mask"]
            )
        return tuple(LogicRelation(p.item()) for p in output["logits"].argmax(dim=-1))

import math
from typing import cast

import pandas as pd
import torch
from torch.nn import functional
from transformers import TensorType
# noinspection protected-member
from transformers.models.bert.tokenization_bert import BertTokenizerFast
from transformers.utils import PaddingStrategy

from looped_transformer.config.data import DataOptions
from looped_transformer.config.model import ModelOptions
from looped_transformer.config.tokenizer import TokenizerOptions
from looped_transformer.loss import LoopedMNLILoss
from looped_transformer.model import LoopedMNLI
from looped_transformer.types.mnli import LogicRelation, get_logic_relation
from looped_transformer.types.model import ModelEvalOutput, ModelTrainOutput


class MNLIRunner:
    def __init__(self, model_options: ModelOptions, data_options: DataOptions):
        self._model_options = model_options
        self._data_options = data_options

        tokenizer: BertTokenizerFast = BertTokenizerFast.from_pretrained(
            "bert-base-uncased"
        )
        tokenizer_options: TokenizerOptions = TokenizerOptions(
            vocab_size=tokenizer.vocab_size,
            pad_token_id=cast(int, tokenizer.convert_tokens_to_ids("[PAD]")),
        )

        self._tokenizer = tokenizer
        self._model = LoopedMNLI(model_options, tokenizer_options)
        self._loss = LoopedMNLILoss(expected_time=4, halt_scale=0.05, aux_scale=0.1)

    def train(self):
        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=3e-5,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01,
        )

        dataset_rows_count = sum(1 for _ in open(self._data_options.train_file_path, encoding="utf-8")) - 1
        steps_per_epoch = math.ceil(dataset_rows_count / self._data_options.batch_size)
        total_steps = self._model_options.training.max_epochs * steps_per_epoch
        warmup_steps = max(1, int(0.1 * total_steps))

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_annealing = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_steps - warmup_steps),
            eta_min=1e-6,
        )

        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            [warmup, cosine_annealing],
            [warmup_steps]
        )

        for epoch in range(self._model_options.training.max_epochs):
            # noinspection argument-list
            for chunk in pd.read_csv(
                self._data_options.train_file_path,
                sep="\t",
                chunksize=self._data_options.batch_size,
                on_bad_lines="skip",
            ):
                premises: list[str] = chunk["sentence1"].tolist()
                hypotheses: list[str] = chunk["sentence2"].tolist()

                encoding = self._tokenizer(
                    text=premises,
                    text_pair=hypotheses,
                    padding=PaddingStrategy.MAX_LENGTH,
                    truncation=True,
                    max_length=self._model_options.max_sequence_length,
                    return_tensors=TensorType.PYTORCH,
                )

                self._model.train()
                output: ModelTrainOutput = self._model(
                    encoding["input_ids"], encoding["attention_mask"]
                )

                labels: list[str] = chunk["gold_label"].tolist()
                label_indices = torch.tensor(
                    [get_logic_relation(label) for label in labels], dtype=torch.long
                )
                label_tensor = functional.one_hot(
                    label_indices, num_classes=len(LogicRelation)
                ).to(torch.long)

                loss = self._loss(output, label_tensor)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()

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

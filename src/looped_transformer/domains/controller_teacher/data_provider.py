from typing import override

from torch import Tensor

from looped_transformer.core.abstractions.training import IDataProvider
from looped_transformer.core.evaluation import Evaluator
from looped_transformer.domains.controller_teacher.batch import (
    ControllerTeacherBatch,
)
from looped_transformer.domains.controller_teacher.collection_policy import (
    ControllerTeacherCollectionPolicy,
)
from looped_transformer.domains.mnli.types.batch import MNLIBatch
from looped_transformer.domains.mnli.types.model import ModelForwardOutput


class ControllerTeacherDataProvider(IDataProvider[ControllerTeacherBatch]):
    """Stream MNLI batches through a frozen backbone and yield its trajectories."""

    def __init__(
        self,
        source: IDataProvider[MNLIBatch],
        evaluator: Evaluator[MNLIBatch, Tensor, Tensor, ModelForwardOutput, Tensor],
        collection_policy: ControllerTeacherCollectionPolicy,
        training_batch_size: int,
    ) -> None:
        if training_batch_size <= 0:
            raise ValueError("training_batch_size must be positive")
        self._source = source
        self._evaluator = evaluator
        self._collection_policy = collection_policy
        self._training_batch_size = training_batch_size
        self._pending: ControllerTeacherBatch | None = None
        self._pending_offset = 0
        self._open = False

    @override
    def reset_source(self, shuffle: bool, seed: int | None) -> None:
        self.close()
        self._source.reset_source(shuffle=shuffle, seed=seed)
        try:
            self._evaluator.open()
        except BaseException:
            self._source.close()
            raise
        self._open = True

    @override
    def fetch(self) -> ControllerTeacherBatch | None:
        if not self._open:
            raise ValueError("A data source must be set first before fetching batches")

        try:
            if self._pending is None:
                batch = self._source.fetch()
                if batch is None:
                    self.close()
                    return None
                self._evaluator.evaluate_batch(batch)
                self._pending = self._collection_policy.take_batch()
                self._pending_offset = 0
            return self._take_training_batch()
        except BaseException:
            self.close()
            raise

    @override
    def get_samples_count(self) -> int:
        return self._source.get_samples_count()

    @override
    def get_batch_size(self) -> int:
        return self._training_batch_size

    @override
    def close(self) -> None:
        if not self._open:
            return
        self._open = False
        self._pending = None
        self._pending_offset = 0
        try:
            self._evaluator.close()
        finally:
            self._source.close()

    def _take_training_batch(self) -> ControllerTeacherBatch:
        pending = self._pending
        if pending is None:
            raise RuntimeError("No collected backbone batch is pending")

        start = self._pending_offset
        stop = min(start + self._training_batch_size, len(pending.mnli.labels))
        self._pending_offset = stop
        if stop == len(pending.mnli.labels):
            self._pending = None
            self._pending_offset = 0

        return ControllerTeacherBatch(
            mnli=MNLIBatch(
                premises=pending.mnli.premises[start:stop],
                hypotheses=pending.mnli.hypotheses[start:stop],
                labels=pending.mnli.labels[start:stop],
            ),
            timed_hidden_states=pending.timed_hidden_states[:, start:stop],
            padding_mask=pending.padding_mask[start:stop],
            timed_logits=pending.timed_logits[:, start:stop],
        )

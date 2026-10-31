from typing import override

from torch import Tensor

from looped_transformer.core.abstractions.training import IDataProvider
from looped_transformer.core.evaluation import Evaluator
from looped_transformer.domains.clutrr.types import CLUTRRBatch
from looped_transformer.domains.text_classification.types import ModelForwardOutput
from looped_transformer.domains.trajectory_readout.batch import TrajectoryBatch
from looped_transformer.domains.trajectory_readout.collection_policy import (
    TrajectoryCollectionPolicy,
)


class TrajectoryDataProvider(IDataProvider[TrajectoryBatch]):
    """Stream source batches through a frozen backbone and yield trajectories."""

    def __init__(
        self,
        source: IDataProvider[CLUTRRBatch],
        evaluator: Evaluator[
            CLUTRRBatch, Tensor, Tensor, Tensor, ModelForwardOutput, Tensor
        ],
        collection_policy: TrajectoryCollectionPolicy,
        output_batch_size: int,
    ) -> None:
        if output_batch_size <= 0:
            raise ValueError("output_batch_size must be positive")
        self._source = source
        self._evaluator = evaluator
        self._collection_policy = collection_policy
        self._output_batch_size = output_batch_size
        self._pending: TrajectoryBatch | None = None
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
    def fetch(self) -> TrajectoryBatch | None:
        if not self._open:
            raise ValueError("A data source must be set before fetching batches")
        try:
            if self._pending is None:
                batch = self._source.fetch()
                if batch is None:
                    self.close()
                    return None
                self._evaluator.evaluate_batch(batch)
                self._pending = self._collection_policy.take_batch()
                self._pending_offset = 0
            return self._take_output_batch()
        except BaseException:
            self.close()
            raise

    @override
    def get_samples_count(self) -> int:
        return self._source.get_samples_count()

    @override
    def get_batch_size(self) -> int:
        return self._output_batch_size

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

    def _take_output_batch(self) -> TrajectoryBatch:
        pending = self._pending
        if pending is None:
            raise RuntimeError("No collected trajectory batch is pending")
        start = self._pending_offset
        stop = min(start + self._output_batch_size, len(pending.source.labels))
        self._pending_offset = stop
        if stop == len(pending.source.labels):
            self._pending = None
            self._pending_offset = 0
        return TrajectoryBatch(
            source=CLUTRRBatch(
                ids=pending.source.ids[start:stop],
                stories=pending.source.stories[start:stop],
                queries=pending.source.queries[start:stop],
                labels=pending.source.labels[start:stop],
                depths=pending.source.depths[start:stop],
            ),
            timed_hidden_states=pending.timed_hidden_states[:, start:stop],
            padding_mask=pending.padding_mask[start:stop],
            final_logits=pending.final_logits[start:stop],
        )

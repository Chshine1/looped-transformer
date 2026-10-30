from typing import Literal, cast

import torch
from torch import Tensor, nn

from looped_transformer.core.modules.masked_mean_pooling import MaskedMeanPooling

TrajectoryReadoutArchitecture = Literal["bidirectional", "causal"]


class BidirectionalTrajectoryReadout(nn.Module):
    """Pool tokens at every loop, then encode the complete loop sequence."""

    def __init__(
        self,
        state_dimension: int,
        encoding_dimension: int,
        classes_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if encoding_dimension % 2:
            raise ValueError("encoding_dimension must be even for a bidirectional GRU")
        self._input_normalization = nn.RMSNorm(state_dimension)
        self._pooling = MaskedMeanPooling()
        self._gru = nn.GRU(
            input_size=state_dimension,
            hidden_size=encoding_dimension // 2,
            bidirectional=True,
            batch_first=False,
        )
        self._output_normalization = nn.RMSNorm(encoding_dimension)
        self._dropout = nn.Dropout(dropout)
        self._classifier = nn.Linear(encoding_dimension, classes_count)

    def encode(self, timed_hidden_states: Tensor, padding_mask: Tensor) -> Tensor:
        pooled = _pool_loop_states(
            timed_hidden_states,
            padding_mask,
            self._input_normalization,
            self._pooling,
        )
        _, hidden = cast(tuple[Tensor, Tensor], self._gru(pooled))
        encoding = torch.cat((hidden[-2], hidden[-1]), dim=-1)
        return self._output_normalization(encoding)

    def forward(self, timed_hidden_states: Tensor, padding_mask: Tensor) -> Tensor:
        return self._classifier(
            self._dropout(self.encode(timed_hidden_states, padding_mask))
        )

    @property
    def classifier_weight(self) -> Tensor:
        return self._classifier.weight


class CausalTrajectoryReadout(nn.Module):
    """Pool tokens, then produce a class prediction from every loop prefix."""

    def __init__(
        self,
        state_dimension: int,
        encoding_dimension: int,
        classes_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self._input_normalization = nn.RMSNorm(state_dimension)
        self._pooling = MaskedMeanPooling()
        self._gru = nn.GRU(
            input_size=state_dimension,
            hidden_size=encoding_dimension,
            bidirectional=False,
            batch_first=False,
        )
        self._output_normalization = nn.RMSNorm(encoding_dimension)
        self._dropout = nn.Dropout(dropout)
        self._classifier = nn.Linear(encoding_dimension, classes_count)

    def encode_prefixes(
        self, timed_hidden_states: Tensor, padding_mask: Tensor
    ) -> Tensor:
        pooled = _pool_loop_states(
            timed_hidden_states,
            padding_mask,
            self._input_normalization,
            self._pooling,
        )
        prefixes, _ = cast(tuple[Tensor, Tensor], self._gru(pooled))
        return self._output_normalization(prefixes)

    def encode(self, timed_hidden_states: Tensor, padding_mask: Tensor) -> Tensor:
        return self.encode_prefixes(timed_hidden_states, padding_mask)[-1]

    def classify_prefixes(
        self, timed_hidden_states: Tensor, padding_mask: Tensor
    ) -> Tensor:
        return self._classifier(
            self._dropout(self.encode_prefixes(timed_hidden_states, padding_mask))
        )

    def forward(self, timed_hidden_states: Tensor, padding_mask: Tensor) -> Tensor:
        return self.classify_prefixes(timed_hidden_states, padding_mask)[-1]

    @property
    def classifier_weight(self) -> Tensor:
        return self._classifier.weight


def _pool_loop_states(
    timed_hidden_states: Tensor,
    padding_mask: Tensor,
    normalization: nn.Module,
    pooling: MaskedMeanPooling,
) -> Tensor:
    _validate_inputs(timed_hidden_states, padding_mask)
    loops_count, batch_size, sequence_length, dimension = timed_hidden_states.shape
    states = normalization(timed_hidden_states.reshape(-1, sequence_length, dimension))
    pooled = pooling(states, padding_mask.repeat(loops_count, 1))
    return pooled.reshape(loops_count, batch_size, dimension)


def _validate_inputs(timed_hidden_states: Tensor, padding_mask: Tensor) -> None:
    if timed_hidden_states.ndim != 4:
        raise ValueError(
            "timed_hidden_states must have shape [loops, batch, sequence, dimension]"
        )
    _, batch_size, sequence_length, _ = timed_hidden_states.shape
    if padding_mask.shape != (batch_size, sequence_length):
        raise ValueError("padding_mask shape does not match hidden states")

from torch.nn import functional
import torch
from torch import Tensor, nn

from looped_transformer.config import ModelOptions
from looped_transformer.evaluation import EvaluationPolicy
from looped_transformer.modules.controller_teacher import ControllerTeacher
from looped_transformer.modules.model import LoopedMNLI
from looped_transformer.types.data import MNLIBatch
from looped_transformer.types.model import ModelForwardOutput


class ControllerTeacherTrainingPolicy(EvaluationPolicy):
    class _ControllerTeacherLoss(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, loop_logits: Tensor, model_output: ModelForwardOutput, labels: Tensor) -> Tensor:
            logits = model_output["timed_logits"].detach()
            t, _, classes = logits.shape

            cross_entropy = functional.cross_entropy(
                logits.reshape(-1, classes),
                labels.repeat(t),
                reduction="none",
            ).reshape(t, -1)

            predictions = logits.argmax(dim=-1)
            correct = predictions.eq(labels).float()

            loop_probabilities = torch.softmax(loop_logits, dim=0)
            loss = (loop_probabilities * cross_entropy).sum(dim=0).mean()
            loss = loss + 0.5 * (loop_probabilities * (1 - correct)).sum(dim = 0).mean()

            return loss

    def __init__(self, model_options: ModelOptions):
        self._model = ControllerTeacher(model_options)
        self._batch_aligned_states: list[Tensor|None] = [None] * model_options.max_iterations
        self._loss = self._ControllerTeacherLoss()


    def reset(self, model: LoopedMNLI) -> None:
        def hook(_, input_args: tuple[int, Tensor], output: Tensor) -> None:
            rotation, _ = input_args
            self._batch_aligned_states[-rotation - 1] = output.clone().detach()

        for i in range(len(self._batch_aligned_states)):
            self._batch_aligned_states[i] = None
        # noinspection protected-member
        model._alignment_rotation.register_forward_hook(hook)


    def attend_batch(self, batch: MNLIBatch, output: ModelForwardOutput) -> None:
        aligned_hidden_states = torch.stack(tuple(t for t in self._batch_aligned_states if t is not None), dim=0)
        loop_logits: Tensor = self._model(aligned_hidden_states) # (t, batch)
        loss = self._loss(loop_logits, output, batch.labels)

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.hooks import RemovableHandle

from looped_transformer.config import ModelOptions, RuntimeOptions
from looped_transformer.evaluation import EvaluationPolicy
from looped_transformer.modules.controller_teacher import ControllerTeacher
from looped_transformer.modules.model import LoopedMNLI
from looped_transformer.types.data import MNLIBatch
from looped_transformer.types.mnli import get_logic_relation
from looped_transformer.types.model import ModelForwardOutput


class ControllerTeacherTrainingPolicy(EvaluationPolicy):
    class _Loss(nn.Module):
        def forward(
            self,
            loop_logits: Tensor,
            model_output: ModelForwardOutput,
            labels: Tensor,
        ) -> Tensor:
            logits = model_output["timed_logits"].detach().float()
            loop_count, _, classes = logits.shape

            cross_entropy = functional.cross_entropy(
                logits.reshape(-1, classes),
                labels.repeat(loop_count),
                reduction="none",
            ).reshape(loop_count, -1)
            incorrect = logits.argmax(dim=-1).ne(labels).float()
            loop_probabilities = torch.softmax(loop_logits.float(), dim=0)

            costs = cross_entropy + 0.5 * incorrect
            return (loop_probabilities * costs).sum(dim=0).mean()

    def __init__(
        self,
        model_options: ModelOptions,
        runtime_options: RuntimeOptions,
        epochs: int,
        learning_rate: float,
        weight_decay: float,
        max_gradient_norm: float,
    ) -> None:
        self._runtime_options = runtime_options
        self._max_gradient_norm = max_gradient_norm

        self._model = ControllerTeacher(model_options).to(runtime_options.device)
        self._loss = self._Loss()
        self._optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
        self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self._optimizer,
            T_max=epochs,
        )
        self._scaler = torch.amp.GradScaler(
            runtime_options.device.type,
            enabled=runtime_options.amp_enabled
            and runtime_options.amp_dtype == torch.float16,
        )

        self._batch_aligned_states: list[Tensor | None] = [
            None
        ] * model_options.max_iterations
        self._padding_mask: Tensor | None = None
        self._hook_handles: list[RemovableHandle] = []
        self._epoch_loss: Tensor | None = None
        self._epoch_batches = 0
        self._global_step = 0

    def reset(self, model: LoopedMNLI) -> None:
        self.close()
        self._model.train()
        self._epoch_loss = None
        self._epoch_batches = 0

        def capture_inputs(_: nn.Module, input_args: tuple[Tensor, Tensor]) -> None:
            self._batch_aligned_states[:] = [None] * len(self._batch_aligned_states)
            self._padding_mask = input_args[1].detach()

        def capture_aligned_state(
            _: nn.Module,
            input_args: tuple[int, Tensor],
            output: Tensor,
        ) -> None:
            rotation, _ignored_state = input_args
            self._batch_aligned_states[-rotation - 1] = output.detach()

        self._hook_handles.append(model.register_forward_pre_hook(capture_inputs))
        # noinspection protected-member
        self._hook_handles.append(
            model._alignment_rotation.register_forward_hook(capture_aligned_state)
        )

    def attend_batch(self, batch: MNLIBatch, output: ModelForwardOutput) -> None:
        padding_mask = self._padding_mask
        states = self._batch_aligned_states
        if padding_mask is None or any(state is None for state in states):
            raise RuntimeError(
                "The backbone did not produce a complete loop trajectory"
            )

        aligned_hidden_states = torch.stack(
            [state for state in states if state is not None], dim=0
        )
        labels = torch.tensor(
            [get_logic_relation(label).value for label in batch.labels],
            dtype=torch.long,
            device=self._runtime_options.device,
        )

        self._optimizer.zero_grad(set_to_none=True)
        with self._runtime_options.autocast:
            loop_logits = self._model(aligned_hidden_states, padding_mask)
            loss = self._loss(loop_logits, output, labels)

        self._scaler.scale(loss).backward()
        self._scaler.unscale_(self._optimizer)
        torch.nn.utils.clip_grad_norm_(
            self._model.parameters(), self._max_gradient_norm
        )
        self._scaler.step(self._optimizer)
        self._scaler.update()

        detached_loss = loss.detach()
        self._epoch_loss = (
            detached_loss
            if self._epoch_loss is None
            else self._epoch_loss + detached_loss
        )
        self._epoch_batches += 1
        self._global_step += 1

    def finish_epoch(self) -> float:
        if self._epoch_loss is None:
            raise RuntimeError("Cannot finish an epoch that contained no batches")
        mean_loss = (self._epoch_loss / self._epoch_batches).item()
        self._scheduler.step()
        return mean_loss

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "model": self._model.state_dict(),
            "optimizer": self._optimizer.state_dict(),
            "scheduler": self._scheduler.state_dict(),
            "scaler": self._scaler.state_dict(),
            "global_step": self._global_step,
        }

    def load_checkpoint_state(self, state: dict[str, Any]) -> None:
        self._model.load_state_dict(state["model"])
        self._optimizer.load_state_dict(state["optimizer"])
        self._scheduler.load_state_dict(state["scheduler"])
        self._scaler.load_state_dict(state["scaler"])
        self._global_step = int(state["global_step"])

    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

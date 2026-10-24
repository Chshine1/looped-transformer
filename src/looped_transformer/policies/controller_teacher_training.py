from typing import cast
from typing import override

import torch
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.hooks import RemovableHandle

from looped_transformer.abstractions.data import MNLIBatch
from looped_transformer.abstractions.mnli import get_logic_relation
from looped_transformer.abstractions.model import ModelForwardOutput
from looped_transformer.abstractions.model import TypedModule
from looped_transformer.config import ModelOptions, RuntimeOptions
from looped_transformer.core.evaluation import EvaluationPolicy
from looped_transformer.modules.controller_teacher import ControllerTeacher
from looped_transformer.modules.model import LoopedMNLI


class ControllerTeacherTrainingPolicy(
    EvaluationPolicy[MNLIBatch, Tensor, Tensor, ModelForwardOutput]
):
    class _Loss(nn.Module):
        # noinspection method-may-be-static
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

    @override
    def reset(self, model: TypedModule) -> None:
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

        handles: list[RemovableHandle] = []
        try:
            handles.append(model.torch.register_forward_pre_hook(capture_inputs))
            # noinspection protected-member
            handles.append(
                cast(LoopedMNLI, model.torch)._alignment_rotation.register_forward_hook(
                    capture_aligned_state
                )
            )
        except BaseException:
            for handle in handles:
                handle.remove()
            raise
        self._hook_handles.extend(handles)

    @override
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

    @override
    def close(self) -> None:
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

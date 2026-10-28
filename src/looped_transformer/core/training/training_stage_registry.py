from collections.abc import Callable

from torch import Tensor

from looped_transformer.core.abstractions.training import IDataProvider
from looped_transformer.core.abstractions.training.training_stage_template import (
    OptimizerFactory,
    SchedulerFactory,
    TrainingStage,
)
from looped_transformer.core.abstractions.typed_module import TypedModule
from looped_transformer.core.training.stage import StageTemplate, TrainingOptions


class TrainingStageRegistry[TBatch, TModelOutput, TLossInput]:
    def __init__(self, options: TrainingOptions) -> None:
        self._options = options
        self._factory: (
            Callable[[], TrainingStage[TBatch, TModelOutput, TLossInput]] | None
        ) = None

    def register[TTemplate: StageTemplate](
        self,
        template: type[TTemplate],
        data_provider_factory: Callable[[TTemplate], IDataProvider[TBatch]],
        loss_factory: Callable[
            [TTemplate], TypedModule[TModelOutput, TLossInput, Tensor]
        ],
        get_optimizer_factory: Callable[[TTemplate], OptimizerFactory],
        get_scheduler_factory: Callable[
            [TTemplate, IDataProvider[TBatch]], SchedulerFactory
        ],
    ) -> None:
        raw_stage = self._options.stage
        if template.type != raw_stage.get("type"):
            return
        if self._factory is not None:
            raise RuntimeError("The configured training stage is already registered")

        template_values = {
            key: value for key, value in raw_stage.items() if key != "type"
        }
        validated = template.model_validate(template_values)
        data_provider = data_provider_factory(validated)

        def get_stage() -> TrainingStage[TBatch, TModelOutput, TLossInput]:
            return TrainingStage(
                id=validated.id,
                stage_options=validated.options,
                data_provider=data_provider,
                loss=loss_factory(validated),
                optimizer_factory=get_optimizer_factory(validated),
                scheduler_factory=get_scheduler_factory(validated, data_provider),
            )

        self._factory = get_stage

    def resolve(self) -> TrainingStage[TBatch, TModelOutput, TLossInput]:
        if self._factory is None:
            stage_type = self._options.stage.get("type")
            raise KeyError(f"No training stage registered for type {stage_type!r}")
        return self._factory()

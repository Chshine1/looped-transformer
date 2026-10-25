from typing import Callable
from typing import TypedDict

from pydantic import JsonValue
from torch import Tensor

from looped_transformer.abstractions.model import TypedModule
from looped_transformer.abstractions.training import IDataProvider
from looped_transformer.abstractions.training.training_stage_template import (
    OptimizerFactory,
)
from looped_transformer.abstractions.training.training_stage_template import (
    SchedulerFactory,
)
from looped_transformer.abstractions.training.training_stage_template import (
    TrainingStage,
)
from looped_transformer.config.base import OptionsModel
from looped_transformer.stages.config import StageTemplate


class TrainingStageRegistryOptions(OptionsModel):
    class _StageTemplateModel(TypedDict, extra_items=JsonValue):
        type: str

    stage_templates: dict[str, _StageTemplateModel]


class TrainingStageRegistry[TBatch, TModelOutput, TLossInput]:
    def __init__(self, options: TrainingStageRegistryOptions) -> None:
        self._options = options
        self._registered_factories: dict[
            str, Callable[[], TrainingStage[TBatch, TModelOutput, TLossInput]]
        ]

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
        for stage_id, raw_dict in self._options.stage_templates.items():
            if template.type != raw_dict["type"]:
                continue
            validated = template.model_validate(raw_dict)
            data_provider = data_provider_factory(validated)

            def get_stage() -> TrainingStage[TBatch, TModelOutput, TLossInput]:
                return TrainingStage(
                    id=stage_id,
                    stage_options=validated.options,
                    data_provider=data_provider,
                    loss=loss_factory(validated),
                    optimizer_factory=get_optimizer_factory(validated),
                    scheduler_factory=get_scheduler_factory(validated, data_provider),
                )

            self._registered_factories[stage_id] = get_stage

    def resolve(self, stage_id: str) -> TrainingStage[TBatch, TModelOutput, TLossInput]:
        return self._registered_factories[stage_id]()

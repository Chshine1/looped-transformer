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
        self._registered_factories: dict[
            str, Callable[[], TrainingStage[TBatch, TModelOutput, TLossInput]]
        ] = {}

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
        def create_factory(
            stage_id: str,
            validated: TTemplate,
            data_provider: IDataProvider[TBatch],
        ) -> Callable[[], TrainingStage[TBatch, TModelOutput, TLossInput]]:
            def get_stage() -> TrainingStage[TBatch, TModelOutput, TLossInput]:
                return TrainingStage(
                    id=stage_id,
                    stage_options=validated.options,
                    data_provider=data_provider,
                    loss=loss_factory(validated),
                    optimizer_factory=get_optimizer_factory(validated),
                    scheduler_factory=get_scheduler_factory(validated, data_provider),
                )

            return get_stage

        for template_stage_id, raw_dict in self._options.stage_templates.items():
            if template.type != raw_dict.get("type"):
                continue
            template_values = {
                key: value for key, value in raw_dict.items() if key != "type"
            }
            validated_template = template.model_validate(template_values)
            template_data_provider = data_provider_factory(validated_template)

            self._registered_factories[template_stage_id] = create_factory(
                template_stage_id, validated_template, template_data_provider
            )

    def resolve(self, stage_id: str) -> TrainingStage[TBatch, TModelOutput, TLossInput]:
        return self._registered_factories[stage_id]()

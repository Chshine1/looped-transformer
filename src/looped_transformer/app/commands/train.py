from argparse import Namespace

from looped_transformer.app.command import Command, CommandContext
from looped_transformer.app.commands.utils.modeling import create_model
from looped_transformer.core.abstractions.typed_module import TypedModule
from looped_transformer.core.config.runtime import DeviceOptions
from looped_transformer.core.training.model_trainer import ModelTrainer
from looped_transformer.core.training.model_trainer_factory import ModelTrainerFactory
from looped_transformer.core.training.training_stage_registry import (
    TrainingStageRegistry,
)
from looped_transformer.domains.clutrr.collator import CLUTRRCollator
from looped_transformer.domains.clutrr.types import CLUTRR_RELATIONS, CLUTRRBatch
from looped_transformer.domains.mnli.training.backbone_collator import (
    BackboneTrainingCollator,
)
from looped_transformer.domains.mnli.training.stage import LoopedMNLIStageTemplate
from looped_transformer.domains.mnli.types.batch import MNLIBatch
from looped_transformer.domains.text_classification.loss import BackboneLossInput
from looped_transformer.domains.text_classification.model import LoopedTextClassifier
from looped_transformer.domains.text_classification.pretrained_embedding_initializer import (
    initialize_pretrained_input_embeddings,
)
from looped_transformer.domains.text_classification.tokenizer import PairTokenizer
from looped_transformer.domains.text_classification.types import ModelForwardOutput


class TrainCommand(Command):
    name = "train"
    help = "Train the configured model"

    def execute(self, args: Namespace, context: CommandContext) -> None:
        del args
        options = context.options
        container = context.container
        stage_type = self._stage_type(context)
        device_options = container[DeviceOptions]
        options.runtime.reproducibility.activate(device_options.device)

        tokenizer = container[PairTokenizer]
        backbone = container.magic_partial(create_model)()
        if options.training.resume_from is None:
            self._initialize_embeddings(context, backbone, tokenizer)

        trainer = self._create_trainer(
            context, stage_type, backbone, tokenizer, device_options
        )
        trainer.train()

    @staticmethod
    def _stage_type(context: CommandContext) -> str:
        stage_type = context.options.training.stage.get("type")
        if stage_type not in {"looped_mnli", "looped_clutrr"}:
            raise ValueError(f"Unknown training stage type: {stage_type!r}")
        return stage_type

    @staticmethod
    def _initialize_embeddings(
        context: CommandContext,
        backbone: TypedModule,
        tokenizer: PairTokenizer,
    ) -> None:
        options = context.options
        initialization = options.model.embedding.initialization
        if (
            options.tokenizer.source.pretrained_model != initialization.pretrained_model
            or options.tokenizer.source.revision != initialization.revision
        ):
            raise ValueError(
                "tokenizer and pretrained input embeddings must use the same "
                "model and revision"
            )
        if not isinstance(backbone.torch, LoopedTextClassifier):
            raise TypeError("Expected a LoopedTextClassifier backbone")
        initialize_pretrained_input_embeddings(
            backbone.torch,
            tokenizer.get_tokenizer_info(),
            initialization,
            options.model.embedding.segment_types_count,
            options.runtime.reproducibility.seed,
        )

    def _create_trainer(
        self,
        context: CommandContext,
        stage_type: str,
        backbone: TypedModule,
        tokenizer: PairTokenizer,
        device_options: DeviceOptions,
    ) -> ModelTrainer:
        if stage_type == "looped_clutrr":
            return self._create_clutrr_trainer(context, backbone, tokenizer)
        return self._create_mnli_trainer(context, backbone, tokenizer, device_options)

    @staticmethod
    def _create_clutrr_trainer(
        context: CommandContext,
        backbone: TypedModule,
        tokenizer: PairTokenizer,
    ) -> ModelTrainer:
        options = context.options
        if options.model.output_classes_count != len(CLUTRR_RELATIONS):
            raise ValueError(f"CLUTRR requires {len(CLUTRR_RELATIONS)} output classes")
        registry = context.container[
            TrainingStageRegistry[CLUTRRBatch, ModelForwardOutput, BackboneLossInput]
        ]
        return context.container[ModelTrainerFactory].create(
            backbone, CLUTRRCollator(tokenizer), registry.resolve()
        )

    @staticmethod
    def _create_mnli_trainer(
        context: CommandContext,
        backbone: TypedModule,
        tokenizer: PairTokenizer,
        device_options: DeviceOptions,
    ) -> ModelTrainer:
        options = context.options
        if options.model.output_classes_count != 3:
            raise ValueError("MNLI requires exactly 3 output classes")
        template_values = {
            key: value for key, value in options.training.stage.items() if key != "type"
        }
        template = LoopedMNLIStageTemplate.model_validate(template_values)
        collator = BackboneTrainingCollator(
            tokenizer,
            template.teacher,
            template.loss_options.distillation_scale,
            device_options.device,
        )
        registry = context.container[
            TrainingStageRegistry[MNLIBatch, ModelForwardOutput, BackboneLossInput]
        ]
        return context.container[ModelTrainerFactory].create(
            backbone, collator, registry.resolve()
        )

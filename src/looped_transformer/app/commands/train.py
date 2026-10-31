from argparse import Namespace

from looped_transformer.app.command import Command, CommandContext
from looped_transformer.app.commands.utils.modeling import create_model
from looped_transformer.core.abstractions.typed_module import TypedBranchedModule
from looped_transformer.core.training.model_trainer import ModelTrainer
from looped_transformer.core.training.model_trainer_factory import ModelTrainerFactory
from looped_transformer.core.training.training_stage_registry import (
    TrainingStageRegistry,
)
from looped_transformer.domains.clutrr.collator import (
    CLUTRRCollator,
    CLUTRRCollatorOptions,
)
from looped_transformer.domains.clutrr.types import CLUTRR_RELATIONS, CLUTRRBatch
from looped_transformer.domains.proofwriter.data_source import PROOFWRITER_TASK
from looped_transformer.domains.proofwriter.types import PROOFWRITER_LABELS
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
        self._validate_stage_type(context)
        options.runtime.reproducibility.activate(options.runtime.device.device)

        tokenizer = container[PairTokenizer]
        backbone = container.magic_partial(create_model)()
        if options.training.resume_from is None:
            self._initialize_embeddings(context, backbone, tokenizer)

        trainer = self._create_trainer(context, backbone, tokenizer)
        trainer.train()

    @staticmethod
    def _validate_stage_type(context: CommandContext) -> None:
        stage_type = context.options.training.stage.get("type")
        if stage_type != "looped_clutrr":
            raise ValueError(f"Unknown training stage type: {stage_type!r}")

    @staticmethod
    def _initialize_embeddings(
        context: CommandContext,
        backbone: TypedBranchedModule,
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
        backbone: TypedBranchedModule,
        tokenizer: PairTokenizer,
    ) -> ModelTrainer:
        options = context.options
        if options.model.output_classes_count != len(CLUTRR_RELATIONS):
            raise ValueError(f"CLUTRR requires {len(CLUTRR_RELATIONS)} output classes")
        raw_collator_options = options.training.stage.get("collator_options")
        collator_options = CLUTRRCollatorOptions.model_validate(raw_collator_options)
        if collator_options.proofwriter is not None:
            configured_classes = options.model.auxiliary_output_classes_counts.get(
                PROOFWRITER_TASK
            )
            if configured_classes != len(PROOFWRITER_LABELS):
                raise ValueError(
                    "ProofWriter auxiliary training requires a three-class "
                    f"{PROOFWRITER_TASK!r} classification head"
                )
        registry = context.container[
            TrainingStageRegistry[CLUTRRBatch, ModelForwardOutput, BackboneLossInput]
        ]
        return context.container[ModelTrainerFactory].create(
            backbone,
            CLUTRRCollator(tokenizer, collator_options),
            registry.resolve(),
        )

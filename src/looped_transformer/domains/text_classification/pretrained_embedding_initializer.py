import os
import uuid
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from transformers import BertModel

from looped_transformer.core.artifacts import get_artifact_id
from looped_transformer.domains.text_classification.model import (
    LoopedTextClassifier,
    ModelOptions,
)
from looped_transformer.domains.text_classification.tokenizer_types import TokenizerInfo

_ALGORITHM_VERSION = "bert-input-pca-v1"


def project_input_embeddings(
    word_embeddings: Tensor,
    segment_embeddings: Tensor,
    target_dimension: int,
    oversampling: int,
    power_iterations: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    if word_embeddings.ndim != 2 or segment_embeddings.ndim != 2:
        raise ValueError("source embeddings must be matrices")
    if word_embeddings.shape[1] != segment_embeddings.shape[1]:
        raise ValueError("word and segment embedding dimensions must match")
    source_dimension = word_embeddings.shape[1]
    if target_dimension >= source_dimension:
        raise ValueError("target embedding dimension must be below source dimension")

    rank = min(source_dimension, target_dimension + oversampling)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        _, _, principal_components = torch.pca_lowrank(
            word_embeddings.float(),
            q=rank,
            center=True,
            niter=power_iterations,
        )
    projection = principal_components[:, :target_dimension]
    return (
        word_embeddings.float() @ projection,
        segment_embeddings.float() @ projection,
    )


def initialize_pretrained_input_embeddings(
    model: LoopedTextClassifier,
    tokenizer_info: TokenizerInfo,
    options: ModelOptions._PretrainedEmbeddingOptions,
    segment_types_count: int,
    seed: int,
) -> Path:
    metadata: dict[str, Any] = {
        "algorithm": _ALGORITHM_VERSION,
        "pretrained_model": options.pretrained_model,
        "revision": options.revision,
        "source_dimension": options.source_dimension,
        "target_dimension": model.embedding_dimension,
        "vocabulary_size": tokenizer_info.vocab_size,
        "segment_types_count": segment_types_count,
        "pca_oversampling": options.pca_oversampling,
        "pca_power_iterations": options.pca_power_iterations,
        "seed": seed,
    }
    cache_path = options.cache_dir / f"{get_artifact_id(metadata)}.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        if payload.get("metadata") != metadata:
            raise ValueError(f"Embedding cache metadata mismatch: {cache_path}")
        token_embeddings = payload["token_embeddings"]
        segment_embeddings = payload["segment_embeddings"]
    else:
        token_embeddings, segment_embeddings = _load_and_project(
            options,
            tokenizer_info,
            model.embedding_dimension,
            segment_types_count,
            seed,
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(f".{cache_path.name}.{uuid.uuid4().hex}.tmp")
        torch.save(
            {
                "metadata": metadata,
                "token_embeddings": token_embeddings,
                "segment_embeddings": segment_embeddings,
            },
            temporary,
        )
        temporary.replace(cache_path)

    model.initialize_input_embeddings(token_embeddings, segment_embeddings)
    print(f"Initialized input embeddings from {cache_path}", flush=True)
    return cache_path


def _load_and_project(
    options: ModelOptions._PretrainedEmbeddingOptions,
    tokenizer_info: TokenizerInfo,
    target_dimension: int,
    segment_types_count: int,
    seed: int,
) -> tuple[Tensor, Tensor]:
    offline = options.local_files_only or os.environ.get(
        "HF_HUB_OFFLINE", ""
    ).lower() in {"1", "true", "yes"}
    print(
        f"Loading pretrained input embeddings from {options.pretrained_model} "
        f"at {options.revision}...",
        flush=True,
    )
    source = BertModel.from_pretrained(
        options.pretrained_model,
        revision=options.revision,
        local_files_only=offline,
    )
    word_embeddings = source.embeddings.word_embeddings.weight.detach().cpu()
    segment_embeddings = source.embeddings.token_type_embeddings.weight.detach().cpu()
    if word_embeddings.shape != (
        tokenizer_info.vocab_size,
        options.source_dimension,
    ):
        raise ValueError(
            "pretrained word embedding shape does not match configured tokenizer "
            f"and source dimension: got {tuple(word_embeddings.shape)}"
        )
    if segment_embeddings.shape != (
        segment_types_count,
        options.source_dimension,
    ):
        raise ValueError(
            "pretrained segment embedding shape does not match configuration: "
            f"got {tuple(segment_embeddings.shape)}"
        )
    return project_input_embeddings(
        word_embeddings,
        segment_embeddings,
        target_dimension,
        options.pca_oversampling,
        options.pca_power_iterations,
        seed,
    )

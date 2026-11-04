from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SplitName = Literal["train", "valid", "test"]
SPLITS: tuple[SplitName, ...] = ("train", "valid", "test")

PositiveInt = Annotated[int, Field(gt=0)]
Proportion = Annotated[float, Field(ge=0.0, le=1.0)]
type DepthProportions = dict[PositiveInt, Proportion]
type DepthTargets = dict[SplitName, dict[int, int]]


class StrictConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SplitConfig(StrictConfig):
    count: Annotated[int, Field(ge=0)]
    depth_proportions: DepthProportions

    @model_validator(mode="after")
    def validate_depth_proportions(self) -> "SplitConfig":
        if abs(sum(self.depth_proportions.values()) - 1.0) > 1e-9:
            raise ValueError("depth_proportions must sum to 1")
        return self


class SplitsConfig(StrictConfig):
    train: SplitConfig
    valid: SplitConfig
    test: SplitConfig

    def get(self, split: SplitName) -> SplitConfig:
        return getattr(self, split)


class GraphGenerationConfig(StrictConfig):
    search_multiplier: PositiveInt
    oversample_multiplier: PositiveInt
    noise: bool
    noise_policy: Literal["supporting", "dangling", "disconnected"]
    max_path_len: PositiveInt


class LanguageGenerationConfig(StrictConfig):
    template_type: Literal["amt", "synthetic"]


class GenerationConfig(StrictConfig):
    data_loc: Annotated[str, Field(min_length=1)]
    splits: SplitsConfig
    graph: GraphGenerationConfig
    language: LanguageGenerationConfig

    def proportions_for(self, split: SplitName) -> DepthProportions:
        return self.splits.get(split).depth_proportions

    def count_for(self, split: SplitName) -> int:
        return self.splits.get(split).count

    def depth_targets(self) -> DepthTargets:
        targets: DepthTargets = {}
        for split in SPLITS:
            total = self.count_for(split)
            proportions = self.proportions_for(split)
            raw = {
                depth: total * proportion for depth, proportion in proportions.items()
            }
            counts = {depth: int(value) for depth, value in raw.items()}
            remainder = total - sum(counts.values())
            order = sorted(
                raw, key=lambda depth: (-(raw[depth] - counts[depth]), depth)
            )
            for depth in order[:remainder]:
                counts[depth] += 1
            targets[split] = counts
        return targets

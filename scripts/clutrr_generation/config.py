from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SplitName = Literal["train", "valid", "test"]
SPLITS: tuple[SplitName, ...] = ("train", "valid", "test")

PositiveInt = Annotated[int, Field(gt=0)]
Lengths = Annotated[list[PositiveInt], Field(min_length=1)]
Proportion = Annotated[float, Field(ge=0.0, le=1.0)]
type DepthProportions = dict[PositiveInt, Proportion]
type DepthTargets = dict[SplitName, dict[int, int]]


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    data_loc: Annotated[str, Field(min_length=1)]
    num_train: Annotated[int, Field(ge=0)]
    num_valid: Annotated[int, Field(ge=0)]
    num_test: Annotated[int, Field(ge=0)]
    train_depth_proportions: DepthProportions
    valid_depth_proportions: DepthProportions
    test_depth_proportions: DepthProportions
    search_multiplier: PositiveInt
    train_descriptor_lengths: Lengths
    val_descriptor_lengths: Lengths
    test_descriptor_lengths: Lengths
    noise: bool
    noise_policy: Literal["supporting", "dangling", "disconnected"]
    max_path_len: PositiveInt
    template_type: Literal["amt", "synthetic"]

    @model_validator(mode="after")
    def validate_depth_proportions(self) -> "GenerationConfig":
        for split in SPLITS:
            proportions = self.proportions_for(split)
            lengths_name = (
                "val_descriptor_lengths"
                if split == "valid"
                else f"{split}_descriptor_lengths"
            )
            lengths = getattr(self, lengths_name)
            if set(proportions) != set(lengths):
                raise ValueError(
                    f"{split}_depth_proportions keys must match {lengths_name}"
                )
            if abs(sum(proportions.values()) - 1.0) > 1e-9:
                raise ValueError(f"{split}_depth_proportions must sum to 1")
        return self

    def proportions_for(self, split: SplitName) -> DepthProportions:
        if split == "train":
            return self.train_depth_proportions
        if split == "valid":
            return self.valid_depth_proportions
        return self.test_depth_proportions

    def count_for(self, split: SplitName) -> int:
        if split == "train":
            return self.num_train
        if split == "valid":
            return self.num_valid
        return self.num_test

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

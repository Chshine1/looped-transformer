import json
import os
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import Annotated, Literal

from clutrr.generator_glc import main as generate_language
from glc.glc import main as generate_graphs
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parent.parent

PositiveInt = Annotated[int, Field(gt=0)]
Lengths = Annotated[list[PositiveInt], Field(min_length=1)]


class GenerationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    data_loc: Annotated[str, Field(min_length=1)]
    num_train: Annotated[int, Field(ge=0)]
    num_valid: Annotated[int, Field(ge=0)]
    num_test: Annotated[int, Field(ge=0)]
    search_multiplier: PositiveInt
    train_descriptor_lengths: Lengths
    val_descriptor_lengths: Lengths
    test_descriptor_lengths: Lengths
    noise: bool
    noise_policy: Literal["supporting", "dangling", "disconnected"]
    max_path_len: PositiveInt
    template_type: Literal["amt", "synthetic"]


def load_config(path: Path) -> GenerationConfig:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as exc:
            raise SystemExit(
                f"PyYAML is required to read {path.name}; install it or use JSON."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return GenerationConfig.model_validate(data)


def resolve_config() -> Path:
    raw = os.environ.get("CLUTRR_CONFIG")
    if raw:
        p = Path(raw)
        if not p.is_absolute():
            p = (ROOT / p).resolve()
        return p

    base = ROOT / "configs" / "clutrr" / "generate"
    for suffix in (".yaml", ".yml", ".json"):
        candidate = base.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    raise SystemExit(
        f"Config file not found: {base.with_suffix('.yaml')} (or .yml/.json)"
    )


@contextmanager
def hydra_runtime():
    """Provide the runtime state required by CLUTRR's get_original_cwd calls."""
    if HydraConfig.initialized():
        yield
        return
    previous = HydraConfig.instance().cfg
    with initialize(version_base="1.1", config_path=None):
        runtime_config = compose(return_hydra_config=True)
        HydraConfig.instance().set_config(runtime_config)
        try:
            yield
        finally:
            HydraConfig.instance().cfg = previous


def main() -> None:
    config_path = resolve_config()
    cfg = load_config(config_path)
    data_loc = Path(cfg.data_loc).resolve()
    data_loc.mkdir(parents=True, exist_ok=True)

    graph_resources = files("glc")
    language_resources = files("clutrr")
    graph_config = OmegaConf.load(str(graph_resources / "graph_config.yaml"))
    language_config = OmegaConf.load(str(language_resources / "config.yaml"))
    # Hydra's callable entry points accept configs directly; CLI defaults are unused.
    for config in (graph_config, language_config):
        del config["defaults"]
        del config["hydra"]

    graph_config.update(
        save_loc=str(data_loc),
        rule_store=str(graph_resources / "rule_bases" / "clutrr"),
        world_prefix="rule",
        num_train_graphs=cfg.num_train * cfg.search_multiplier,
        num_valid_graphs=cfg.num_valid * cfg.search_multiplier,
        num_test_graphs=cfg.num_test * cfg.search_multiplier,
        train_descriptor_lengths=",".join(map(str, cfg.train_descriptor_lengths)),
        val_descriptor_lengths=",".join(map(str, cfg.val_descriptor_lengths)),
        test_descriptor_lengths=",".join(map(str, cfg.test_descriptor_lengths)),
        add_noise=cfg.noise,
        noise_policy=cfg.noise_policy,
        max_path_len=cfg.max_path_len,
    )
    language_config.update(
        data_loc=str(data_loc),
        num_train=cfg.num_train,
        num_valid=cfg.num_valid,
        num_test=cfg.num_test,
        template_type=cfg.template_type,
        template_folder=str(language_resources / "templates"),
        gender_map=str(language_resources / "store" / "relations_store.yaml"),
        names_file=str(language_resources / "names.csv"),
    )
    with hydra_runtime():
        generate_graphs(graph_config)
        generate_language(language_config)
    print(f"done. jsonl at {data_loc / 'rule_0'}")


if __name__ == "__main__":
    main()

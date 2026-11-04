import os
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path

import yaml
from clutrr.generator_glc import main as generate_language
from glc.glc import main as generate_graphs
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf

from .config import GenerationConfig
from .graphs import postprocess_graphs

PROJECT_ROOT = Path(os.environ.get("PIXI_PROJECT_ROOT")).resolve()


def resolve_config_path() -> Path:
    raw = os.environ.get("CLUTRR_CONFIG")
    if raw:
        path = Path(raw)
        return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()

    base = PROJECT_ROOT / "configs" / "clutrr" / "generate"
    for suffix in (".yaml", ".yml"):
        candidate = base.with_suffix(suffix)
        if candidate.is_file():
            return candidate
    raise SystemExit(f"Config file not found: {base.with_suffix('.yaml')} (or .yml)")


@contextmanager
def hydra_runtime():
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
    config_path = resolve_config_path()
    config = GenerationConfig.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    depth_targets = config.depth_targets()
    data_loc = Path(PROJECT_ROOT / config.data_loc).resolve()
    data_loc.mkdir(parents=True, exist_ok=True)

    graph_resources = files("glc")
    language_resources = files("clutrr")
    graph_config = OmegaConf.load(str(graph_resources / "graph_config.yaml"))
    language_config = OmegaConf.load(str(language_resources / "config.yaml"))
    for external_config in (graph_config, language_config):
        del external_config["defaults"]
        del external_config["hydra"]

    graph_config.update(
        save_loc=str(data_loc),
        rule_store=str(graph_resources / "rule_bases" / "clutrr"),
        world_prefix="rule",
        num_train_graphs=sum(depth_targets["train"].values())
        * config.search_multiplier,
        num_valid_graphs=sum(depth_targets["valid"].values())
        * config.search_multiplier,
        num_test_graphs=sum(depth_targets["test"].values()) * config.search_multiplier,
        train_descriptor_lengths=",".join(map(str, config.train_descriptor_lengths)),
        val_descriptor_lengths=",".join(map(str, config.val_descriptor_lengths)),
        test_descriptor_lengths=",".join(map(str, config.test_descriptor_lengths)),
        add_noise=config.noise,
        noise_policy=config.noise_policy,
        max_path_len=config.max_path_len,
    )
    language_config.update(
        data_loc=str(data_loc),
        num_train=config.num_train,
        num_valid=config.num_valid,
        num_test=config.num_test,
        template_type=config.template_type,
        template_folder=str(language_resources / "templates"),
        gender_map=str(language_resources / "store" / "relations_store.yaml"),
        names_file=str(language_resources / "names.csv"),
    )
    with hydra_runtime():
        generate_graphs(graph_config)
        postprocess_graphs(data_loc / "rule_0", depth_targets)
        generate_language(language_config)
    print(f"done. jsonl at {data_loc / 'rule_0'}")

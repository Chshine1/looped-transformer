import argparse
import logging
import os
import sys
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from importlib.resources import files
from pathlib import Path
from typing import TextIO, cast

import yaml
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from .config import SPLITS, GenerationConfig
from .graphs import merge_depth_graphs, postprocess_graphs

PROJECT_ROOT = Path(
    os.environ.get("PIXI_PROJECT_ROOT") or Path(__file__).resolve().parents[2]
).resolve()
LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the CLUTRR dataset")
    parser.add_argument(
        "--log-level",
        choices=LOG_LEVELS,
        default="INFO",
        help="minimum severity for Python log records (default: INFO)",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help=(
            "write all generator output to this file; relative paths are placed "
            "under data_loc, and '-' writes to the console"
        ),
    )
    return parser.parse_args(argv)


def resolve_log_file(log_file: Path, data_loc: Path) -> Path:
    return log_file if log_file.is_absolute() else data_loc / log_file


@contextmanager
def generator_output(stream: TextIO, level: str):
    """Route logging, prints, Rich output, and progress bars to one stream."""
    root_logger = logging.getLogger()
    previous_handlers = root_logger.handlers[:]
    previous_level = root_logger.level
    handler = logging.StreamHandler(stream)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    )
    root_logger.handlers[:] = [handler]
    root_logger.setLevel(getattr(logging, level))
    try:
        with redirect_stdout(stream), redirect_stderr(stream):
            yield
    finally:
        root_logger.handlers[:] = previous_handlers
        root_logger.setLevel(previous_level)
        handler.close()


def run(config: GenerationConfig, data_loc: Path) -> None:
    # Import after output redirection so the vendored Rich consoles and tqdm
    # progress bars bind to the requested destination.
    from clutrr.generator_glc import main as generate_language
    from glc.glc import main as generate_graphs

    depth_targets = config.depth_targets()
    data_loc.mkdir(parents=True, exist_ok=True)

    logging.getLogger(__name__).info("Generating CLUTRR data in %s", data_loc)
    graph_resources = files("glc")
    language_resources = files("clutrr")
    language_config = cast(
        DictConfig, OmegaConf.load(str(language_resources / "config.yaml"))
    )
    del language_config["defaults"]
    del language_config["hydra"]
    language_config.update(
        data_loc=str(data_loc),
        num_train=config.splits.train.count,
        num_valid=config.splits.valid.count,
        num_test=config.splits.test.count,
        template_type=config.language.template_type,
        template_folder=str(language_resources / "templates"),
        gender_map=str(language_resources / "store" / "relations_store.yaml"),
        names_file=str(language_resources / "names.csv"),
    )
    with hydra_runtime():
        depth_directories: dict[int, Path] = {}
        all_depths = sorted(
            {depth for split in SPLITS for depth in depth_targets[split]}
        )
        for depth in all_depths:
            depth_root = data_loc / ".glc_by_depth" / f"depth_{depth}"
            held_out_depth = max(depth, 3)
            graph_config = cast(
                DictConfig, OmegaConf.load(str(graph_resources / "graph_config.yaml"))
            )
            del graph_config["defaults"]
            del graph_config["hydra"]
            graph_config.update(
                save_loc=str(depth_root),
                rule_store=str(graph_resources / "rule_bases" / "clutrr"),
                world_prefix="rule",
                search_multiplier=config.graph.search_multiplier,
                num_train_graphs=max(
                    1,
                    depth_targets["train"].get(depth, 0)
                    * config.graph.oversample_multiplier,
                ),
                num_valid_graphs=max(
                    1,
                    depth_targets["valid"].get(depth, 0)
                    * config.graph.oversample_multiplier,
                ),
                num_test_graphs=max(
                    1,
                    depth_targets["test"].get(depth, 0)
                    * config.graph.oversample_multiplier,
                ),
                train_descriptor_lengths=str(depth),
                val_descriptor_lengths=str(held_out_depth),
                test_descriptor_lengths=str(held_out_depth),
                add_noise=config.graph.noise,
                noise_policy=config.graph.noise_policy,
                max_path_len=config.graph.max_path_len,
            )
            generate_graphs(graph_config)
            depth_directories[depth] = depth_root / "rule_0"

        graph_dir = data_loc / "rule_0"
        merge_depth_graphs(depth_directories, graph_dir)
        postprocess_graphs(graph_dir, depth_targets)
        generate_language(language_config)
    print(f"done. jsonl at {data_loc / 'rule_0'}")


def run_with_error_logging(config: GenerationConfig, data_loc: Path) -> None:
    try:
        run(config, data_loc)
    except Exception:
        logging.getLogger(__name__).exception("CLUTRR generation failed")
        raise


def main(argv: list[str] | None = None, *, log_stream: TextIO | None = None) -> None:
    args = parse_args(argv)
    config_path = resolve_config_path()
    config = GenerationConfig.model_validate(
        yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    data_loc = (PROJECT_ROOT / config.data_loc).resolve()

    if log_stream is not None:
        with generator_output(log_stream, args.log_level):
            run_with_error_logging(config, data_loc)
        return

    if args.log_file is None or str(args.log_file) == "-":
        with generator_output(sys.stderr, args.log_level):
            run_with_error_logging(config, data_loc)
        return

    log_path = resolve_log_file(args.log_file, data_loc)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        log_path.open("w", encoding="utf-8", buffering=1) as stream,
        generator_output(stream, args.log_level),
    ):
        run_with_error_logging(config, data_loc)

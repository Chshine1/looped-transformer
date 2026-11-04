import tempfile
import unittest
from pathlib import Path

from scripts.clutrr_generation.graphs import GraphRecord, postprocess_graphs


def graph(identifier: int, depth: int, *, noise: bool = False) -> GraphRecord:
    path_edges = [
        (index, index + 1, f"relation-{identifier}-{index}") for index in range(depth)
    ]
    noise_edges = [(100, 101, "noise")] if noise else []
    return GraphRecord(
        edges=path_edges + noise_edges,
        source=0,
        sink=depth,
        query=(0, depth),
        target=f"target-{identifier}",
        descriptor=",".join(f"relation-{identifier}-{index}" for index in range(depth)),
        rules_used=[],
        rules_used_pos=[],
        resolution_path=list(range(depth + 1)),
        noise_edges=noise_edges,
    )


class ClutrrGraphPostprocessingTests(unittest.TestCase):
    def test_logical_key_ignores_noise(self) -> None:
        self.assertEqual(
            graph(1, 2).logical_key(), graph(1, 2, noise=True).logical_key()
        )

    def test_exact_depth_counts_and_held_out_deduplication(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            graph_dir = Path(temporary_directory)
            shared = graph(1, 2)
            inputs = {
                "train": [shared, shared, graph(2, 2), graph(3, 3)],
                "valid": [graph(4, 2), graph(5, 3)],
                "test": [shared, graph(6, 2), graph(7, 3)],
            }
            for split, records in inputs.items():
                with (graph_dir / f"{split}.jsonl").open(
                    "w", encoding="utf-8"
                ) as stream:
                    for record in records:
                        stream.write(record.model_dump_json() + "\n")

            postprocess_graphs(
                graph_dir,
                {
                    "train": {2: 1, 3: 1},
                    "valid": {2: 1, 3: 1},
                    "test": {2: 2, 3: 1},
                },
            )

            outputs = {}
            for split in inputs:
                outputs[split] = [
                    GraphRecord.model_validate_json(line)
                    for line in (graph_dir / f"{split}.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
            self.assertEqual([row.depth for row in outputs["train"]], [2, 3])
            self.assertEqual([row.depth for row in outputs["valid"]], [2, 3])
            self.assertEqual([row.depth for row in outputs["test"]], [2, 2, 3])
            self.assertNotIn(
                shared.logical_key(), {row.logical_key() for row in outputs["train"]}
            )

    def test_shortage_is_reported_before_writing(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            graph_dir = Path(temporary_directory)
            for split in ("train", "valid", "test"):
                (graph_dir / f"{split}.jsonl").write_text(
                    graph(1, 2).model_dump_json() + "\n", encoding="utf-8"
                )
            with self.assertRaisesRegex(ValueError, "Increase search_multiplier"):
                postprocess_graphs(
                    graph_dir,
                    {"train": {2: 2}, "valid": {2: 0}, "test": {2: 1}},
                )


if __name__ == "__main__":
    unittest.main()

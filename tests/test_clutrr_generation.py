import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

PROJECT_ROOT = Path(
    os.environ.get("PIXI_PROJECT_ROOT") or Path(__file__).resolve().parents[1]
).resolve()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from clutrr_generation.graphs import DepthTargets, select_rendered_graphs


class SelectRenderedGraphsTest(unittest.TestCase):
    def test_selects_exact_per_depth_quotas_after_rendering(self) -> None:
        targets: DepthTargets = {
            "train": {2: 2, 3: 1},
            "valid": {3: 1},
            "test": {4: 1},
        }
        rows = {
            "train": ["a,b", "c,d", "e,f", "a,b,c", "d,e,f"],
            "valid": ["a,b,c", "d,e,f"],
            "test": ["a,b,c,d", "e,f,g,h"],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            graph_dir = Path(temporary_directory)
            for split, descriptors in rows.items():
                with (graph_dir / f"{split}.jsonl").open(
                    "w", encoding="utf-8"
                ) as stream:
                    for index, descriptor in enumerate(descriptors):
                        stream.write(
                            json.dumps({"descriptor": descriptor, "rendered_id": index})
                            + "\n"
                        )

            output = io.StringIO()
            with redirect_stdout(output):
                select_rendered_graphs(graph_dir, targets)

            for split, depth_targets in targets.items():
                records = [
                    json.loads(line)
                    for line in (graph_dir / f"{split}.jsonl")
                    .read_text(encoding="utf-8")
                    .splitlines()
                ]
                depths = [len(record["descriptor"].split(",")) for record in records]
                self.assertEqual(len(records), sum(depth_targets.values()))
                for depth, expected in depth_targets.items():
                    self.assertEqual(depths.count(depth), expected)
            diagnostics = output.getvalue()
            self.assertIn(
                "language rendering train: expected 3, rendered 5 "
                "(depth 2: expected 2, rendered 3, depth 3: expected 1, rendered 2)",
                diagnostics,
            )
            self.assertIn("final train: expected 3, wrote 3 samples", diagnostics)

    def test_fails_when_rendering_leaves_a_depth_short(self) -> None:
        targets: DepthTargets = {"train": {2: 2}, "valid": {}, "test": {}}
        with tempfile.TemporaryDirectory() as temporary_directory:
            graph_dir = Path(temporary_directory)
            (graph_dir / "train.jsonl").write_text(
                json.dumps({"descriptor": "a,b"}) + "\n", encoding="utf-8"
            )
            (graph_dir / "valid.jsonl").write_text("", encoding="utf-8")
            (graph_dir / "test.jsonl").write_text("", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError,
                "Not enough renderable train graphs at depth 2: expected 2, found 1",
            ):
                select_rendered_graphs(graph_dir, targets)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from common import CLASSES, PROJECT, RUNS, macro_f1
from data import read_split, validate_split
from protocol import effect_summary, load_records, select_run
from analysis import analyze, error_examples
import numpy as np


class ProtocolTests(unittest.TestCase):
    def records(self, scores):
        return [{"regime": regime, "seed": seed, "history": [{"validation_macro_f1": score}]}
                for (regime, seed), score in zip(RUNS, scores)]

    def test_selection_uses_regime_mean_before_best_seed(self):

        records = self.records([0.5, 0.99, 0.7, 0.68, 0.90, 0.92, 0.91])
        chosen = select_run(records)
        self.assertEqual((chosen["regime"], chosen["seed"]), ("unfreeze", 2))
        effects = effect_summary(records)
        self.assertAlmostEqual(effects["delta_pre"], 0.29)
        self.assertAlmostEqual(effects["delta_unf"], 0.12)
        self.assertAlmostEqual(effects["R"], 0.31)
        self.assertFalse(effects["pre_exceeds_noise"])
        self.assertFalse(effects["unf_exceeds_noise"])

    def test_boundary_equal_to_noise_is_not_an_effect(self):
        effects = effect_summary(self.records([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]))
        self.assertFalse(effects["pre_exceeds_noise"])
        self.assertFalse(effects["unf_exceeds_noise"])
        chosen = select_run(self.records([0.5] * 7))
        self.assertEqual((chosen["regime"], chosen["seed"]), ("scratch", 1))

    def test_error_order_frequency_class_indices_and_string_filenames(self):
        y = np.array([2, 2, 0, 0, 1, 1])
        pred = np.array([3, 3, 1, 1, 0, 0])
        logits = np.zeros((6, 6), dtype=np.float32)
        logits[np.arange(6), pred] = 4
        examples = error_examples(["20.jpg", "100.jpg", "2.jpg", "10.jpg", "9.jpg", "8.jpg"], y, logits)
        self.assertEqual([x["file"] for x in examples], ["10.jpg", "8.jpg", "100.jpg", "2.jpg", "9.jpg"])
        self.assertEqual([x["true_class"] for x in examples[:3]], [CLASSES[i] for i in [0, 1, 2]])
        self.assertTrue(all(x["pair_count"] == 2 for x in examples))

    def test_macro_f1_includes_all_six_classes(self):
        logits = np.zeros((2, 6), dtype=np.float32)
        logits[:, 0] = 1
        self.assertAlmostEqual(macro_f1(np.array([0, 0]), logits), 1 / 6)

    def copy_manifest(self, temporary: str):
        target = Path(temporary)
        shutil.copy(PROJECT / "split.csv", target / "split.csv")
        for regime, seed in RUNS:
            name = f"{regime}_seed{seed}"
            source = PROJECT / "runs" / name
            directory = target / "runs" / name
            directory.mkdir(parents=True)
            for filename in ("run.json", "val_logits.npy", "val_files.npy"):
                shutil.copy(source / filename, directory / filename)
        return target

    @unittest.skipUnless((PROJECT / "selection.json").exists(), "Requires completed seven runs")
    def test_split_hash_change_stops_before_test_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = self.copy_manifest(temporary)

            lines = (experiment / "split.csv").read_text().splitlines()
            lines[1], lines[2] = lines[2], lines[1]
            (experiment / "split.csv").write_text("\n".join(lines) + "\n")
            validate_split(experiment / "split.csv")
            with patch("analysis.SplitDataset", side_effect=AssertionError("Test must not be opened")) as dataset:
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    analyze(PROJECT / "data", experiment, experiment / "output")
                dataset.assert_not_called()

    @unittest.skipUnless((PROJECT / "selection.json").exists(), "Requires completed seven runs")
    def test_wrong_selection_stops_before_test_loading(self):
        with tempfile.TemporaryDirectory() as temporary:
            experiment = self.copy_manifest(temporary)
            selection = json.loads((PROJECT / "selection.json").read_text())
            selection["seed"] = 99
            (experiment / "selection.json").write_text(json.dumps(selection))
            with patch("analysis.SplitDataset", side_effect=AssertionError("Test must not be opened")) as dataset:
                with self.assertRaisesRegex(ValueError, "selection.json"):
                    analyze(PROJECT / "data", experiment, experiment / "output")
                dataset.assert_not_called()

    @unittest.skipUnless((PROJECT / "selection.json").exists(), "Requires completed seven runs")
    def test_saved_validation_and_split_reproduce_exactly(self):
        records = load_records(PROJECT)
        self.assertEqual(len(records), 7)
        self.assertEqual(len(validate_split(PROJECT / "split.csv")), 18)
        selection = json.loads((PROJECT / "selection.json").read_text())
        self.assertEqual(select_run(records), {k: selection[k] for k in select_run(records)})


if __name__ == "__main__":
    unittest.main()

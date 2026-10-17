"""Перевірки аналізу без навчання та незмінності готового README."""

from __future__ import annotations

from contextlib import chdir, redirect_stdout
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from analysis import analyze, main

PROJECT = Path(__file__).resolve().parent


class AnalysisTests(unittest.TestCase):
    def analyze_quietly(self, saved: Path, out: Path) -> dict:
        with redirect_stdout(StringIO()):
            analyze(saved, out)
        self.assertEqual(list(out.rglob("*.md")), [])
        return json.loads((out / "metrics.json").read_text(encoding="utf-8"))

    def fixture(self, directory: Path, validation_y: list, validation_scores: list,
                test_y: list, test_scores: list) -> None:
        np.savez_compressed(
            directory / "predictions.npz",
            train_y=np.array([0, 1]),
            validation_y=np.array(validation_y),
            validation_scores=np.array(validation_scores, dtype=np.float32),
            validation_row_ids=np.arange(len(validation_y)),
            test_y=np.array(test_y),
            test_scores=np.array(test_scores, dtype=np.float32),
            test_row_ids=np.arange(100, 100 + len(test_y)),
            test_amount=np.arange(1, 1 + len(test_y), dtype=np.float64),
        )

    def test_cli_reproduces_results_and_preserves_readme_and_saved_files(self) -> None:
        saved = PROJECT / "results"
        before = {p: sha256(p.read_bytes()).hexdigest() for p in saved.rglob("*") if p.is_file()}
        report = (PROJECT / "README.md").read_bytes()
        report_mtime = (PROJECT / "README.md").stat().st_mtime_ns
        # Пробіли й /tmp перевіряють запуск з іншої теки на macOS.
        with tempfile.TemporaryDirectory(prefix="sgn analysis with spaces ", dir="/tmp") as temporary:
            directory = Path(temporary)
            (directory / "README.md").write_bytes(report)
            copied_report_mtime = (directory / "README.md").stat().st_mtime_ns
            with chdir(directory), patch.object(sys, "argv", ["analysis.py", "--saved", str(saved)]):
                with redirect_stdout(StringIO()):
                    main()
            self.assertEqual((directory / "README.md").read_bytes(), report)
            self.assertEqual((directory / "README.md").stat().st_mtime_ns, copied_report_mtime)
            out = directory / "reproduced"
            self.assertEqual(list(out.rglob("*.md")), [])
            for name in ("metrics.json", "threshold.json"):
                actual = json.loads((out / name).read_text(encoding="utf-8"))
                expected = json.loads((saved / name).read_text(encoding="utf-8"))
                self.assertEqual(actual, expected)
            for original in saved.rglob("*.csv"):
                relative = original.relative_to(saved)
                self.assertEqual((out / relative).read_bytes(), original.read_bytes(), str(relative))
            for name in ("validation_pr.png", "validation_roc.png"):
                self.assertTrue((out / name).is_file(), name)
        self.assertEqual((PROJECT / "README.md").read_bytes(), report)
        self.assertEqual((PROJECT / "README.md").stat().st_mtime_ns, report_mtime)
        self.assertEqual(before, {p: sha256(p.read_bytes()).hexdigest() for p in saved.rglob("*") if p.is_file()})

    def test_infinite_threshold_and_undefined_precision(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory, [1] + [0] * 20, [0.1] * 21, [0, 1, 0], [0.9, 0.05, 0.01])
            out = directory / "analysis"
            metrics = self.analyze_quietly(directory, out)
            self.assertEqual(metrics["threshold"], "inf")
            self.assertIsNone(metrics["test"]["precision"])
            self.assertEqual(metrics["test"]["recall"], 0)
            validation = pd.read_csv(out / "validation_summary.csv")
            self.assertEqual(validation.loc[validation["rule"] == "standard", "C"].item(), 10)
            self.assertEqual(metrics["validation"]["C"], 10)
            self.assertEqual(metrics["test_baseline_0_5"]["C"], 11)
            self.assertEqual(metrics["test"]["C"], 10)
            errors = pd.read_csv(out / "errors.csv")
            self.assertTrue(errors["distance_to_threshold"].isna().all())
            self.assertTrue(errors["score_minus_threshold"].isna().all())

    def test_precision_increases_recall_decreases_and_test_cost_rises(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            self.fixture(directory, [1, 0], [0.9, 0.6], [1, 1, 0], [0.95, 0.7, 0.6])
            metrics = self.analyze_quietly(directory, directory / "analysis")
            baseline, test = metrics["test_baseline_0_5"], metrics["test"]
            self.assertGreater(test["precision"], baseline["precision"])
            self.assertLess(test["recall"], baseline["recall"])
            self.assertEqual(test["C"] - baseline["C"], 9)


if __name__ == "__main__":
    unittest.main()

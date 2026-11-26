from __future__ import annotations

import unittest
from unittest.mock import Mock, patch

import numpy as np

from metrics import measure_latency, quality


class QualityTests(unittest.TestCase):
    def test_identical_predictions_include_shared_errors(self):
        labels = np.arange(6)
        prediction = np.array([0, 0, 2, 1, 4, 5])
        result = quality(labels, prediction, prediction)
        self.assertEqual(result["correct"], 4)
        self.assertEqual(result["agreement_with_fp32"], 1)
        self.assertEqual(result["changed_predictions"], 0)
        self.assertEqual(result["b"], 0)
        self.assertEqual(result["c"], 0)
        self.assertEqual(result["paired_standard_error"], 0)
        self.assertEqual(result["delta_accuracy"], 0)
        self.assertFalse(result["accuracy_change_exceeds_noise"])
        self.assertEqual(result["accuracy_change_direction"], "unchanged")

    def test_compensated_changes_and_changed_shared_error(self):
        labels = np.zeros(6, dtype=np.int64)
        reference = np.array([1, 1, 0, 0, 2, 0])
        prediction = np.array([0, 0, 1, 1, 3, 0])
        result = quality(labels, prediction, reference)
        self.assertEqual((result["b"], result["c"]), (2, 2))
        self.assertEqual(result["changed_predictions"], 5)
        self.assertEqual(result["changed_both_wrong"], 1)
        self.assertAlmostEqual(result["agreement_with_fp32"], 1 / 6)
        self.assertEqual(result["accuracy"], 0.5)
        self.assertEqual(result["delta_accuracy"], 0)
        self.assertAlmostEqual(result["paired_standard_error"], 1 / 3)
        self.assertAlmostEqual(result["single_model_standard_error"], np.sqrt(0.25 / 6))
        self.assertFalse(result["accuracy_change_exceeds_noise"])

    def test_noise_threshold_is_inclusive(self):
        labels = np.zeros(4, dtype=np.int64)
        result = quality(labels, labels, np.ones_like(labels))
        self.assertEqual((result["b"], result["c"]), (4, 0))
        self.assertEqual(result["delta_accuracy"], 1)
        self.assertEqual(result["paired_standard_error"], 0.5)
        self.assertFalse(result["accuracy_change_exceeds_noise"])

    def test_significant_gain_and_loss_have_opposite_directions(self):
        labels = np.zeros(5, dtype=np.int64)
        wrong = np.ones_like(labels)
        gain = quality(labels, labels, wrong)
        loss = quality(labels, wrong, labels)
        self.assertTrue(gain["accuracy_change_exceeds_noise"])
        self.assertTrue(loss["accuracy_change_exceeds_noise"])
        self.assertEqual(gain["accuracy_change_direction"], "increase")
        self.assertEqual(loss["accuracy_change_direction"], "decrease")
        self.assertEqual(gain["delta_accuracy"], 1)
        self.assertEqual(loss["delta_accuracy"], -1)
        self.assertEqual(gain["paired_standard_error"], loss["paired_standard_error"])

    def test_macro_f1_averages_all_six_classes(self):
        result = quality(np.array([0, 0]), np.array([0, 0]), np.array([0, 0]))
        self.assertEqual(result["macro_f1"], 1 / 6)
        matrix = np.asarray(result["confusion_matrix"])
        self.assertEqual(matrix.shape, (6, 6))
        self.assertEqual(matrix[0, 0], 2)
        self.assertEqual(matrix.sum(), 2)

    def test_invalid_labels_and_vector_shapes_are_rejected(self):
        cases = [
            ([], [], []),
            ([0], [0, 1], [0]),
            ([[0]], [[0]], [[0]]),
            ([6], [0], [0]),
            ([0], [-1], [0]),
            ([0], [0], [np.nan]),
        ]
        for labels, prediction, reference in cases:
            with self.subTest(labels=labels, prediction=prediction, reference=reference):
                with self.assertRaises(ValueError):
                    quality(np.array(labels), np.array(prediction), np.array(reference))


class LatencyTests(unittest.TestCase):
    def run_mock_measurement(self, ticks):
        events = []
        sessions = {name: Mock() for name in ("fp32", "int8")}
        for name, session in sessions.items():
            session.run.side_effect = lambda output, feed, name=name: events.append(name)
        clock_values = []
        current_tick = 0
        for round_index in range(3):
            for name in ("fp32", "int8"):
                for duration in ticks[name][round_index]:
                    current_tick += 2048
                    clock_values.append(current_tick / 1024)
                    current_tick += int(duration)
                    clock_values.append(current_tick / 1024)
        clock_iterator = iter(clock_values)

        def read_clock():
            events.append("clock")
            return next(clock_iterator)

        inputs = np.zeros((1, 3, 160, 160), dtype=np.float32)
        with patch("metrics.time.perf_counter", side_effect=read_clock) as counter:
            result = measure_latency(sessions, inputs)
        self.assertEqual(counter.call_count, 1200)
        self.assertEqual(list(clock_iterator), [])
        return result, events, sessions, inputs

    def test_exact_protocol_and_pooled_percentiles(self):
        ticks = {
            "fp32": [np.arange(1, 101), np.arange(1, 101), np.arange(1000, 1100)],
            "int8": [np.arange(1, 101)] * 3,
        }
        result, events, sessions, inputs = self.run_mock_measurement(ticks)
        expected_events = []
        for _ in range(3):
            for name in ("fp32", "int8"):
                expected_events.extend([name] * 20)
                expected_events.extend(["clock", name, "clock"] * 100)
        self.assertEqual(events, expected_events)
        first_feed = sessions["fp32"].run.call_args_list[0].args[1]
        for name, session in sessions.items():
            self.assertEqual(session.run.call_count, 360)
            for call in session.run.call_args_list:
                self.assertEqual(call.args[0], ["logits"])
                self.assertIs(call.args[1], first_feed)
                self.assertIs(call.args[1]["input"], inputs)
            expected = np.concatenate(ticks[name]) * (1000 / 1024)
            self.assertEqual(result[name]["measurements"], 300)
            np.testing.assert_array_equal(
                [sample for row in result[name]["rounds"] for sample in row["measurements_ms"]],
                expected,
            )
            self.assertEqual(result[name]["p50_ms"], np.percentile(expected, 50))
            self.assertEqual(result[name]["p95_ms"], np.percentile(expected, 95))
            self.assertEqual([row["round"] for row in result[name]["rounds"]], [1, 2, 3])
        median_round_medians = np.median([row["p50_ms"] for row in result["fp32"]["rounds"]])
        self.assertNotEqual(result["fp32"]["p50_ms"], median_round_medians)
        self.assertEqual(result["p50_speedup"], result["fp32"]["p50_ms"] / result["int8"]["p50_ms"])
        self.assertEqual(result["p95_speedup"], result["fp32"]["p95_ms"] / result["int8"]["p95_ms"])

    def test_round_intervals_require_strict_separation(self):
        cases = [((10, 12, 14), (1, 3, 5), True), ((3, 4, 5), (1, 2, 3), False)]
        for fp32, int8, expected in cases:
            ticks = {
                "fp32": [np.full(100, duration) for duration in fp32],
                "int8": [np.full(100, duration) for duration in int8],
            }
            with self.subTest(fp32=fp32, int8=int8):
                result, _, _, _ = self.run_mock_measurement(ticks)
                self.assertEqual(result["intervals_disjoint"], expected)

    def test_input_shape_and_type_are_checked_before_calls(self):
        sessions = {name: Mock() for name in ("fp32", "int8")}
        for inputs in (
            np.zeros((2, 3, 160, 160), dtype=np.float32),
            np.zeros((1, 3, 160, 160), dtype=np.float64),
        ):
            with self.subTest(shape=inputs.shape, dtype=inputs.dtype):
                with self.assertRaises(ValueError):
                    measure_latency(sessions, inputs)
        for session in sessions.values():
            session.run.assert_not_called()


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

from pathlib import Path
import time

import numpy as np
import onnxruntime as ort
from sklearn.metrics import confusion_matrix, f1_score
import torch


def create_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), sess_options=options, providers=["CPUExecutionProvider"]
    )
    if session.get_providers() != ["CPUExecutionProvider"]:
        raise AssertionError("Inference must use CPUExecutionProvider only")
    if session.get_inputs()[0].name != "input" or session.get_outputs()[0].name != "logits":
        raise AssertionError("Unexpected inference interface")
    return session


def quality(y: np.ndarray, prediction: np.ndarray, reference: np.ndarray) -> dict:
    y = np.asarray(y)
    prediction = np.asarray(prediction)
    reference = np.asarray(reference)
    if y.ndim != 1 or y.size == 0 or y.shape != prediction.shape or y.shape != reference.shape:
        raise ValueError("Labels and predictions must be nonempty vectors of equal length")
    if any(not np.isin(values, np.arange(6)).all() for values in (y, prediction, reference)):
        raise ValueError("Class indices must be between 0 and 5")
    correct = prediction == y
    reference_correct = reference == y
    b = int(np.count_nonzero(correct & ~reference_correct))
    c = int(np.count_nonzero(~correct & reference_correct))
    n = int(y.size)
    accuracy = float(correct.mean())
    paired_se = float(np.sqrt(b + c) / n)
    single_se = float(np.sqrt(accuracy * (1 - accuracy) / n))
    significant = bool(abs(b - c) > 2 * np.sqrt(b + c))
    return {
        "n": n,
        "correct": int(correct.sum()),
        "macro_f1": float(f1_score(y, prediction, labels=range(6), average="macro", zero_division=0)),
        "accuracy": accuracy,
        "agreement_with_fp32": float(np.mean(prediction == reference)),
        "changed_predictions": int(np.count_nonzero(prediction != reference)),
        "changed_both_wrong": int(np.count_nonzero((prediction != reference) & ~correct & ~reference_correct)),
        "b": b,
        "c": c,
        "delta_accuracy": (b - c) / n,
        "paired_standard_error": paired_se,
        "single_model_standard_error": single_se,
        "accuracy_change_exceeds_noise": significant,
        "accuracy_change_direction": "increase" if b > c else "decrease" if b < c else "unchanged",
        "confusion_matrix": confusion_matrix(y, prediction, labels=range(6)).tolist(),
    }


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader, sessions: dict[str, ort.InferenceSession]) -> tuple[dict, dict]:
    model.eval()
    torch_batches = []
    labels = []
    onnx_batches = {name: [] for name in sessions}
    for batch, target in loader:
        inputs = np.ascontiguousarray(batch.numpy(), dtype=np.float32)
        labels.append(target.numpy())
        torch_batches.append(model(batch).cpu().numpy())
        for name, session in sessions.items():
            onnx_batches[name].append(session.run(["logits"], {"input": inputs})[0])
    y = np.concatenate(labels)
    torch_logits = np.concatenate(torch_batches)
    logits = {name: np.concatenate(batches) for name, batches in onnx_batches.items()}
    for values in [torch_logits, *logits.values()]:
        if values.shape != (1502, 6) or not np.isfinite(values).all():
            raise AssertionError("Expected finite logits for all 1502 test images")
    reference = logits["fp32"].argmax(axis=1)
    torch_prediction = torch_logits.argmax(axis=1)
    maximum_difference = float(np.max(np.abs(torch_logits - logits["fp32"])))
    torch_quality = quality(y, torch_prediction, reference)
    rows = {name: quality(y, values.argmax(axis=1), reference) for name, values in logits.items()}
    if maximum_difference >= 1e-3 or not np.array_equal(torch_prediction, reference):
        raise AssertionError("PyTorch and ONNX FP32 export equivalence check failed")
    if torch_quality["macro_f1"] != rows["fp32"]["macro_f1"]:
        raise AssertionError("PyTorch and ONNX FP32 macro-F1 differ")
    comparison = {
        "maximum_absolute_logit_difference": maximum_difference,
        "prediction_agreement": torch_quality["agreement_with_fp32"],
        "pytorch_macro_f1": torch_quality["macro_f1"],
        "onnx_fp32_macro_f1": rows["fp32"]["macro_f1"],
        "test_images": int(y.size),
        "batch_size": 64,
        "torch_no_grad": True,
    }
    baseline_se = rows["fp32"]["single_model_standard_error"]
    for row in rows.values():
        row["delta_macro_f1"] = row["macro_f1"] - rows["fp32"]["macro_f1"]
        paired_se = row["paired_standard_error"]
        row["fp32_single_se_over_paired_se"] = baseline_se / paired_se if paired_se else None
    return comparison, rows


def measure_latency(sessions: dict[str, ort.InferenceSession], inputs: np.ndarray) -> dict:
    if inputs.shape != (1, 3, 160, 160) or inputs.dtype != np.float32:
        raise ValueError("Latency input must be float32 with shape (1,3,160,160)")
    feed = {"input": np.ascontiguousarray(inputs)}
    output_names = ["logits"]
    rounds = {name: [] for name in ("fp32", "int8")}
    for round_index in range(3):
        for name in ("fp32", "int8"):
            session = sessions[name]
            for _ in range(20):
                session.run(output_names, feed)
            measurements = []
            for _ in range(100):
                started = time.perf_counter()
                session.run(output_names, feed)
                elapsed = time.perf_counter() - started
                measurements.append(elapsed * 1000)
            rounds[name].append({
                "round": round_index + 1,
                "measurements_ms": measurements,
                "p50_ms": float(np.percentile(measurements, 50)),
                "p95_ms": float(np.percentile(measurements, 95)),
            })
    result = {}
    for name, values in rounds.items():
        samples = [sample for row in values for sample in row["measurements_ms"]]
        medians = [row["p50_ms"] for row in values]
        result[name] = {
            "measurements": len(samples),
            "p50_ms": float(np.percentile(samples, 50)),
            "p95_ms": float(np.percentile(samples, 95)),
            "round_p50_interval_ms": [min(medians), max(medians)],
            "rounds": values,
        }
    fp32_interval = result["fp32"]["round_p50_interval_ms"]
    int8_interval = result["int8"]["round_p50_interval_ms"]
    result["intervals_disjoint"] = bool(
        fp32_interval[1] < int8_interval[0] or int8_interval[1] < fp32_interval[0]
    )
    result["p50_speedup"] = result["fp32"]["p50_ms"] / result["int8"]["p50_ms"]
    result["p95_speedup"] = result["fp32"]["p95_ms"] / result["int8"]["p95_ms"]
    result["protocol"] = {
        "batch_size": 1,
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "rounds": 3,
        "variant_order": ["fp32", "int8"],
        "warmups_per_variant_per_round": 20,
        "measurements_per_variant_per_round": 100,
        "timed_operation": "one session.run between two time.perf_counter calls",
        "percentile_method": "numpy.percentile default linear",
        "input_prepared_before_timing": True,
        "sessions_created_before_timing": True,
    }
    return result

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from pathlib import Path
import time

import numpy as np
import onnx
from onnx import numpy_helper
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
import torch


class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, inputs: Iterable[np.ndarray]):
        self.inputs: Iterator[np.ndarray] = iter(inputs)
        self.count = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        image = next(self.inputs, None)
        if image is None:
            return None
        if image.dtype != np.float32 or image.shape != (1, 3, 160, 160):
            raise ValueError("Calibration inputs must be float32 with shape (1, 3, 160, 160)")
        if not np.isfinite(image).all():
            raise ValueError("Calibration input contains non-finite values")
        self.count += 1
        return {"input": np.ascontiguousarray(image)}


def export_model(model: torch.nn.Module, path: Path, img_size: int = 160) -> dict:
    path = Path(path)
    model.eval()
    example = torch.zeros((1, 3, img_size, img_size), dtype=torch.float32)
    started = time.perf_counter()
    with torch.no_grad():
        torch.onnx.export(
            model,
            example,
            path,
            dynamo=False,
            opset_version=17,
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        )
    seconds = time.perf_counter() - started
    graph = onnx.load(path)
    onnx.checker.check_model(graph)
    counts = Counter(node.op_type for node in graph.graph.node)
    if counts["Conv"] != 20 or counts["BatchNormalization"] != 0:
        raise AssertionError("Expected 20 Conv nodes and no BatchNormalization after export")
    if graph.graph.input[0].name != "input" or graph.graph.output[0].name != "logits":
        raise AssertionError("Unexpected ONNX input or output names")
    if not graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param:
        raise AssertionError("ONNX input batch axis is not dynamic")
    if not graph.graph.output[0].type.tensor_type.shape.dim[0].dim_param:
        raise AssertionError("ONNX output batch axis is not dynamic")
    return {
        "seconds": seconds,
        "opset_version": 17,
        "dynamo": False,
        "input_name": "input",
        "output_name": "logits",
        "dynamic_batch": True,
        "conv_nodes": counts["Conv"],
        "batch_normalization_nodes": counts["BatchNormalization"],
        "node_counts": dict(sorted(counts.items())),
    }


def channel_ranges(path: Path) -> dict:
    graph = onnx.load(path).graph
    initializers = {value.name: value for value in graph.initializer}
    convolutions = []
    for node in graph.node:
        if node.op_type != "Conv":
            continue
        if len(node.input) < 2 or node.input[1] not in initializers:
            raise AssertionError(f"Missing constant convolution weights: {node.name}")
        weights = numpy_helper.to_array(initializers[node.input[1]])
        if weights.ndim != 4 or not np.isfinite(weights).all():
            raise ValueError(f"Invalid convolution weights: {node.name}")
        alpha = np.max(np.abs(weights).astype(np.float64), axis=(1, 2, 3))
        median_alpha = float(np.median(alpha))
        if median_alpha <= 0:
            raise ValueError(f"Zero median channel range: {node.name}")
        maximum_alpha = float(alpha.max())
        ratio = maximum_alpha / median_alpha
        convolutions.append({
            "name": node.name,
            "weight_name": node.input[1],
            "weight_shape": list(weights.shape),
            "alpha_min": float(alpha.min()),
            "alpha_median": median_alpha,
            "alpha_max": maximum_alpha,
            "r": ratio,
            "median_channel_bits_lost": float(np.log2(ratio)),
        })
    if len(convolutions) != 20:
        raise AssertionError("Expected 20 convolution weight tensors")
    median_r = float(np.median([item["r"] for item in convolutions]))
    return {
        "convolutions": convolutions,
        "top_three": sorted(convolutions, key=lambda item: item["r"], reverse=True)[:3],
        "median_r": median_r,
        "median_bits_lost": float(np.log2(median_r)),
    }


def preprocess_model(fp32: Path, prep: Path) -> dict:
    started = time.perf_counter()
    quant_pre_process(fp32, prep, skip_symbolic_shape=True)
    return {
        "seconds": time.perf_counter() - started,
        "skip_symbolic_shape": True,
    }


def quantize_model(
    prep: Path,
    output: Path,
    calibration_inputs: Iterable[np.ndarray],
    per_channel: bool = True,
) -> dict:
    reader = ImageCalibrationReader(calibration_inputs)
    started = time.perf_counter()
    quantize_static(
        prep,
        output,
        reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        per_channel=per_channel,
        calibrate_method=CalibrationMethod.MinMax,
    )
    seconds = time.perf_counter() - started
    if reader.count != 200:
        raise AssertionError(f"Expected 200 calibration images, got {reader.count}")
    onnx.checker.check_model(onnx.load(output))
    return {
        "seconds": seconds,
        "quant_format": "QDQ",
        "activation_type": "QUInt8",
        "weight_type": "QInt8",
        "per_channel": per_channel,
        "calibrate_method": "MinMax",
        "calibration_batch_size": 1,
        "calibration_images": reader.count,
        "other_arguments": "defaults",
    }

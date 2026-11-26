from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import subprocess

import numpy as np
import onnx
import onnxruntime as ort
import PIL
import sklearn
import torch
import torchvision

from data import calibration_batches, load_model, make_test_loader, prepare_data, sha256
from download_data import ensure_data
from metrics import create_session, evaluate, measure_latency
from quantization import channel_ranges, export_model, preprocess_model, quantize_model
from reporting import write_forecast, write_report


PROJECT = Path(__file__).resolve().parent


def command_output(arguments: list[str]) -> str | None:
    try:
        result = subprocess.run(arguments, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def machine_environment() -> dict:
    cpu = command_output(["sysctl", "-n", "machdep.cpu.brand_string"]) if platform.system() == "Darwin" else None
    if cpu is None and platform.system() == "Linux":
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.is_file():
            cpu = next((line.split(":", 1)[1].strip() for line in cpuinfo.read_text().splitlines()
                        if line.startswith("model name")), None)
    return {
        "cpu": cpu or platform.processor() or platform.machine(),
        "operating_system": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": ort.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "pillow": PIL.__version__,
        "providers": ["CPUExecutionProvider"],
        "intra_op_num_threads": 1,
        "inter_op_num_threads": 1,
        "graph_optimization_level": str(ort.SessionOptions().graph_optimization_level),
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cpu_features": {
            name: command_output(["sysctl", "-n", f"hw.optional.arm.{name}"])
            for name in ("FEAT_DotProd", "FEAT_I8MM")
        } if platform.system() == "Darwin" else {},
    }


def benchmark_context() -> dict:
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "power": command_output(["pmset", "-g", "batt"]) if platform.system() == "Darwin" else None,
        "process_cpu_snapshot": command_output(["ps", "-axo", "pcpu,comm", "-r"]),
    }


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ЛР4: експорт, статична INT8-квантизація та CPU-вимірювання")
    parser.add_argument("--model", type=Path, default=PROJECT / "lab3_model.pt")
    parser.add_argument("--split", type=Path, default=PROJECT / "split.csv")
    parser.add_argument("--data", type=Path, default=PROJECT / "data")
    parser.add_argument("--out", type=Path)
    return parser.parse_args(argv)


def choose_output(requested: Path | None) -> Path:
    if requested is not None:
        requested.mkdir(parents=True, exist_ok=True)
        if any(requested.iterdir()):
            raise ValueError(f"Output directory is not empty: {requested}")
        return requested
    index = 1
    while True:
        path = PROJECT / ("reproduced" if index == 1 else f"reproduced_run{index}")
        try:
            path.mkdir()
        except FileExistsError:
            index += 1
        else:
            return path


def main() -> None:
    arguments = parse_arguments()
    arguments.out = choose_output(arguments.out)
    results_path = arguments.out / "lab4_results.json"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    np.random.seed(0)
    environment = machine_environment()
    print("Перевірка моделі, поділу й калібрувальних даних", flush=True)
    model, model_metadata = load_model(arguments.model)
    ensure_data(arguments.data, arguments.split)
    test, calibration, data_metadata = prepare_data(arguments.data, arguments.split, model_metadata["img_size"])
    files = {
        "fp32": arguments.out / "lab4_fp32.onnx",
        "prepared": arguments.out / "lab4_fp32_prepared.onnx",
        "int8": arguments.out / "lab4_int8.onnx",
        "int8_per_tensor": arguments.out / "lab4_int8_per_tensor.onnx",
    }
    print("Експорт FP32 та діапазони каналів", flush=True)
    export = export_model(model, files["fp32"], model_metadata["img_size"])
    ranges = channel_ranges(files["fp32"])
    forecast = write_forecast(arguments.out, ranges)
    print("Прогноз записано в README до запуску потензорного досліду", flush=True)
    preprocessing = preprocess_model(files["fp32"], files["prepared"])
    print("Поканальна INT8-квантизація на 200 train-зображеннях", flush=True)
    quantization = {
        "int8": quantize_model(files["prepared"], files["int8"], calibration_batches(calibration), per_channel=True)
    }
    print("Потензорна INT8-квантизація на тій самій калібрувальній вибірці", flush=True)
    quantization["int8_per_tensor"] = quantize_model(
        files["prepared"], files["int8_per_tensor"], calibration_batches(calibration), per_channel=False
    )
    sessions = {name: create_session(files[name]) for name in ("fp32", "int8", "int8_per_tensor")}
    print("Якість PyTorch і трьох ONNX-варіантів на всіх 1502 test-зображеннях", flush=True)
    equivalence, quality = evaluate(model, make_test_loader(test), sessions)
    previous_metrics_path = PROJECT / "lab4_results.json"
    if previous_metrics_path.is_file():
        previous = json.loads(previous_metrics_path.read_text(encoding="utf-8"))
        if (previous.get("model", {}).get("sha256") == model_metadata["sha256"]
                and previous.get("data", {}).get("split_sha256") == data_metadata["split_sha256"]):
            equivalence["recorded_test_macro_f1"] = previous["quality"]["fp32"]["macro_f1"]
            equivalence["matches_recorded_run"] = bool(np.isclose(
                quality["fp32"]["macro_f1"], previous["quality"]["fp32"]["macro_f1"], rtol=0, atol=1e-12
            ))
            if not equivalence["matches_recorded_run"]:
                raise AssertionError("FP32 macro-F1 does not match the saved result for this model and split")
    first_image, _ = test[0]
    latency_input = np.ascontiguousarray(first_image.unsqueeze(0).numpy(), dtype=np.float32)
    del sessions["int8_per_tensor"]
    before = benchmark_context()
    print("Затримка: 3 раунди FP32 → INT8, по 20 прогрівів і 100 вимірів", flush=True)
    latency = measure_latency(sessions, latency_input)
    after = benchmark_context()
    latency["input_file"] = test.rows[0]["file"]
    latency["context_before"] = before
    latency["context_after"] = after
    artifacts = {
        name: {"filename": path.name, "bytes": path.stat().st_size,
               "megabytes": path.stat().st_size / 1e6, "sha256": sha256(path)}
        for name, path in files.items()
    }
    source_files = sorted([*PROJECT.glob("*.py"), PROJECT / "pyproject.toml", PROJECT / "uv.lock"])
    result = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": environment,
        "model": model_metadata,
        "data": data_metadata,
        "export": export,
        "export_equivalence": equivalence,
        "preprocessing": preprocessing,
        "quantization": quantization,
        "quality": quality,
        "channel_ranges": ranges,
        "forecast": forecast,
        "latency": latency,
        "artifacts": artifacts,
        "size_reduction_fp32_over_int8": artifacts["fp32"]["bytes"] / artifacts["int8"]["bytes"],
        "source_files_sha256": {path.name: sha256(path) for path in source_files},
        "settings_deviations": [],
    }
    write_json(results_path, result)
    write_report(arguments.out / "README.md", result)
    print(f"Готово: {results_path.resolve()}", flush=True)
    print(f"FP32 F1={quality['fp32']['macro_f1']:.6f}; INT8 F1={quality['int8']['macro_f1']:.6f}; "
          f"прискорення p50={latency['p50_speedup']:.3f}×", flush=True)


if __name__ == "__main__":
    main()

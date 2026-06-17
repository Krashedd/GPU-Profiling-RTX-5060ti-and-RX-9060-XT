#!/usr/bin/env python3

import argparse
import contextlib
import ctypes
import importlib.util
import json
import math
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision
from torch.profiler import ProfilerActivity, profile
from torchvision.models import resnet50


PROJECT_DIR = Path.home() / "Thesis-Testing"
SCRIPT_DIR = PROJECT_DIR / "scripts"
REGULAR_SCRIPT = SCRIPT_DIR / "resnet50_regular_amd.py"
PROFILE_SCRIPT = SCRIPT_DIR / "resnet50_profiled_amd.py"
DATA_DIR = PROJECT_DIR / "Dataset" / "CIFAR10"
PROFILE_DIR = PROJECT_DIR / "Logs" / "resnetprofiledrun"
GPU_ERROR_HISTORY = PROJECT_DIR / "Logs" / "resnet_gpu_error_history.log"
ROCPROFV3 = Path("/opt/rocm/bin/rocprofv3")
TRAINING_RANGE_NAME = "RESNET50_TRAINING"

EXPECTED_CONFIG = {
    "SEED": 55,
    "BATCH_SIZE": 256,
    "EPOCHS": 10,
    "LR": 0.01,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 1e-4,
    "NUM_WORKERS": 4,
    "SAMPLE_INTERVAL_S": 0.2,
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def reset_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def load_regular_module():
    if not REGULAR_SCRIPT.is_file():
        raise RuntimeError(f"Required regular script is missing: {REGULAR_SCRIPT}")

    spec = importlib.util.spec_from_file_location(
        "resnet50_regular_amd_shared", REGULAR_SCRIPT
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load regular script: {REGULAR_SCRIPT}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    mismatches = []
    for name, expected in EXPECTED_CONFIG.items():
        actual = getattr(module, name, None)
        if actual != expected:
            mismatches.append(f"{name}: expected {expected!r}, found {actual!r}")

    if mismatches:
        raise RuntimeError(
            "The profiling script and regular script no longer use the same "
            "fixed workload configuration:\n" + "\n".join(mismatches)
        )

    return module


def append_gpu_error_history(label: str, traceback_text: str) -> None:
    GPU_ERROR_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with GPU_ERROR_HISTORY.open("a", encoding="utf-8") as file:
        file.write("\n" + "=" * 88 + "\n")
        file.write(f"Timestamp: {datetime.now().astimezone().isoformat()}\n")
        file.write(f"Workload: ResNet-50 CIFAR-10 AMD {label}\n")
        file.write(traceback_text.rstrip() + "\n")


def is_gpu_related_error(text: str) -> bool:
    terms = (
        "amd-smi",
        "amdsmi",
        "amdgpu",
        "cuda",
        "device-side",
        "gfx1200",
        "gpu",
        "hip",
        "hsa",
        "memory access fault",
        "miopen",
        "out of memory",
        "rocblas",
        "rocm",
        "rocprof",
        "roctx",
    )
    lowered = text.lower()
    return any(term in lowered for term in terms)


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"Expected output is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def stream_subprocess(command, env=None) -> int:
    print("Command:")
    print(" ".join(str(part) for part in command))
    print()

    process = subprocess.Popen(
        [str(part) for part in command],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        env=env,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="")
    process.stdout.close()
    return process.wait()


class RoctxController:
    """Minimal ROCTx range wrapper used only to mark the training window.

    Profiling pause/resume is intentionally not required. ROCm installations
    can expose the annotation APIs while omitting the profiler-control symbols
    from the legacy libroctx64 compatibility library. The parser filters kernel
    rows to the explicit RESNET50_TRAINING push/pop range instead.
    """

    def __init__(self):
        candidates = (
            Path("/opt/rocm/lib/librocprofiler-sdk-roctx.so"),
            Path("/opt/rocm/lib64/librocprofiler-sdk-roctx.so"),
            Path("/opt/rocm/lib/libroctx64.so"),
            Path("/opt/rocm/lib64/libroctx64.so"),
        )

        selected = None
        failures = []
        for library_path in candidates:
            if not library_path.is_file():
                continue
            try:
                library = ctypes.CDLL(str(library_path))
            except OSError as error:
                failures.append(f"{library_path}: {error}")
                continue

            required = ("roctxRangePushA", "roctxRangePop")
            missing = [name for name in required if not hasattr(library, name)]
            if missing:
                failures.append(
                    f"{library_path}: missing {', '.join(missing)}"
                )
                continue

            selected = (library_path, library)
            break

        if selected is None:
            details = "; ".join(failures) if failures else "no candidate files found"
            raise RuntimeError(
                "No usable ROCTx library with roctxRangePushA/roctxRangePop "
                f"was found. Details: {details}"
            )

        self.library_path, self.lib = selected
        self.lib.roctxRangePushA.argtypes = [ctypes.c_char_p]
        self.lib.roctxRangePushA.restype = ctypes.c_int
        self.lib.roctxRangePop.argtypes = []
        self.lib.roctxRangePop.restype = ctypes.c_int
        self.range_active = False

    def push(self, label: str) -> None:
        result = self.lib.roctxRangePushA(label.encode("utf-8"))
        if result < 0:
            raise RuntimeError(f"roctxRangePushA failed with result {result}")
        self.range_active = True

    def pop(self) -> None:
        if self.range_active:
            result = self.lib.roctxRangePop()
            self.range_active = False
            if result < 0:
                raise RuntimeError(f"roctxRangePop failed with result {result}")


def operator_note(name: str) -> str:
    notes = {
        "aten::convolution_backward": "Convolution backward",
        "aten::convolution": "Convolution forward",
        "aten::_convolution": "Convolution dispatch",
        "aten::native_batch_norm_backward": "Batch normalization backward",
        "aten::batch_norm": "Batch normalization forward",
        "aten::copy_": "Tensor copy",
        "aten::add_": "In-place tensor addition",
        "aten::foreach_add_": "Multi-tensor optimizer addition",
        "Optimizer.step#SGD.step": "SGD optimizer update",
        "aten::amp_foreach_non_finite_check_and_unscale_": (
            "AMP gradient check and unscale"
        ),
    }
    return notes.get(name, "")


def event_device_times(event):
    self_us = getattr(event, "self_device_time_total", None)
    total_us = getattr(event, "device_time_total", None)
    if self_us is None:
        self_us = getattr(event, "self_cuda_time_total", 0.0)
    if total_us is None:
        total_us = getattr(event, "cuda_time_total", 0.0)
    return float(self_us or 0.0), float(total_us or 0.0)


def save_operator_outputs(profiler, output_dir: Path) -> pd.DataFrame:
    rows = []
    for event in profiler.key_averages():
        self_us, total_us = event_device_times(event)
        if self_us <= 0 and total_us <= 0:
            continue
        rows.append(
            {
                "operator": event.key,
                "self_gpu_time_ms": self_us / 1000.0,
                "gpu_total_ms": total_us / 1000.0,
                "calls": int(event.count),
                "notes": operator_note(event.key),
            }
        )

    rows.sort(key=lambda row: row["self_gpu_time_ms"], reverse=True)
    total_self_ms = sum(row["self_gpu_time_ms"] for row in rows)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["self_gpu_percent"] = (
            row["self_gpu_time_ms"] / total_self_ms * 100.0
            if total_self_ms > 0
            else float("nan")
        )

    columns = [
        "rank",
        "operator",
        "self_gpu_time_ms",
        "self_gpu_percent",
        "gpu_total_ms",
        "calls",
        "notes",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(output_dir / "pytorch_all_operators.csv", index=False)
    frame.head(10).to_csv(output_dir / "pytorch_top10_operators.csv", index=False)

    table_text = None
    for sort_key in ("self_device_time_total", "self_cuda_time_total"):
        try:
            table_text = profiler.key_averages().table(
                sort_by=sort_key, row_limit=100
            )
            break
        except Exception:
            continue
    if table_text is None:
        table_text = frame.head(100).to_string(index=False)

    (output_dir / "pytorch_profiler_full_table.txt").write_text(
        table_text, encoding="utf-8"
    )
    return frame


def run_worker(profile_kind: str, output_dir: Path, smoke: bool) -> None:
    regular = load_regular_module()
    reset_directory(output_dir)

    log_file = (output_dir / "training_output.log").open(
        "w", encoding="utf-8", buffering=1
    )
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    monitor = None
    roctx = None
    profiler_object = None

    try:
        regular.set_seed(regular.SEED)

        if not DATA_DIR.exists():
            raise RuntimeError(f"Dataset directory does not exist: {DATA_DIR}")
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot detect the AMD GPU.")
        if torch.version.hip is None:
            raise RuntimeError("This PyTorch build does not report a HIP runtime.")

        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_arch = torch.cuda.get_device_properties(0).gcnArchName
        if "9060 XT" not in gpu_name:
            raise RuntimeError(
                f"Unexpected GPU selected: {gpu_name}. Expected RX 9060 XT."
            )

        epochs_to_run = 1 if smoke else regular.EPOCHS
        max_train_batches = 5 if smoke else None
        max_validation_batches = 2 if smoke else None

        print("=" * 80)
        print(
            f"ResNet-50 CIFAR-10 AMD {profile_kind} profiling "
            f"{'smoke test' if smoke else 'official pass'}"
        )
        print("=" * 80)
        print(f"Timestamp: {datetime.now().astimezone().isoformat()}")
        print(f"Output directory: {output_dir}")
        print(f"Dataset directory: {DATA_DIR}")
        print(f"Python: {platform.python_version()}")
        print(f"Platform: {platform.platform()}")
        print(f"PyTorch: {torch.__version__}")
        print(f"Torchvision: {torchvision.__version__}")
        print(f"HIP runtime: {torch.version.hip}")
        print(f"System ROCm: {regular.read_rocm_version()}")
        print(f"GPU: {gpu_name}")
        print(f"GPU architecture: {gpu_arch}")
        print()

        # Mark the exact training interval. rocprofv3 may trace the whole
        # process, but post-processing keeps only kernels fully contained in
        # this ROCTx push/pop range.
        if profile_kind == "rocprofv3":
            roctx = RoctxController()
            print(f"ROCTx library: {roctx.library_path}")
            print(
                "ROCTx training range enabled; parsed kernel rankings will "
                "exclude setup and validation activity."
            )
            print()

        trainset, testset, trainloader, testloader = regular.build_loaders(smoke)
        model = resnet50(num_classes=10).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=regular.LR,
            momentum=regular.MOMENTUM,
            weight_decay=regular.WEIGHT_DECAY,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        if not scaler.is_enabled():
            raise RuntimeError("AMP GradScaler is not enabled.")

        config = {
            "mode": "smoke" if smoke else "official_profiled",
            "profile_kind": profile_kind,
            "seed": regular.SEED,
            "model": "torchvision.models.resnet50",
            "dataset": "CIFAR-10",
            "training_images": len(trainset),
            "test_images": len(testset),
            "batch_size": regular.BATCH_SIZE,
            "epochs": epochs_to_run,
            "official_epochs": regular.EPOCHS,
            "optimizer": "SGD",
            "learning_rate": regular.LR,
            "momentum": regular.MOMENTUM,
            "weight_decay": regular.WEIGHT_DECAY,
            "loss_function": "CrossEntropyLoss",
            "precision": "AMP FP16",
            "transform": "ToTensor only",
            "num_workers": regular.NUM_WORKERS,
            "pin_memory": True,
            "persistent_workers": regular.NUM_WORKERS > 0,
            "telemetry_interval_s": regular.SAMPLE_INTERVAL_S,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "hip_runtime": torch.version.hip,
            "system_rocm": regular.read_rocm_version(),
            "gpu": gpu_name,
            "gpu_architecture": gpu_arch,
            "max_train_batches": max_train_batches,
            "max_validation_batches": max_validation_batches,
            "training_range_name": TRAINING_RANGE_NAME,
        }
        with (output_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

        epoch_rows = []
        total_samples_processed = 0
        total_batches_processed = 0
        first_batch_output_dtype = "not_recorded"

        monitor = regular.TelemetryMonitor(
            interval_s=regular.SAMPLE_INTERVAL_S
        )
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        profiler_context = contextlib.nullcontext()
        if profile_kind == "torch_operator":
            profiler_context = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )

        print("Starting monitored training...")
        print(
            "Precision: AMP FP16 | Optimizer: SGD | "
            f"Batch size: {regular.BATCH_SIZE} | Epochs: {epochs_to_run}"
        )

        with profiler_context as active_profiler:
            if profile_kind == "torch_operator":
                profiler_object = active_profiler

            if roctx is not None:
                roctx.push(TRAINING_RANGE_NAME)
                print("ROCTx training range started.")

            monitor.start()
            total_start = time.perf_counter()

            for epoch in range(epochs_to_run):
                model.train()
                epoch_start = time.perf_counter()
                running_loss = 0.0
                epoch_samples = 0
                epoch_batches = 0

                for batch_index, (images, labels) in enumerate(trainloader):
                    if (
                        max_train_batches is not None
                        and batch_index >= max_train_batches
                    ):
                        break

                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)

                    with torch.amp.autocast(
                        device_type="cuda", dtype=torch.float16, enabled=True
                    ):
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    if first_batch_output_dtype == "not_recorded":
                        first_batch_output_dtype = str(outputs.dtype)
                        print(
                            "Precision verification | "
                            f"GradScaler enabled: {scaler.is_enabled()} | "
                            "First-batch output dtype: "
                            f"{first_batch_output_dtype}"
                        )

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    current_batch_size = images.size(0)
                    epoch_samples += current_batch_size
                    total_samples_processed += current_batch_size
                    total_batches_processed += 1
                    epoch_batches += 1
                    running_loss += loss.item() * current_batch_size

                torch.cuda.synchronize(device)
                epoch_time = time.perf_counter() - epoch_start
                epoch_throughput = epoch_samples / epoch_time
                average_loss = running_loss / epoch_samples

                epoch_rows.append(
                    {
                        "epoch": epoch + 1,
                        "loss": average_loss,
                        "epoch_time_s": epoch_time,
                        "throughput_samples_s": epoch_throughput,
                        "samples": epoch_samples,
                        "batches": epoch_batches,
                        "grad_scaler_scale_end": float(scaler.get_scale()),
                    }
                )

                print(
                    f"Epoch {epoch + 1}/{epochs_to_run} | "
                    f"Loss: {average_loss:.4f} | "
                    f"Time: {epoch_time:.2f}s | "
                    f"Throughput: {epoch_throughput:.2f} samples/s | "
                    f"Batches: {epoch_batches}"
                )

            torch.cuda.synchronize(device)
            total_training_time = time.perf_counter() - total_start
            monitor.stop()

            if roctx is not None:
                roctx.pop()
                print("ROCTx training range ended.")

        print(f"Total Training Time: {total_training_time:.2f}s")
        print("Training complete.")
        print()

        epoch_df = pd.DataFrame(epoch_rows)
        epoch_df.to_csv(output_dir / "epoch_metrics.csv", index=False)

        telemetry_df = monitor.dataframe()
        telemetry_df.to_csv(output_dir / "telemetry.csv", index=False)
        with (output_dir / "amdsmi_api_errors.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(
                monitor.api_error_counts, file, indent=2, sort_keys=True
            )
        regular.validate_telemetry(telemetry_df)

        if profile_kind == "torch_operator":
            if profiler_object is None:
                raise RuntimeError("PyTorch profiler did not initialize.")
            print("Aggregating PyTorch operator statistics...")
            operator_df = save_operator_outputs(profiler_object, output_dir)
            if operator_df.empty:
                raise RuntimeError("No GPU operator statistics were produced.")
            profiler_object.export_chrome_trace(
                str(output_dir / "pytorch_trace.json")
            )
            print(f"Top PyTorch operator: {operator_df.iloc[0]['operator']}")
            print()

        expected_batches_per_epoch = math.ceil(len(trainset) / regular.BATCH_SIZE)
        if not smoke:
            expected_samples = len(trainset) * regular.EPOCHS
            expected_batches = expected_batches_per_epoch * regular.EPOCHS
            if len(epoch_rows) != regular.EPOCHS:
                raise RuntimeError(
                    f"Incomplete pass: expected {regular.EPOCHS} epochs, "
                    f"got {len(epoch_rows)}."
                )
            if total_samples_processed != expected_samples:
                raise RuntimeError(
                    f"Incomplete pass: expected {expected_samples} training "
                    f"samples, got {total_samples_processed}."
                )
            if total_batches_processed != expected_batches:
                raise RuntimeError(
                    f"Incomplete pass: expected {expected_batches} batches, "
                    f"got {total_batches_processed}."
                )

        print("Starting final validation...")
        model.eval()
        correct = 0
        validation_total = 0
        validation_batches = 0

        with torch.no_grad():
            for batch_index, (images, labels) in enumerate(testloader):
                if (
                    max_validation_batches is not None
                    and batch_index >= max_validation_batches
                ):
                    break
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(images)
                predicted = outputs.argmax(dim=1)
                validation_total += labels.size(0)
                validation_batches += 1
                correct += (predicted == labels).sum().item()

        if not smoke and validation_total != len(testset):
            raise RuntimeError(
                f"Incomplete validation: expected {len(testset)} samples, "
                f"got {validation_total}."
            )

        validation_accuracy = correct / validation_total
        print(
            f"Final Validation Score: {validation_accuracy:.4f} "
            f"({validation_total} samples, {validation_batches} batches)"
        )

        throughput = total_samples_processed / total_training_time
        batch_latency_ms = (
            total_training_time / total_batches_processed
        ) * 1000.0
        average_epoch_time = statistics.mean(
            row["epoch_time_s"] for row in epoch_rows
        )

        average_power_w = regular.series_stat(telemetry_df, "power_w", "mean")
        average_gpu_util = regular.series_stat(
            telemetry_df, "gpu_util_percent", "mean"
        )
        average_vram_mb = regular.series_stat(
            telemetry_df, "vram_used_mb", "mean"
        )
        peak_vram_mb = regular.series_stat(
            telemetry_df, "vram_used_mb", "max"
        )
        average_edge_temp_c = regular.series_stat(
            telemetry_df, "gpu_edge_temperature_c", "mean"
        )
        average_hotspot_temp_c = regular.series_stat(
            telemetry_df, "gpu_hotspot_temperature_c", "mean"
        )
        average_memory_temp_c = regular.series_stat(
            telemetry_df, "gpu_memory_temperature_c", "mean"
        )
        average_cpu_util = regular.series_stat(
            telemetry_df, "cpu_util_percent", "mean"
        )
        average_cpu_temp_c = regular.series_stat(
            telemetry_df, "cpu_package_temperature_c", "mean"
        )
        average_system_ram_mb = regular.series_stat(
            telemetry_df, "system_ram_used_mb", "mean"
        )
        peak_system_ram_mb = regular.series_stat(
            telemetry_df, "system_ram_used_mb", "max"
        )
        average_disk_read_mb_s = regular.series_stat(
            telemetry_df, "disk_read_mb_s", "mean"
        )
        average_disk_write_mb_s = regular.series_stat(
            telemetry_df, "disk_write_mb_s", "mean"
        )
        peak_disk_read_mb_s = regular.series_stat(
            telemetry_df, "disk_read_mb_s", "max"
        )
        peak_disk_write_mb_s = regular.series_stat(
            telemetry_df, "disk_write_mb_s", "max"
        )

        performance_per_watt = (
            throughput / average_power_w
            if not np.isnan(average_power_w) and average_power_w > 0
            else float("nan")
        )

        stability_notes = (
            "Completed successfully; no Python, HIP, ROCm, AMD SMI, or "
            "profiler fatal errors detected."
        )
        if monitor.thermal_limit_seen:
            stability_notes += (
                " At least one monitored GPU temperature reached its AMD SMI "
                "critical limit."
            )
        else:
            stability_notes += (
                " No monitored GPU temperature reached its AMD SMI critical limit."
            )
        if monitor.api_error_count:
            stability_notes += (
                f" AMD SMI had {monitor.api_error_count} recoverable API errors."
            )
        else:
            stability_notes += " All AMD SMI telemetry queries succeeded."

        summary = {
            "mode": "smoke" if smoke else "official_profiled",
            "profile_kind": profile_kind,
            "throughput_samples_s": throughput,
            "batch_latency_ms_batch": batch_latency_ms,
            "total_training_time_s": total_training_time,
            "average_epoch_time_s": average_epoch_time,
            "average_gpu_power_w": average_power_w,
            "performance_per_watt_samples_s_w": performance_per_watt,
            "average_gpu_util_percent": average_gpu_util,
            "average_vram_usage_mb": average_vram_mb,
            "peak_vram_usage_mb": peak_vram_mb,
            "average_gpu_edge_temperature_c": average_edge_temp_c,
            "average_gpu_hotspot_temperature_c": average_hotspot_temp_c,
            "average_gpu_memory_temperature_c": average_memory_temp_c,
            "final_validation_score": validation_accuracy,
            "validation_samples": validation_total,
            "peak_system_ram_mb": peak_system_ram_mb,
            "average_system_ram_mb": average_system_ram_mb,
            "average_cpu_util_percent": average_cpu_util,
            "average_cpu_package_temperature_c": average_cpu_temp_c,
            "average_disk_read_mb_s": average_disk_read_mb_s,
            "average_disk_write_mb_s": average_disk_write_mb_s,
            "peak_disk_read_mb_s": peak_disk_read_mb_s,
            "peak_disk_write_mb_s": peak_disk_write_mb_s,
            "total_samples_processed": total_samples_processed,
            "total_batches_processed": total_batches_processed,
            "telemetry_samples": len(telemetry_df),
            "telemetry_interval_s": regular.SAMPLE_INTERVAL_S,
            "amdsmi_api_error_count": monitor.api_error_count,
            "amp_grad_scaler_enabled": scaler.is_enabled(),
            "amp_first_batch_output_dtype": first_batch_output_dtype,
            "stability_error_notes": stability_notes,
        }
        pd.DataFrame([summary]).to_csv(
            output_dir / "summary.csv", index=False
        )

        with (output_dir / "summary.txt").open("w", encoding="utf-8") as file:
            file.write(
                f"ResNet-50 AMD {profile_kind} "
                f"{'Smoke Test' if smoke else 'Profiled Pass'}\n"
            )
            file.write("=" * 64 + "\n")
            file.write(f"Throughput: {throughput:.2f} samples/s\n")
            file.write(f"Batch Latency: {batch_latency_ms:.2f} ms/batch\n")
            file.write(f"Total Training Time: {total_training_time:.2f} s\n")
            file.write(f"Average Epoch Time: {average_epoch_time:.2f} s\n")
            file.write(f"Average GPU Power Draw: {average_power_w:.2f} W\n")
            file.write(
                "Performance per Watt: "
                f"{performance_per_watt:.2f} samples/s/W\n"
            )
            file.write(
                f"Average GPU Utilization: {average_gpu_util:.2f} %\n"
            )
            file.write(f"Average VRAM Usage: {average_vram_mb:.2f} MB\n")
            file.write(f"Peak VRAM Usage: {peak_vram_mb:.2f} MB\n")
            file.write(
                "Average GPU Edge Temperature: "
                f"{average_edge_temp_c:.2f} C\n"
            )
            file.write(
                "Average GPU Hotspot Temperature: "
                f"{average_hotspot_temp_c:.2f} C\n"
            )
            file.write(
                "Average GPU Memory Temperature: "
                f"{average_memory_temp_c:.2f} C\n"
            )
            file.write(f"Final Validation Score: {validation_accuracy:.4f}\n")
            file.write(f"Peak System RAM: {peak_system_ram_mb:.2f} MB\n")
            file.write(
                f"Average System RAM: {average_system_ram_mb:.2f} MB\n"
            )
            file.write(f"Average CPU Utilization: {average_cpu_util:.2f} %\n")
            file.write(
                "Average CPU Package Temperature: "
                f"{average_cpu_temp_c:.2f} C\n"
            )
            file.write(
                "Average Disk Read / Write: "
                f"{average_disk_read_mb_s:.2f} / "
                f"{average_disk_write_mb_s:.2f} MB/s\n"
            )
            file.write(
                "Peak Disk Read / Write: "
                f"{peak_disk_read_mb_s:.2f} / "
                f"{peak_disk_write_mb_s:.2f} MB/s\n"
            )
            file.write(
                "Precision: GradScaler enabled; first-batch output dtype "
                f"{first_batch_output_dtype}\n"
            )
            file.write(f"Stability / Error Notes: {stability_notes}\n")

        valid = {
            "status": "VALID",
            "timestamp": datetime.now().astimezone().isoformat(),
            "mode": "smoke" if smoke else "official_profiled",
            "profile_kind": profile_kind,
        }
        with (output_dir / "WORKER_VALID.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(valid, file, indent=2)

        print()
        print("=" * 80)
        print("PROFILE WORKER VALID")
        print("=" * 80)
        print(f"Profile kind: {profile_kind}")
        print(f"Throughput: {throughput:.2f} samples/s")
        print(f"Batch Latency: {batch_latency_ms:.2f} ms/batch")
        print(f"Total Training Time: {total_training_time:.2f} s")
        print(f"Final Validation Score: {validation_accuracy:.4f}")
        print(f"Results saved to: {output_dir}")

    except Exception:
        if roctx is not None:
            try:
                roctx.pop()
            except Exception:
                pass
        if monitor is not None:
            try:
                monitor.stop()
            except Exception:
                pass
            try:
                partial = monitor.dataframe()
                if not partial.empty:
                    partial.to_csv(
                        output_dir / "telemetry_partial.csv", index=False
                    )
            except Exception:
                pass
        print("\nPROFILE WORKER FAILED")
        traceback.print_exc()
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def find_trace_files(root: Path, suffix: str):
    return sorted(path for path in root.rglob(f"*{suffix}") if path.is_file())


def read_trace_csvs(paths) -> pd.DataFrame:
    frames = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        frame["source_file"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def parse_rocprof_outputs(raw_dir: Path, destination: Path) -> dict:
    kernel_paths = find_trace_files(raw_dir, "kernel_trace.csv")
    marker_paths = find_trace_files(raw_dir, "marker_api_trace.csv")

    if not kernel_paths:
        raise RuntimeError("rocprofv3 produced no kernel_trace.csv file.")
    if not marker_paths:
        raise RuntimeError("rocprofv3 produced no marker_api_trace.csv file.")

    kernel_df = read_trace_csvs(kernel_paths)
    marker_df = read_trace_csvs(marker_paths)
    if kernel_df.empty:
        raise RuntimeError("rocprofv3 kernel trace is empty.")
    if marker_df.empty:
        raise RuntimeError("rocprofv3 marker trace is empty.")

    required_kernel_columns = {
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
    }
    missing_kernel = required_kernel_columns.difference(kernel_df.columns)
    if missing_kernel:
        raise RuntimeError(
            "Kernel trace is missing columns: "
            + ", ".join(sorted(missing_kernel))
        )

    required_marker_columns = {
        "Function",
        "Start_Timestamp",
        "End_Timestamp",
    }
    missing_marker = required_marker_columns.difference(marker_df.columns)
    if missing_marker:
        raise RuntimeError(
            "Marker trace is missing columns: "
            + ", ".join(sorted(missing_marker))
        )

    marker_matches = marker_df[
        marker_df["Function"].astype(str).str.contains(
            TRAINING_RANGE_NAME, regex=False, na=False
        )
    ].copy()
    if marker_matches.empty:
        available = marker_df["Function"].dropna().astype(str).unique()[:20]
        raise RuntimeError(
            f"Training marker {TRAINING_RANGE_NAME!r} was not found. "
            f"Available marker names include: {available.tolist()}"
        )

    marker_matches["duration_ns"] = (
        pd.to_numeric(marker_matches["End_Timestamp"], errors="coerce")
        - pd.to_numeric(marker_matches["Start_Timestamp"], errors="coerce")
    )
    marker_row = marker_matches.sort_values(
        "duration_ns", ascending=False
    ).iloc[0]
    training_start = int(marker_row["Start_Timestamp"])
    training_end = int(marker_row["End_Timestamp"])
    if training_end <= training_start:
        raise RuntimeError("The ROCTx training marker has an invalid duration.")

    kernel_df["Start_Timestamp"] = pd.to_numeric(
        kernel_df["Start_Timestamp"], errors="coerce"
    )
    kernel_df["End_Timestamp"] = pd.to_numeric(
        kernel_df["End_Timestamp"], errors="coerce"
    )
    kernel_df = kernel_df.dropna(
        subset=["Kernel_Name", "Start_Timestamp", "End_Timestamp"]
    )
    training_kernels = kernel_df[
        (kernel_df["Start_Timestamp"] >= training_start)
        & (kernel_df["End_Timestamp"] <= training_end)
    ].copy()

    if training_kernels.empty:
        raise RuntimeError("No kernels were found inside the ROCTx training range.")

    training_kernels["duration_ns"] = (
        training_kernels["End_Timestamp"]
        - training_kernels["Start_Timestamp"]
    )
    training_kernels = training_kernels[training_kernels["duration_ns"] >= 0]

    grouped = (
        training_kernels.groupby("Kernel_Name", dropna=False)["duration_ns"]
        .agg(["count", "sum", "mean", "min", "max", "std"])
        .reset_index()
        .rename(
            columns={
                "Kernel_Name": "kernel_name",
                "count": "calls",
                "sum": "total_duration_ns",
                "mean": "average_duration_ns",
                "min": "minimum_duration_ns",
                "max": "maximum_duration_ns",
                "std": "stddev_duration_ns",
            }
        )
    )
    grouped = grouped.sort_values("total_duration_ns", ascending=False)
    total_duration_ns = grouped["total_duration_ns"].sum()
    grouped.insert(0, "rank", range(1, len(grouped) + 1))
    grouped["total_duration_ms"] = grouped["total_duration_ns"] / 1e6
    grouped["average_duration_us"] = grouped["average_duration_ns"] / 1e3
    grouped["percentage"] = (
        grouped["total_duration_ns"] / total_duration_ns * 100.0
        if total_duration_ns > 0
        else float("nan")
    )

    columns = [
        "rank",
        "kernel_name",
        "calls",
        "total_duration_ms",
        "average_duration_us",
        "percentage",
        "minimum_duration_ns",
        "maximum_duration_ns",
        "stddev_duration_ns",
        "total_duration_ns",
    ]
    grouped = grouped[columns]
    grouped.to_csv(destination / "rocprofv3_parsed_summary.csv", index=False)
    grouped.head(5).to_csv(destination / "amd_top5_kernels.csv", index=False)
    training_kernels.to_csv(
        destination / "rocprofv3_training_kernel_trace.csv", index=False
    )

    metadata = {
        "kernel_trace_files": [str(path) for path in kernel_paths],
        "marker_trace_files": [str(path) for path in marker_paths],
        "training_marker": TRAINING_RANGE_NAME,
        "training_start_timestamp_ns": training_start,
        "training_end_timestamp_ns": training_end,
        "training_marker_duration_s": (training_end - training_start) / 1e9,
        "raw_kernel_rows": int(len(kernel_df)),
        "training_kernel_rows": int(len(training_kernels)),
        "unique_training_kernels": int(len(grouped)),
        "training_window_filter_applied": True,
    }
    with (destination / "rocprofv3_parse_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    return metadata


def validate_worker(worker_dir: Path, kind: str, smoke: bool) -> None:
    marker = worker_dir / "WORKER_VALID.json"
    if not marker.is_file():
        raise RuntimeError(f"{kind} worker did not produce WORKER_VALID.json")
    data = json.loads(marker.read_text(encoding="utf-8"))
    if data.get("status") != "VALID":
        raise RuntimeError(f"{kind} worker status is not VALID: {data}")

    summary_path = worker_dir / "summary.csv"
    if not summary_path.is_file():
        raise RuntimeError(f"{kind} worker summary.csv is missing.")
    summary = pd.read_csv(summary_path).iloc[0]

    expected_samples = 1280 if smoke else 500000
    expected_batches = 5 if smoke else 1960
    if int(summary["total_samples_processed"]) != expected_samples:
        raise RuntimeError(
            f"{kind} worker processed {summary['total_samples_processed']} "
            f"samples; expected {expected_samples}."
        )
    if int(summary["total_batches_processed"]) != expected_batches:
        raise RuntimeError(
            f"{kind} worker processed {summary['total_batches_processed']} "
            f"batches; expected {expected_batches}."
        )
    if str(summary["amp_first_batch_output_dtype"]) != "torch.float16":
        raise RuntimeError(
            f"{kind} worker did not verify FP16 autocast output."
        )


def run_parent(smoke: bool) -> None:
    output_dir = (
        Path("/tmp/resnet50_amd_profile_smoke") if smoke else PROFILE_DIR
    )
    reset_directory(output_dir)

    log_file = (output_dir / "training_output.log").open(
        "w", encoding="utf-8", buffering=1
    )
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    try:
        load_regular_module()
        if not PROFILE_SCRIPT.is_file():
            raise RuntimeError(
                "Run this script from its installed location: "
                f"{PROFILE_SCRIPT}"
            )
        if not ROCPROFV3.is_file():
            raise RuntimeError(f"rocprofv3 is missing: {ROCPROFV3}")

        torch_dir = output_dir / "pytorch_operator_pass"
        kernel_worker_dir = output_dir / "rocprofv3_worker_pass"
        raw_dir = output_dir / "rocprofv3_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)

        parent_config = {
            "mode": "smoke" if smoke else "official_profiled",
            "workflow": "two_pass_single_command",
            "primary_profile_metrics_source": "pytorch_operator_pass",
            "operator_profiler": "torch.profiler",
            "kernel_profiler": "rocprofv3 runtime trace with ROCTx training-window filtering",
            "reason_for_two_passes": (
                "PyTorch operator profiling and rocprofv3 kernel tracing are "
                "executed separately to avoid profiler interference."
            ),
            "both_passes_use_identical_fixed_workload": True,
            "training_range_name": TRAINING_RANGE_NAME,
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        with (output_dir / "profile_workflow_config.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(parent_config, file, indent=2)

        worker_smoke_arg = ["--internal-smoke"] if smoke else []

        print("=" * 88)
        print(
            "ResNet-50 AMD profiled workflow "
            f"({'smoke test' if smoke else 'official'})"
        )
        print("=" * 88)
        print(f"Output directory: {output_dir}")
        print(
            "This single command performs two controlled passes: "
            "PyTorch operators, then rocprofv3 kernels."
        )
        print()

        print("PHASE 1/2: PYTORCH OPERATOR PROFILING")
        print("-" * 88)
        torch_command = [
            sys.executable,
            str(PROFILE_SCRIPT),
            "--internal-worker",
            "torch_operator",
            "--internal-output",
            str(torch_dir),
            *worker_smoke_arg,
        ]
        return_code = stream_subprocess(torch_command, env=os.environ.copy())
        if return_code != 0:
            raise RuntimeError(
                f"PyTorch operator worker exited with code {return_code}."
            )
        validate_worker(torch_dir, "PyTorch operator", smoke)
        print()

        print("PHASE 2/2: ROCPROFV3 KERNEL/RUNTIME TRACING")
        print("-" * 88)
        kernel_command = [
            str(ROCPROFV3),
            "--runtime-trace",
            "--stats",
            "--summary",
            "--summary-units",
            "msec",
            "--summary-output-file",
            "rocprofv3_summary",
            "--output-format",
            "csv",
            "pftrace",
            "--output-directory",
            str(raw_dir),
            "--output-file",
            "resnet50_profile",
            "--perfetto-buffer-size",
            "2097152",
            "--perfetto-buffer-fill-policy",
            "ring_buffer",
            "--",
            sys.executable,
            str(PROFILE_SCRIPT),
            "--internal-worker",
            "rocprofv3",
            "--internal-output",
            str(kernel_worker_dir),
            *worker_smoke_arg,
        ]
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = (
            "/opt/rocm/lib:/opt/rocm/lib64"
            + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        )
        return_code = stream_subprocess(kernel_command, env=env)
        if return_code != 0:
            raise RuntimeError(f"rocprofv3 exited with code {return_code}.")
        validate_worker(kernel_worker_dir, "rocprofv3", smoke)
        print()

        print("PARSING ROCPROFV3 KERNEL DATA")
        print("-" * 88)
        parse_metadata = parse_rocprof_outputs(raw_dir, output_dir)
        print(
            f"Training kernels parsed: {parse_metadata['training_kernel_rows']} "
            f"dispatches across {parse_metadata['unique_training_kernels']} "
            "unique kernel names."
        )
        print()

        # The PyTorch profiler pass is the official profiled-performance source.
        primary_files = (
            "config.json",
            "epoch_metrics.csv",
            "telemetry.csv",
            "amdsmi_api_errors.json",
            "summary.csv",
            "summary.txt",
            "pytorch_all_operators.csv",
            "pytorch_top10_operators.csv",
            "pytorch_profiler_full_table.txt",
            "pytorch_trace.json",
        )
        for name in primary_files:
            copy_file(torch_dir / name, output_dir / name)

        copy_file(
            torch_dir / "training_output.log",
            output_dir / "pytorch_operator_training_output.log",
        )
        copy_file(
            kernel_worker_dir / "training_output.log",
            output_dir / "rocprofv3_worker_training_output.log",
        )
        copy_file(
            kernel_worker_dir / "summary.csv",
            output_dir / "rocprofv3_worker_summary.csv",
        )
        copy_file(
            kernel_worker_dir / "summary.txt",
            output_dir / "rocprofv3_worker_summary.txt",
        )

        top_operators = pd.read_csv(output_dir / "pytorch_top10_operators.csv")
        top_kernels = pd.read_csv(output_dir / "amd_top5_kernels.csv")
        if top_operators.empty:
            raise RuntimeError("Top PyTorch operator table is empty.")
        if top_kernels.empty:
            raise RuntimeError("Top AMD kernel table is empty.")

        run_valid = {
            "status": "VALID",
            "timestamp": datetime.now().astimezone().isoformat(),
            "mode": "smoke" if smoke else "official_profiled",
            "primary_metrics_source": "pytorch_operator_pass",
            "pytorch_operator_pass_valid": True,
            "rocprofv3_pass_valid": True,
            "training_window_filter_applied": parse_metadata[
                "training_window_filter_applied"
            ],
            "top_operator": str(top_operators.iloc[0]["operator"]),
            "top_kernel": str(top_kernels.iloc[0]["kernel_name"]),
        }
        with (output_dir / "RUN_VALID.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(run_valid, file, indent=2)

        print("=" * 88)
        print("PROFILED WORKFLOW VALID")
        print("=" * 88)
        print("Primary performance metrics: PyTorch operator profiler pass")
        print(f"Top PyTorch operator: {run_valid['top_operator']}")
        print(f"Top AMD kernel: {run_valid['top_kernel']}")
        print(f"Results saved to: {output_dir}")

    except Exception:
        traceback_text = traceback.format_exc()
        print("\nPROFILED WORKFLOW FAILED")
        print(traceback_text)
        if is_gpu_related_error(traceback_text):
            append_gpu_error_history(
                "profiled smoke test" if smoke else "official profiled run",
                traceback_text,
            )
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Automated ResNet-50 AMD profiling workflow using torch.profiler "
            "and rocprofv3"
        )
    )
    public = parser.add_mutually_exclusive_group()
    public.add_argument(
        "--official",
        action="store_true",
        help="Clear resnetprofiledrun and execute the full profiled workflow.",
    )
    public.add_argument(
        "--smoke",
        action="store_true",
        help="Execute a disposable two-pass profiling smoke test in /tmp.",
    )

    parser.add_argument(
        "--internal-worker",
        choices=("torch_operator", "rocprofv3"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--internal-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--internal-smoke", action="store_true", help=argparse.SUPPRESS
    )
    args = parser.parse_args()

    if args.internal_worker:
        if args.internal_output is None:
            parser.error("--internal-output is required for internal workers")
        run_worker(
            profile_kind=args.internal_worker,
            output_dir=args.internal_output,
            smoke=bool(args.internal_smoke),
        )
        return

    if not args.official and not args.smoke:
        parser.error("Choose --smoke or --official")

    run_parent(smoke=bool(args.smoke))


if __name__ == "__main__":
    main()

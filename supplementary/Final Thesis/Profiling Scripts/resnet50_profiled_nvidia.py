#!/usr/bin/env python3

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import platform
import shutil
import sqlite3
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
from torch.autograd.profiler import profile as autograd_profile
from torchvision.models import resnet50


PROJECT_DIR = Path.home() / "Thesis-Testing"
SCRIPT_DIR = PROJECT_DIR / "scripts"
REGULAR_SCRIPT = SCRIPT_DIR / "resnet50_regular_nvidia.py"
PROFILE_SCRIPT = SCRIPT_DIR / "resnet50_profiled_nvidia.py"
DATA_DIR = PROJECT_DIR / "Dataset" / "CIFAR10"
PROFILE_DIR = PROJECT_DIR / "Logs" / "resnetprofiledrun"
TEMP_PROFILE_DIR = (
    Path.home()
    / "Downloads"
    / "Thesis-Testing-Temporary"
    / "ResNet50"
    / "profile_smoke"
)
ONE_EPOCH_PROFILE_DIR = (
    Path.home()
    / "Downloads"
    / "Thesis-Testing-Temporary"
    / "ResNet50"
    / "profile_one_epoch"
)
GPU_ERROR_HISTORY = PROJECT_DIR / "Logs" / "resnet_gpu_error_history.log"
NSYS = Path("/usr/local/bin/nsys")
TRAINING_RANGE_NAME = "RESNET50_TRAINING"
EXPECTED_REGULAR_SCRIPT_SHA256 = (
    "674bfa5f8319c3cc2412b4ea3a7cb13a8192e2dcd9faa3e6a3aafffdcad69b44"
)
SCRIPT_REVISION = "2026-06-20 nvidia-profiled-v4 incremental-batch-operator-profiling"
EXPECTED_NVIDIA_DRIVER = "580.167.08"
MAX_LOG_BYTES = 64 * 1024 * 1024

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


class CappedLogFile:
    """Text log writer with a hard size ceiling to prevent runaway logs."""

    def __init__(self, path: Path, max_bytes: int = MAX_LOG_BYTES):
        self.path = path
        self.max_bytes = max_bytes
        self.file = path.open("w", encoding="utf-8", buffering=1)
        self.bytes_written = 0
        self.capped = False

    def write(self, data):
        if self.capped:
            return len(data)
        encoded = data.encode("utf-8", errors="replace")
        remaining = self.max_bytes - self.bytes_written
        if len(encoded) <= remaining:
            self.file.write(data)
            self.bytes_written += len(encoded)
            return len(data)

        if remaining > 0:
            partial = encoded[:remaining].decode("utf-8", errors="ignore")
            self.file.write(partial)
            self.bytes_written += len(partial.encode("utf-8"))
        self.file.write(
            "\n[LOG CAPPED AT 64 MiB: further output omitted to protect disk space]\n"
        )
        self.file.flush()
        self.capped = True
        return len(data)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


def reset_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_regular_module():
    if not REGULAR_SCRIPT.is_file():
        raise RuntimeError(f"Required regular script is missing: {REGULAR_SCRIPT}")

    actual_sha = sha256_file(REGULAR_SCRIPT)
    if actual_sha != EXPECTED_REGULAR_SCRIPT_SHA256:
        raise RuntimeError(
            "The installed regular NVIDIA script no longer matches the frozen "
            "validated baseline.\n"
            f"Expected SHA-256: {EXPECTED_REGULAR_SCRIPT_SHA256}\n"
            f"Actual SHA-256:   {actual_sha}\n"
            "Stop and resolve the script mismatch before profiling."
        )

    spec = importlib.util.spec_from_file_location(
        "resnet50_regular_nvidia_shared", REGULAR_SCRIPT
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
            "The profiling and regular scripts no longer use the same fixed "
            "workload configuration:\n" + "\n".join(mismatches)
        )

    return module


def append_gpu_error_history(label: str, traceback_text: str) -> None:
    GPU_ERROR_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with GPU_ERROR_HISTORY.open("a", encoding="utf-8") as file:
        file.write("\n" + "=" * 88 + "\n")
        file.write(f"Timestamp: {datetime.now().astimezone().isoformat()}\n")
        file.write(f"Workload: ResNet-50 CIFAR-10 NVIDIA {label}\n")
        file.write(traceback_text.rstrip() + "\n")


def is_gpu_related_error(text: str) -> bool:
    terms = (
        "cuda",
        "cudnn",
        "device-side",
        "gpu",
        "nsight",
        "nsys",
        "nvml",
        "nvidia",
        "out of memory",
        "pynvml",
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


def operator_note(name: str) -> str:
    notes = {
        "aten::convolution_backward": "Convolution backward",
        "aten::convolution": "Convolution forward",
        "aten::_convolution": "Convolution dispatch",
        "aten::cudnn_convolution": "cuDNN convolution",
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


def rank_profile_rows(rows):
    ranked = sorted(
        rows,
        key=lambda row: row["self_gpu_time_ms"],
        reverse=True,
    )
    total_self_ms = sum(row["self_gpu_time_ms"] for row in ranked)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["self_gpu_percent"] = (
            row["self_gpu_time_ms"] / total_self_ms * 100.0
            if total_self_ms > 0
            else float("nan")
        )
    return ranked


def aggregate_profiler_events(profiler, accumulator: dict) -> None:
    """Merge one completed batch profiler into a compact running table."""
    for event in profiler.key_averages():
        self_us, total_us = event_device_times(event)
        if self_us <= 0 and total_us <= 0:
            continue
        row = accumulator.setdefault(
            event.key,
            {
                "operator": event.key,
                "self_gpu_time_ms": 0.0,
                "gpu_total_ms": 0.0,
                "calls": 0,
                "notes": operator_note(event.key),
            },
        )
        row["self_gpu_time_ms"] += self_us / 1000.0
        row["gpu_total_ms"] += total_us / 1000.0
        row["calls"] += int(event.count)


def save_operator_outputs(accumulator: dict, output_dir: Path) -> pd.DataFrame:
    all_rows = [dict(row) for row in accumulator.values()]

    columns = [
        "rank",
        "operator",
        "self_gpu_time_ms",
        "self_gpu_percent",
        "gpu_total_ms",
        "calls",
        "notes",
    ]

    # Preserve every CUDA-timed profiler event for transparency. Framework
    # scopes are diagnostics, while the official table remains ATen-only.
    all_ranked = rank_profile_rows(all_rows)
    all_frame = pd.DataFrame(all_ranked, columns=columns)
    all_frame.to_csv(
        output_dir / "pytorch_all_profile_events.csv",
        index=False,
    )

    operator_rows = [
        dict(row)
        for row in all_rows
        if str(row["operator"]).startswith("aten::")
    ]
    operator_ranked = rank_profile_rows(operator_rows)
    frame = pd.DataFrame(operator_ranked, columns=columns)
    frame.to_csv(output_dir / "pytorch_all_operators.csv", index=False)
    frame.head(10).to_csv(output_dir / "pytorch_top10_operators.csv", index=False)

    (output_dir / "pytorch_profiler_full_table.txt").write_text(
        all_frame.head(200).to_string(index=False),
        encoding="utf-8",
    )
    return frame


def run_worker(
    profile_kind: str,
    output_dir: Path,
    smoke: bool,
    one_epoch: bool = False,
) -> None:
    regular = load_regular_module()
    reset_directory(output_dir)

    log_file = CappedLogFile(output_dir / "training_output.log")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    monitor = None
    nvtx_active = False

    try:
        regular.set_seed(regular.SEED)

        if not DATA_DIR.exists():
            raise RuntimeError(f"Dataset directory does not exist: {DATA_DIR}")
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot detect the NVIDIA GPU.")
        if torch.version.cuda is None:
            raise RuntimeError("This PyTorch build does not report a CUDA runtime.")

        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        compute_capability = torch.cuda.get_device_capability(0)
        expected_gpu = getattr(regular, "EXPECTED_GPU_NAME_FRAGMENT", "RTX 5060 Ti")
        if expected_gpu not in gpu_name:
            raise RuntimeError(
                f"Unexpected GPU selected: {gpu_name}. Expected {expected_gpu}."
            )

        if smoke:
            run_mode = "smoke"
            display_mode = "smoke test"
            epochs_to_run = 1
            max_train_batches = 5
            max_validation_batches = 2
        elif one_epoch:
            run_mode = "one_epoch_profile_test"
            display_mode = "one-epoch full-dataset test"
            epochs_to_run = 1
            max_train_batches = None
            max_validation_batches = None
        else:
            run_mode = "official_profiled"
            display_mode = "official pass"
            epochs_to_run = regular.EPOCHS
            max_train_batches = None
            max_validation_batches = None

        print("=" * 80)
        print(
            f"ResNet-50 CIFAR-10 NVIDIA {profile_kind} profiling "
            f"{display_mode}"
        )
        print("=" * 80)
        print(f"Timestamp: {datetime.now().astimezone().isoformat()}")
        print(f"Output directory: {output_dir}")
        print(f"Dataset directory: {DATA_DIR}")
        print(f"Python: {platform.python_version()}")
        print(f"Platform: {platform.platform()}")
        print(f"PyTorch: {torch.__version__}")
        print(f"Torchvision: {torchvision.__version__}")
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"GPU: {gpu_name}")
        print(
            "Compute capability: "
            f"{compute_capability[0]}.{compute_capability[1]}"
        )
        print(f"Profile script revision: {SCRIPT_REVISION}")
        print()

        trainset, testset, trainloader, testloader = regular.build_loaders()
        order_sha256 = regular.training_order_sha256(
            len(trainset),
            epochs_to_run,
            max_train_batches,
        )

        model = resnet50(num_classes=10)
        initial_model_sha256 = regular.model_state_sha256(model)
        model = model.to(device)

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

        monitor = regular.TelemetryMonitor(
            interval_s=regular.SAMPLE_INTERVAL_S
        )
        if monitor.gpu_name != gpu_name:
            raise RuntimeError(
                "PyTorch and NVML selected different GPUs: "
                f"PyTorch={gpu_name!r}, NVML={monitor.gpu_name!r}"
            )

        if monitor.driver_version != EXPECTED_NVIDIA_DRIVER:
            raise RuntimeError(
                "NVIDIA driver changed from the frozen profiling baseline: "
                f"expected {EXPECTED_NVIDIA_DRIVER}, found "
                f"{monitor.driver_version}."
            )

        config = {
            "mode": run_mode,
            "profile_kind": profile_kind,
            "seed": regular.SEED,
            "model": "torchvision.models.resnet50",
            "model_initialization": "from scratch",
            "initial_model_sha256": initial_model_sha256,
            "training_order_sha256": order_sha256,
            "dataset": "CIFAR-10",
            "dataset_path": str(DATA_DIR),
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
            "training_shuffle": True,
            "validation_shuffle": False,
            "telemetry_interval_s": regular.SAMPLE_INTERVAL_S,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "nvidia_driver": monitor.driver_version,
            "gpu": gpu_name,
            "compute_capability": (
                f"{compute_capability[0]}.{compute_capability[1]}"
            ),
            "max_train_batches": max_train_batches,
            "max_validation_batches": max_validation_batches,
            "training_range_name": TRAINING_RANGE_NAME,
            "regular_script_sha256": sha256_file(REGULAR_SCRIPT),
            "profile_script_sha256": sha256_file(Path(__file__).resolve()),
            "profile_script_revision": SCRIPT_REVISION,
            "validation_metric": "top-1 accuracy",
            "operator_profiler_backend": (
                "torch.autograd.profiler CUDA-event timing "
                "(use_device=cuda, use_kineto=False)"
                if profile_kind == "torch_operator"
                else None
            ),
            "official_operator_ranking_filter": (
                "ATen operators only (operator name starts with aten::); "
                "all CUDA-timed scopes retained separately"
                if profile_kind == "torch_operator"
                else None
            ),
            "operator_capture_strategy": (
                "One autograd CUDA-event profiler context per training batch; "
                "DataLoader next() occurs outside the profiler; event totals "
                "are merged immediately and released after every batch"
                if profile_kind == "torch_operator"
                else None
            ),
            "full_trace_scope": (
                "Representative first profiled batch only; full-run operator "
                "totals are stored in aggregated CSV tables"
                if profile_kind == "torch_operator"
                else None
            ),
            "log_size_cap_bytes": MAX_LOG_BYTES,
        }
        with (output_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

        epoch_rows = []
        total_samples_processed = 0
        total_batches_processed = 0
        first_batch_output_dtype = "not_recorded"

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        operator_accumulator = {}
        operator_aggregation_overhead_s = 0.0
        operator_capture_segment_time_s = 0.0
        representative_trace_saved = False
        full_pass_wall_start = None

        print("Starting monitored AMP training...")
        print(
            "Precision: AMP FP16 | Optimizer: SGD | "
            f"Batch size: {regular.BATCH_SIZE} | Epochs: {epochs_to_run}"
        )
        if profile_kind == "torch_operator":
            print(
                "Operator capture: one CUDA-event profiler context per batch; "
                "DataLoader fetching is outside the profiler context."
            )

        monitor.start()

        if profile_kind == "nsight_systems":
            torch.cuda.nvtx.range_push(TRAINING_RANGE_NAME)
            nvtx_active = True
            print(f"NVTX training range started: {TRAINING_RANGE_NAME}")

        full_pass_wall_start = time.perf_counter()
        total_training_time = 0.0

        for epoch in range(epochs_to_run):
            model.train()
            epoch_wall_start = time.perf_counter()
            epoch_profiled_training_time = 0.0
            running_loss = 0.0
            epoch_samples = 0
            epoch_batches = 0

            # Iterator creation and every next() call stay outside CUDA profiler
            # scopes. Persistent workers preserve the fixed regular-run loader.
            train_iterator = iter(trainloader)
            batch_index = 0

            while True:
                if (
                    max_train_batches is not None
                    and batch_index >= max_train_batches
                ):
                    break

                batch_segment_start = time.perf_counter()
                try:
                    images, labels = next(train_iterator)
                except StopIteration:
                    break

                if profile_kind == "torch_operator":
                    with autograd_profile(
                        use_device="cuda",
                        use_kineto=False,
                        record_shapes=False,
                        profile_memory=False,
                        with_stack=False,
                    ) as batch_profiler:
                        images = images.to(device, non_blocking=True)
                        labels = labels.to(device, non_blocking=True)
                        optimizer.zero_grad(set_to_none=True)

                        with torch.amp.autocast(
                            device_type="cuda",
                            dtype=torch.float16,
                            enabled=True,
                        ):
                            outputs = model(images)
                            loss = criterion(outputs, labels)

                        if first_batch_output_dtype == "not_recorded":
                            first_batch_output_dtype = str(outputs.dtype)
                            print(
                                "AMP verification | "
                                f"GradScaler enabled: {scaler.is_enabled()} | "
                                "First-batch output dtype: "
                                f"{first_batch_output_dtype}"
                            )

                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    # Context exit completes CUDA-event timing. End the timed
                    # training segment before compact post-processing begins.
                    batch_segment_time = time.perf_counter() - batch_segment_start
                    epoch_profiled_training_time += batch_segment_time
                    operator_capture_segment_time_s += batch_segment_time

                    aggregation_start = time.perf_counter()
                    aggregate_profiler_events(
                        batch_profiler,
                        operator_accumulator,
                    )
                    if not representative_trace_saved:
                        batch_profiler.export_chrome_trace(
                            str(
                                output_dir
                                / "pytorch_trace_representative_batch.json"
                            )
                        )
                        representative_trace_saved = True
                    operator_aggregation_overhead_s += (
                        time.perf_counter() - aggregation_start
                    )
                    del batch_profiler
                    if (total_batches_processed + 1) % 50 == 0:
                        gc.collect()
                else:
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)

                    with torch.amp.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=True,
                    ):
                        outputs = model(images)
                        loss = criterion(outputs, labels)

                    if first_batch_output_dtype == "not_recorded":
                        first_batch_output_dtype = str(outputs.dtype)
                        print(
                            "AMP verification | "
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
                batch_index += 1

            torch.cuda.synchronize(device)
            # Primary profiled performance uses the complete training-pass wall
            # time, including per-batch profiler context management and compact
            # event aggregation. This keeps timing and continuous telemetry over
            # the same interval. Validation remains outside both.
            epoch_time = time.perf_counter() - epoch_wall_start

            total_training_time += epoch_time
            if epoch_samples <= 0:
                raise RuntimeError("No training samples were processed.")
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
        full_pass_wall_time_s = time.perf_counter() - full_pass_wall_start

        if nvtx_active:
            torch.cuda.nvtx.range_pop()
            nvtx_active = False
            print(f"NVTX training range ended: {TRAINING_RANGE_NAME}")

        monitor.stop()

        print(f"Total Training Time: {total_training_time:.2f}s")
        print("Training complete.")
        print()

        epoch_df = pd.DataFrame(epoch_rows)
        epoch_df.to_csv(output_dir / "epoch_metrics.csv", index=False)

        telemetry_df = monitor.dataframe()
        telemetry_df.to_csv(output_dir / "telemetry.csv", index=False)
        with (output_dir / "nvml_api_errors.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(monitor.api_errors, file, indent=2, sort_keys=True)
        regular.write_nvml_error_log(output_dir, monitor)
        regular.validate_telemetry(telemetry_df)

        if profile_kind == "torch_operator":
            print("Writing incrementally aggregated PyTorch operator statistics...")
            operator_df = save_operator_outputs(
                operator_accumulator,
                output_dir,
            )
            if operator_df.empty:
                raise RuntimeError("No GPU operator statistics were produced.")
            if not representative_trace_saved:
                raise RuntimeError(
                    "Representative PyTorch batch trace was not created."
                )
            print(f"Top PyTorch operator: {operator_df.iloc[0]['operator']}")
            print(
                "Incremental operator aggregation included in profiled pass "
                f"wall time: {operator_aggregation_overhead_s:.3f}s"
            )
            print()

        expected_batches_per_epoch = math.ceil(len(trainset) / regular.BATCH_SIZE)
        expected_samples = (
            min(len(trainset), max_train_batches * regular.BATCH_SIZE)
            if smoke
            else len(trainset) * epochs_to_run
        )
        expected_batches = (
            max_train_batches
            if smoke
            else expected_batches_per_epoch * epochs_to_run
        )
        if len(epoch_rows) != epochs_to_run:
            raise RuntimeError(
                f"Incomplete pass: expected {epochs_to_run} epochs, "
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
        if not all(math.isfinite(row["loss"]) for row in epoch_rows):
            raise RuntimeError("One or more epoch losses are non-finite.")
        if monitor.thermal_throttle_seen:
            raise RuntimeError(
                "NVML detected thermal throttling during the profiled pass."
            )

        print("Starting final validation outside the timed training region...")
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

        expected_validation_samples = (
            min(len(testset), max_validation_batches * regular.BATCH_SIZE)
            if smoke
            else len(testset)
        )
        if validation_total != expected_validation_samples:
            raise RuntimeError(
                f"Incomplete validation: expected {expected_validation_samples} "
                f"samples, got {validation_total}."
            )

        validation_accuracy = correct / validation_total
        if not 0.0 <= validation_accuracy <= 1.0:
            raise RuntimeError("Validation accuracy is outside the valid range.")
        print(
            f"Final Validation Score: {validation_accuracy:.4f} "
            f"({validation_total} samples, {validation_batches} batches)"
        )

        throughput = total_samples_processed / total_training_time
        batch_latency_ms = (
            total_training_time / total_batches_processed
        ) * 1000.0
        average_epoch_time = total_training_time / epochs_to_run
        mean_epoch_loop_time = statistics.mean(
            row["epoch_time_s"] for row in epoch_rows
        )

        average_power_w = regular.series_stat(telemetry_df, "power_w", "mean")
        average_gpu_util = regular.series_stat(
            telemetry_df,
            "gpu_util_percent",
            "mean",
        )
        average_vram_mb = regular.series_stat(
            telemetry_df,
            "vram_used_mb",
            "mean",
        )
        peak_vram_mb = regular.series_stat(
            telemetry_df,
            "vram_used_mb",
            "max",
        )
        average_gpu_temp_c = regular.series_stat(
            telemetry_df,
            "gpu_temperature_c",
            "mean",
        )
        peak_gpu_temp_c = regular.series_stat(
            telemetry_df,
            "gpu_temperature_c",
            "max",
        )
        average_gfx_clock_mhz = regular.series_stat(
            telemetry_df,
            "graphics_clock_mhz",
            "mean",
        )
        average_mem_clock_mhz = regular.series_stat(
            telemetry_df,
            "memory_clock_mhz",
            "mean",
        )
        average_cpu_util = regular.series_stat(
            telemetry_df,
            "cpu_util_percent",
            "mean",
        )
        average_cpu_temp_c = regular.series_stat(
            telemetry_df,
            "cpu_package_temperature_c",
            "mean",
        )
        average_system_ram_mb = regular.series_stat(
            telemetry_df,
            "system_ram_used_mb",
            "mean",
        )
        peak_system_ram_mb = regular.series_stat(
            telemetry_df,
            "system_ram_used_mb",
            "max",
        )
        average_disk_read_mb_s = regular.series_stat(
            telemetry_df,
            "disk_read_mb_s",
            "mean",
        )
        average_disk_write_mb_s = regular.series_stat(
            telemetry_df,
            "disk_write_mb_s",
            "mean",
        )
        peak_disk_read_mb_s = regular.series_stat(
            telemetry_df,
            "disk_read_mb_s",
            "max",
        )
        peak_disk_write_mb_s = regular.series_stat(
            telemetry_df,
            "disk_write_mb_s",
            "max",
        )
        torch_peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

        performance_per_watt = (
            throughput / average_power_w
            if not np.isnan(average_power_w) and average_power_w > 0
            else float("nan")
        )

        stability_notes = (
            "Completed successfully; no Python, CUDA, NVIDIA driver, NVML, "
            "or profiler fatal errors detected."
        )
        if monitor.api_error_count:
            stability_notes += (
                f" NVML recorded {monitor.api_error_count} recoverable API errors."
            )
        else:
            stability_notes += " All required NVML telemetry queries succeeded."
        stability_notes += " No thermal throttling was observed."

        summary = {
            "mode": run_mode,
            "profile_kind": profile_kind,
            "throughput_samples_s": throughput,
            "batch_latency_ms_batch": batch_latency_ms,
            "total_training_time_s": total_training_time,
            "average_epoch_time_s": average_epoch_time,
            "mean_epoch_loop_time_s": mean_epoch_loop_time,
            "full_pass_wall_time_s": full_pass_wall_time_s,
            "operator_capture_segment_time_s": (
                operator_capture_segment_time_s
                if profile_kind == "torch_operator"
                else 0.0
            ),
            "operator_incremental_aggregation_time_s": (
                operator_aggregation_overhead_s
                if profile_kind == "torch_operator"
                else 0.0
            ),
            "profiled_training_time_definition": (
                "Complete training-pass wall time from before each DataLoader "
                "fetch through per-batch CUDA-event capture and incremental "
                "aggregation; validation is excluded"
                if profile_kind == "torch_operator"
                else "Continuous NVTX training-range wall time"
            ),
            "average_gpu_power_w": average_power_w,
            "performance_per_watt_samples_s_w": performance_per_watt,
            "average_gpu_util_percent": average_gpu_util,
            "average_vram_usage_mb": average_vram_mb,
            "peak_vram_usage_mb": peak_vram_mb,
            "torch_peak_allocated_memory_mb": torch_peak_allocated_mb,
            "average_gpu_temperature_c": average_gpu_temp_c,
            "peak_gpu_temperature_c": peak_gpu_temp_c,
            "average_graphics_clock_mhz": average_gfx_clock_mhz,
            "average_memory_clock_mhz": average_mem_clock_mhz,
            "thermal_throttling_observed": monitor.thermal_throttle_seen,
            "final_validation_score": validation_accuracy,
            "validation_metric": "top-1 accuracy",
            "operator_profiler_backend": (
                "torch.autograd.profiler CUDA-event timing "
                "(use_device=cuda, use_kineto=False)"
                if profile_kind == "torch_operator"
                else None
            ),
            "official_operator_ranking_filter": (
                "ATen operators only (operator name starts with aten::); "
                "all CUDA-timed scopes retained separately"
                if profile_kind == "torch_operator"
                else None
            ),
            "operator_capture_strategy": (
                "One profiler context per batch; DataLoader fetch outside; "
                "immediate aggregate-and-release"
                if profile_kind == "torch_operator"
                else None
            ),
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
            "nvml_api_error_count": monitor.api_error_count,
            "amp_grad_scaler_enabled": scaler.is_enabled(),
            "amp_first_batch_output_dtype": first_batch_output_dtype,
            "initial_model_sha256": initial_model_sha256,
            "training_order_sha256": order_sha256,
            "regular_script_sha256": sha256_file(REGULAR_SCRIPT),
            "profile_script_sha256": sha256_file(Path(__file__).resolve()),
            "stability_error_notes": stability_notes,
        }
        pd.DataFrame([summary]).to_csv(
            output_dir / "summary.csv",
            index=False,
        )

        with (output_dir / "summary.txt").open("w", encoding="utf-8") as file:
            file.write(
                f"ResNet-50 NVIDIA {profile_kind} "
                f"{display_mode.title()}\n"
            )
            file.write("=" * 64 + "\n")
            file.write(f"Throughput: {throughput:.2f} samples/s\n")
            file.write(f"Batch Latency: {batch_latency_ms:.2f} ms/batch\n")
            file.write(f"Total Training Time: {total_training_time:.2f} s\n")
            file.write(f"Full Pass Wall Time: {full_pass_wall_time_s:.2f} s\n")
            if profile_kind == "torch_operator":
                file.write(
                    "Operator Capture Segments: "
                    f"{operator_capture_segment_time_s:.2f} s\n"
                )
                file.write(
                    "Incremental Operator Aggregation (included in pass): "
                    f"{operator_aggregation_overhead_s:.2f} s\n"
                )
            file.write(f"Average Epoch Time: {average_epoch_time:.2f} s\n")
            file.write(f"Average GPU Power Draw: {average_power_w:.2f} W\n")
            file.write(
                f"Performance per Watt: {performance_per_watt:.2f} samples/s/W\n"
            )
            file.write(f"Average GPU Utilization: {average_gpu_util:.2f} %\n")
            file.write(f"Average VRAM Usage: {average_vram_mb:.2f} MB\n")
            file.write(f"Peak VRAM Usage: {peak_vram_mb:.2f} MB\n")
            file.write(f"Average GPU Temperature: {average_gpu_temp_c:.2f} C\n")
            file.write(f"Final Validation Top-1 Accuracy: {validation_accuracy:.4f}\n")
            file.write(f"Peak System RAM: {peak_system_ram_mb:.2f} MB\n")
            file.write(f"Average System RAM: {average_system_ram_mb:.2f} MB\n")
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
                "AMP: GradScaler enabled; first-batch output dtype "
                f"{first_batch_output_dtype}\n"
            )
            file.write(f"Initial Model SHA-256: {initial_model_sha256}\n")
            file.write(f"Training Order SHA-256: {order_sha256}\n")
            file.write(f"Stability / Error Notes: {stability_notes}\n")

        valid = {
            "status": "VALID",
            "timestamp": datetime.now().astimezone().isoformat(),
            "mode": run_mode,
            "profile_kind": profile_kind,
            "epochs_completed": len(epoch_rows),
            "total_samples_processed": total_samples_processed,
            "total_batches_processed": total_batches_processed,
            "validation_samples": validation_total,
            "validation_score": validation_accuracy,
            "telemetry_samples": len(telemetry_df),
            "thermal_throttling_observed": monitor.thermal_throttle_seen,
            "initial_model_sha256": initial_model_sha256,
            "training_order_sha256": order_sha256,
            "regular_script_sha256": sha256_file(REGULAR_SCRIPT),
            "profile_script_sha256": sha256_file(Path(__file__).resolve()),
        }
        with (output_dir / "WORKER_VALID.json").open(
            "w",
            encoding="utf-8",
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
        if nvtx_active:
            try:
                torch.cuda.nvtx.range_pop()
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
                        output_dir / "telemetry_partial.csv",
                        index=False,
                    )
                with (output_dir / "nvml_api_errors.json").open(
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(monitor.api_errors, file, indent=2, sort_keys=True)
            except Exception:
                pass
        print("\nPROFILE WORKER FAILED")
        traceback.print_exc()
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def _table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def _table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    safe_table = table.replace('"', '""')
    rows = connection.execute(f'PRAGMA table_info("{safe_table}")').fetchall()
    return [str(row[1]) for row in rows]


def _read_table(connection: sqlite3.Connection, table: str) -> pd.DataFrame:
    safe_table = table.replace('"', '""')
    return pd.read_sql_query(f'SELECT * FROM "{safe_table}"', connection)


def _first_existing(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    return None


def _resolve_string_column(
    frame: pd.DataFrame,
    connection: sqlite3.Connection,
    direct_candidates,
    id_candidates,
) -> pd.Series:
    # Nsight schemas sometimes use names such as demangledName/shortName for
    # integer StringIds foreign keys, while other versions expose literal text.
    # Prefer a candidate as direct text only when it actually contains strings.
    for direct in direct_candidates:
        if direct not in frame.columns:
            continue
        values = frame[direct]
        non_null = values.dropna()
        if non_null.empty:
            continue
        if not pd.api.types.is_numeric_dtype(non_null):
            return values.fillna("").astype(str)

    tables = _table_names(connection)
    if "StringIds" in tables:
        string_df = _read_table(connection, "StringIds")
        if {"id", "value"}.issubset(string_df.columns):
            mapping = dict(zip(string_df["id"], string_df["value"]))
            for id_column in id_candidates:
                if id_column in frame.columns:
                    resolved = frame[id_column].map(mapping)
                    if resolved.notna().any():
                        return resolved.fillna("").astype(str)

    return pd.Series([""] * len(frame), index=frame.index, dtype="object")


def _normalize_intervals(
    frame: pd.DataFrame,
    start_column: str,
    end_column: str,
    window_start: int,
    window_end: int,
) -> pd.DataFrame:
    work = frame.copy()
    work["start_ns"] = pd.to_numeric(work[start_column], errors="coerce")
    work["end_ns"] = pd.to_numeric(work[end_column], errors="coerce")
    work = work.dropna(subset=["start_ns", "end_ns"])
    work["start_ns"] = work["start_ns"].astype(np.int64)
    work["end_ns"] = work["end_ns"].astype(np.int64)
    work = work[(work["end_ns"] > window_start) & (work["start_ns"] < window_end)]
    work["start_ns"] = work["start_ns"].clip(lower=window_start)
    work["end_ns"] = work["end_ns"].clip(upper=window_end)
    work = work[work["end_ns"] >= work["start_ns"]].copy()
    work["duration_ns"] = work["end_ns"] - work["start_ns"]
    return work


def _union_duration_ns(intervals) -> int:
    ordered = sorted(
        (int(start), int(end))
        for start, end in intervals
        if int(end) > int(start)
    )
    if not ordered:
        return 0

    total = 0
    current_start, current_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    total += current_end - current_start
    return int(total)


def _find_training_window(
    connection: sqlite3.Connection,
) -> tuple[int, int, str]:
    tables = _table_names(connection)
    candidate_tables = [
        name
        for name in tables
        if "NVTX" in name.upper() and "EVENT" in name.upper()
    ]
    if "NVTX_EVENTS" in tables:
        candidate_tables.remove("NVTX_EVENTS") if "NVTX_EVENTS" in candidate_tables else None
        candidate_tables.insert(0, "NVTX_EVENTS")

    diagnostics = []
    for table in candidate_tables:
        frame = _read_table(connection, table)
        if frame.empty:
            continue
        start_column = _first_existing(frame.columns, ("start", "startNs", "startTime"))
        end_column = _first_existing(frame.columns, ("end", "endNs", "endTime"))
        if start_column is None or end_column is None:
            diagnostics.append(f"{table}: missing start/end columns")
            continue

        names = _resolve_string_column(
            frame,
            connection,
            direct_candidates=("text", "message", "name"),
            id_candidates=("textId", "messageId", "nameId"),
        )
        matches = frame[names.str.contains(TRAINING_RANGE_NAME, regex=False, na=False)].copy()
        if matches.empty:
            diagnostics.append(f"{table}: marker not found")
            continue

        matches["_start"] = pd.to_numeric(matches[start_column], errors="coerce")
        matches["_end"] = pd.to_numeric(matches[end_column], errors="coerce")
        matches = matches.dropna(subset=["_start", "_end"])
        matches = matches[matches["_end"] > matches["_start"]]
        if matches.empty:
            diagnostics.append(f"{table}: marker rows had no positive duration")
            continue
        matches["_duration"] = matches["_end"] - matches["_start"]
        row = matches.sort_values("_duration", ascending=False).iloc[0]
        return int(row["_start"]), int(row["_end"]), table

    raise RuntimeError(
        f"NVTX range {TRAINING_RANGE_NAME!r} was not found in the Nsight "
        "Systems SQLite export. Diagnostics: " + "; ".join(diagnostics)
    )


def _kernel_table(tables: set[str]) -> str:
    preferred = (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
    )
    for table in preferred:
        if table in tables:
            return table
    candidates = [
        table
        for table in tables
        if "CUPTI" in table.upper() and "KERNEL" in table.upper()
    ]
    if not candidates:
        raise RuntimeError("Nsight SQLite export contains no CUDA kernel table.")
    return sorted(candidates)[0]


def _memcpy_tables(tables: set[str]) -> list[str]:
    preferred = [
        "CUPTI_ACTIVITY_KIND_MEMCPY",
        "CUPTI_ACTIVITY_KIND_MEMCPY2",
    ]
    found = [name for name in preferred if name in tables]
    if found:
        return found
    return sorted(
        table
        for table in tables
        if "CUPTI" in table.upper() and "MEMCPY" in table.upper()
    )


def parse_nsys_sqlite(sqlite_path: Path, destination: Path) -> dict:
    if not sqlite_path.is_file() or sqlite_path.stat().st_size <= 0:
        raise RuntimeError(f"Nsight SQLite export is missing or empty: {sqlite_path}")

    with sqlite3.connect(sqlite_path) as connection:
        tables = _table_names(connection)
        if not tables:
            raise RuntimeError("Nsight SQLite export contains no tables.")

        training_start, training_end, marker_table = _find_training_window(connection)
        training_duration_ns = training_end - training_start
        if training_duration_ns <= 0:
            raise RuntimeError("The NVTX training range has an invalid duration.")

        kernel_table = _kernel_table(tables)
        kernel_raw = _read_table(connection, kernel_table)
        if kernel_raw.empty:
            raise RuntimeError("Nsight kernel table is empty.")
        kernel_start = _first_existing(
            kernel_raw.columns,
            ("start", "startNs", "startTime"),
        )
        kernel_end = _first_existing(
            kernel_raw.columns,
            ("end", "endNs", "endTime"),
        )
        if kernel_start is None or kernel_end is None:
            raise RuntimeError(
                f"Kernel table {kernel_table} is missing start/end columns. "
                f"Columns: {list(kernel_raw.columns)}"
            )
        kernel_names = _resolve_string_column(
            kernel_raw,
            connection,
            direct_candidates=(
                "demangledName",
                "shortName",
                "name",
                "kernelName",
            ),
            id_candidates=(
                "demangledName",
                "shortName",
                "nameId",
                "kernelNameId",
            ),
        )
        kernels = _normalize_intervals(
            kernel_raw,
            kernel_start,
            kernel_end,
            training_start,
            training_end,
        )
        kernels["kernel_name"] = kernel_names.loc[kernels.index].replace("", "UNKNOWN_KERNEL")
        if kernels.empty:
            raise RuntimeError("No CUDA kernels were found inside the NVTX training range.")

        grouped = (
            kernels.groupby("kernel_name", dropna=False)["duration_ns"]
            .agg(["count", "sum", "mean", "min", "max", "std"])
            .reset_index()
            .rename(
                columns={
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
        summed_kernel_duration_ns = int(grouped["total_duration_ns"].sum())
        grouped.insert(0, "rank", range(1, len(grouped) + 1))
        grouped["total_duration_ms"] = grouped["total_duration_ns"] / 1e6
        grouped["average_duration_us"] = grouped["average_duration_ns"] / 1e3
        grouped["percentage_of_summed_kernel_time"] = (
            grouped["total_duration_ns"] / summed_kernel_duration_ns * 100.0
            if summed_kernel_duration_ns > 0
            else float("nan")
        )
        grouped = grouped[
            [
                "rank",
                "kernel_name",
                "calls",
                "total_duration_ms",
                "average_duration_us",
                "percentage_of_summed_kernel_time",
                "minimum_duration_ns",
                "maximum_duration_ns",
                "stddev_duration_ns",
                "total_duration_ns",
            ]
        ]
        grouped.to_csv(destination / "nsys_parsed_kernel_summary.csv", index=False)
        grouped.head(5).to_csv(destination / "nvidia_top5_kernels.csv", index=False)
        kernels.to_csv(destination / "nsys_training_kernel_trace.csv", index=False)

        memcpy_frames = []
        memcpy_table_names = _memcpy_tables(tables)
        for table in memcpy_table_names:
            raw = _read_table(connection, table)
            if raw.empty:
                continue
            start_col = _first_existing(raw.columns, ("start", "startNs", "startTime"))
            end_col = _first_existing(raw.columns, ("end", "endNs", "endTime"))
            if start_col is None or end_col is None:
                continue
            normalized = _normalize_intervals(
                raw,
                start_col,
                end_col,
                training_start,
                training_end,
            )
            normalized["source_table"] = table
            copy_kind_col = _first_existing(
                normalized.columns,
                ("copyKind", "kind", "memcpyKind"),
            )
            if copy_kind_col is not None:
                normalized["copy_kind"] = normalized[copy_kind_col].astype(str)
            else:
                normalized["copy_kind"] = "unknown"
            memcpy_frames.append(normalized)

        if memcpy_frames:
            memcopies = pd.concat(memcpy_frames, ignore_index=True)
        else:
            memcopies = pd.DataFrame(
                columns=["start_ns", "end_ns", "duration_ns", "copy_kind", "source_table"]
            )
        memcopies.to_csv(destination / "nsys_training_memcpy_trace.csv", index=False)

        kernel_busy_union_ns = _union_duration_ns(
            kernels[["start_ns", "end_ns"]].itertuples(index=False, name=None)
        )
        kernel_idle_ns = max(0, training_duration_ns - kernel_busy_union_ns)
        kernel_idle_percent = kernel_idle_ns / training_duration_ns * 100.0

        cuda_activity_intervals = list(
            kernels[["start_ns", "end_ns"]].itertuples(index=False, name=None)
        )
        if not memcopies.empty:
            cuda_activity_intervals.extend(
                memcopies[["start_ns", "end_ns"]].itertuples(index=False, name=None)
            )
        cuda_busy_union_ns = _union_duration_ns(cuda_activity_intervals)
        cuda_idle_ns = max(0, training_duration_ns - cuda_busy_union_ns)
        cuda_idle_percent = cuda_idle_ns / training_duration_ns * 100.0

        memcpy_total_duration_ns = (
            int(pd.to_numeric(memcopies["duration_ns"], errors="coerce").fillna(0).sum())
            if not memcopies.empty
            else 0
        )
        memcpy_calls = int(len(memcopies))
        average_memcpy_duration_us = (
            memcpy_total_duration_ns / memcpy_calls / 1e3
            if memcpy_calls > 0
            else float("nan")
        )

        top_kernel = grouped.iloc[0]
        metrics = {
            "training_range_name": TRAINING_RANGE_NAME,
            "training_start_timestamp_ns": training_start,
            "training_end_timestamp_ns": training_end,
            "training_window_duration_s": training_duration_ns / 1e9,
            "total_kernel_launches": int(len(kernels)),
            "unique_kernel_names": int(len(grouped)),
            "summed_kernel_duration_s": summed_kernel_duration_ns / 1e9,
            "kernel_busy_union_s": kernel_busy_union_ns / 1e9,
            "gpu_kernel_idle_s": kernel_idle_ns / 1e9,
            "gpu_kernel_idle_percent": kernel_idle_percent,
            "cuda_activity_busy_union_s": cuda_busy_union_ns / 1e9,
            "cuda_activity_idle_s": cuda_idle_ns / 1e9,
            "cuda_activity_idle_percent": cuda_idle_percent,
            "memory_copy_calls": memcpy_calls,
            "memory_copy_total_duration_ms": memcpy_total_duration_ns / 1e6,
            "average_memory_copy_duration_us": average_memcpy_duration_us,
            "top_kernel": str(top_kernel["kernel_name"]),
            "top_kernel_calls": int(top_kernel["calls"]),
            "top_kernel_cumulative_duration_ms": float(top_kernel["total_duration_ms"]),
            "top_kernel_percentage_of_summed_kernel_time": float(
                top_kernel["percentage_of_summed_kernel_time"]
            ),
            "nvtx_marker_table": marker_table,
            "kernel_table": kernel_table,
            "memcpy_tables": ",".join(memcpy_table_names),
            "training_window_filter_applied": True,
        }
        pd.DataFrame([metrics]).to_csv(
            destination / "nsys_training_metrics.csv",
            index=False,
        )
        with (destination / "nsys_training_metrics.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metrics, file, indent=2)

        metadata = {
            "sqlite_file": str(sqlite_path),
            "sqlite_tables": sorted(tables),
            "nvtx_marker_table": marker_table,
            "kernel_table": kernel_table,
            "memcpy_tables": memcpy_table_names,
            "training_marker": TRAINING_RANGE_NAME,
            "training_start_timestamp_ns": training_start,
            "training_end_timestamp_ns": training_end,
            "training_marker_duration_s": training_duration_ns / 1e9,
            "raw_kernel_rows": int(len(kernel_raw)),
            "training_kernel_rows": int(len(kernels)),
            "unique_training_kernels": int(len(grouped)),
            "training_memcpy_rows": memcpy_calls,
            "training_window_filter_applied": True,
            "idle_definition": (
                "gpu_kernel_idle is NVTX training-window time not covered by "
                "the union of CUDA kernel intervals; cuda_activity_idle also "
                "counts CUDA memcpy intervals as busy."
            ),
        }
        with (destination / "nsys_parse_metadata.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(metadata, file, indent=2)

        return metrics


def validate_worker(
    worker_dir: Path,
    kind: str,
    smoke: bool,
    one_epoch: bool = False,
) -> None:
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

    if smoke:
        expected_samples = 1280
        expected_batches = 5
        expected_validation = 512
    elif one_epoch:
        expected_samples = 50000
        expected_batches = 196
        expected_validation = 10000
    else:
        expected_samples = 500000
        expected_batches = 1960
        expected_validation = 10000
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
    if int(summary["validation_samples"]) != expected_validation:
        raise RuntimeError(
            f"{kind} worker validated {summary['validation_samples']} samples; "
            f"expected {expected_validation}."
        )
    if str(summary["amp_first_batch_output_dtype"]) != "torch.float16":
        raise RuntimeError(f"{kind} worker did not verify FP16 autocast output.")
    thermal_value = summary["thermal_throttling_observed"]
    thermal_observed = (
        bool(thermal_value)
        if isinstance(thermal_value, (bool, np.bool_))
        else str(thermal_value).strip().lower() in {"true", "1", "yes"}
    )
    if thermal_observed:
        raise RuntimeError(f"{kind} worker reported thermal throttling.")


def run_parent(smoke: bool, one_epoch: bool = False) -> None:
    if smoke:
        output_dir = TEMP_PROFILE_DIR
        run_mode = "smoke"
        display_mode = "smoke test"
    elif one_epoch:
        output_dir = ONE_EPOCH_PROFILE_DIR
        run_mode = "one_epoch_profile_test"
        display_mode = "one-epoch full-dataset test"
    else:
        output_dir = PROFILE_DIR
        run_mode = "official_profiled"
        display_mode = "official"
    reset_directory(output_dir)

    log_file = CappedLogFile(output_dir / "training_output.log")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    try:
        regular = load_regular_module()
        if not PROFILE_SCRIPT.is_file():
            raise RuntimeError(
                "Run this script from its installed location: "
                f"{PROFILE_SCRIPT}"
            )
        if not NSYS.is_file():
            raise RuntimeError(f"Nsight Systems CLI is missing: {NSYS}")

        torch_dir = output_dir / "pytorch_operator_pass"
        nsys_worker_dir = output_dir / "nsight_systems_worker_pass"
        raw_dir = output_dir / "nsight_systems_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        report_prefix = raw_dir / "resnet50_nsys"
        report_path = Path(str(report_prefix) + ".nsys-rep")
        sqlite_path = raw_dir / "resnet50_nsys.sqlite"

        parent_config = {
            "mode": run_mode,
            "workflow": "two_pass_single_command",
            "primary_profile_metrics_source": "pytorch_operator_pass",
            "operator_profiler": (
                "torch.autograd.profiler.profile with CUDA-event timing "
                "(use_device=cuda, use_kineto=False)"
            ),
            "operator_profiler_compatibility_note": (
                "The Kineto-backed torch.profiler path produced CPU-only "
                "events on PyTorch 2.9.1+cu130 with this RTX 5060 Ti despite "
                "a matching, loadable CUPTI runtime. The verified legacy "
                "CUDA-event backend is used for NVIDIA operator timing. "
                "Official rankings are restricted to aten:: operators; "
                "unfiltered CUDA-timed scopes are retained separately. "
                "Version 4 uses one profiler context per batch, keeps each "
                "DataLoader next() outside CUDA profiling, aggregates event "
                "totals immediately, and releases the batch profiler before "
                "continuing."
            ),
            "official_operator_ranking_filter": (
                "ATen operators only (operator name starts with aten::)"
            ),
            "kernel_profiler": (
                "NVIDIA Nsight Systems CUDA+NVTX trace with exact "
                "NVTX training-window filtering"
            ),
            "reason_for_two_passes": (
                "PyTorch operator profiling and Nsight Systems tracing are "
                "executed separately to avoid profiler interference."
            ),
            "both_passes_use_identical_fixed_workload": True,
            "operator_memory_safety": (
                "Incremental per-batch aggregation; no full-run profiler "
                "event object is retained in memory"
            ),
            "log_size_cap_bytes": MAX_LOG_BYTES,
            "expected_nvidia_driver": EXPECTED_NVIDIA_DRIVER,
            "training_range_name": TRAINING_RANGE_NAME,
            "regular_script_sha256": sha256_file(REGULAR_SCRIPT),
            "profile_script_sha256": sha256_file(PROFILE_SCRIPT),
            "profile_script_revision": SCRIPT_REVISION,
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        with (output_dir / "profile_workflow_config.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(parent_config, file, indent=2)

        worker_mode_args = []
        if smoke:
            worker_mode_args.append("--internal-smoke")
        elif one_epoch:
            worker_mode_args.append("--internal-one-epoch")

        print("=" * 88)
        print(
            "ResNet-50 NVIDIA profiled workflow "
            f"({display_mode})"
        )
        print("=" * 88)
        print(f"Output directory: {output_dir}")
        print(
            "This command performs two separate controlled passes: "
            "PyTorch operators, then Nsight Systems kernels/copies."
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
            *worker_mode_args,
        ]
        return_code = stream_subprocess(torch_command, env=os.environ.copy())
        if return_code != 0:
            raise RuntimeError(
                f"PyTorch operator worker exited with code {return_code}."
            )
        validate_worker(
            torch_dir,
            "PyTorch operator",
            smoke,
            one_epoch,
        )
        print()

        print("PHASE 2/2: NSIGHT SYSTEMS CUDA/NVTX TRACING")
        print("-" * 88)
        nsys_command = [
            str(NSYS),
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=true",
            "--output",
            str(report_prefix),
            sys.executable,
            str(PROFILE_SCRIPT),
            "--internal-worker",
            "nsight_systems",
            "--internal-output",
            str(nsys_worker_dir),
            *worker_mode_args,
        ]
        return_code = stream_subprocess(nsys_command, env=os.environ.copy())
        if return_code != 0:
            raise RuntimeError(f"Nsight Systems exited with code {return_code}.")
        validate_worker(
            nsys_worker_dir,
            "Nsight Systems",
            smoke,
            one_epoch,
        )
        if not report_path.is_file():
            raise RuntimeError(
                f"Nsight Systems report was not created: {report_path}"
            )
        print()

        print("EXPORTING NSIGHT SYSTEMS SQLITE")
        print("-" * 88)
        export_command = [
            str(NSYS),
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            "--output",
            str(sqlite_path),
            str(report_path),
        ]
        return_code = stream_subprocess(export_command, env=os.environ.copy())
        if return_code != 0:
            raise RuntimeError(
                f"Nsight Systems SQLite export exited with code {return_code}."
            )
        if not sqlite_path.is_file():
            alternative = Path(str(sqlite_path) + ".sqlite")
            if alternative.is_file():
                sqlite_path = alternative
            else:
                raise RuntimeError(
                    f"Nsight Systems SQLite export was not created: {sqlite_path}"
                )
        print()

        print("PARSING NSIGHT SYSTEMS KERNEL/COPY DATA")
        print("-" * 88)
        nsys_metrics = parse_nsys_sqlite(sqlite_path, output_dir)
        print(
            f"Training kernels parsed: {nsys_metrics['total_kernel_launches']} "
            f"launches across {nsys_metrics['unique_kernel_names']} unique names."
        )
        print(
            "GPU kernel-idle time: "
            f"{nsys_metrics['gpu_kernel_idle_s']:.6f} s "
            f"({nsys_metrics['gpu_kernel_idle_percent']:.3f}%)."
        )
        print()

        primary_files = (
            "config.json",
            "epoch_metrics.csv",
            "telemetry.csv",
            "nvml_api_errors.json",
            "nvml_error_log.txt",
            "summary.csv",
            "summary.txt",
            "pytorch_all_profile_events.csv",
            "pytorch_all_operators.csv",
            "pytorch_top10_operators.csv",
            "pytorch_profiler_full_table.txt",
            "pytorch_trace_representative_batch.json",
        )
        for name in primary_files:
            copy_file(torch_dir / name, output_dir / name)

        copy_file(
            torch_dir / "training_output.log",
            output_dir / "pytorch_operator_training_output.log",
        )
        copy_file(
            nsys_worker_dir / "training_output.log",
            output_dir / "nsight_systems_worker_training_output.log",
        )
        copy_file(
            nsys_worker_dir / "summary.csv",
            output_dir / "nsight_systems_worker_summary.csv",
        )
        copy_file(
            nsys_worker_dir / "summary.txt",
            output_dir / "nsight_systems_worker_summary.txt",
        )
        copy_file(
            nsys_worker_dir / "config.json",
            output_dir / "nsight_systems_worker_config.json",
        )
        copy_file(
            nsys_worker_dir / "telemetry.csv",
            output_dir / "nsight_systems_worker_telemetry.csv",
        )

        top_operators = pd.read_csv(output_dir / "pytorch_top10_operators.csv")
        top_kernels = pd.read_csv(output_dir / "nvidia_top5_kernels.csv")
        if top_operators.empty:
            raise RuntimeError("Top PyTorch operator table is empty.")
        if top_kernels.empty:
            raise RuntimeError("Top NVIDIA kernel table is empty.")

        torch_summary = pd.read_csv(output_dir / "summary.csv").iloc[0]
        nsys_summary = pd.read_csv(
            output_dir / "nsight_systems_worker_summary.csv"
        ).iloc[0]
        if torch_summary["initial_model_sha256"] != nsys_summary["initial_model_sha256"]:
            raise RuntimeError("The two profiling passes used different initial models.")
        if torch_summary["training_order_sha256"] != nsys_summary["training_order_sha256"]:
            raise RuntimeError("The two profiling passes used different batch orders.")

        consolidated = {
            "mode": run_mode,
            "primary_profiled_performance_source": "pytorch_operator_pass",
            "profiled_throughput_samples_s": float(
                torch_summary["throughput_samples_s"]
            ),
            "profiled_batch_latency_ms_batch": float(
                torch_summary["batch_latency_ms_batch"]
            ),
            "profiled_total_training_time_s": float(
                torch_summary["total_training_time_s"]
            ),
            "profiled_average_epoch_time_s": float(
                torch_summary["average_epoch_time_s"]
            ),
            "profiled_average_gpu_power_w": float(
                torch_summary["average_gpu_power_w"]
            ),
            "profiled_performance_per_watt_samples_s_w": float(
                torch_summary["performance_per_watt_samples_s_w"]
            ),
            "profiled_average_gpu_util_percent": float(
                torch_summary["average_gpu_util_percent"]
            ),
            "profiled_average_vram_usage_mb": float(
                torch_summary["average_vram_usage_mb"]
            ),
            "profiled_peak_vram_usage_mb": float(
                torch_summary["peak_vram_usage_mb"]
            ),
            "profiled_average_gpu_temperature_c": float(
                torch_summary["average_gpu_temperature_c"]
            ),
            "profiled_final_validation_score": float(
                torch_summary["final_validation_score"]
            ),
            "top_pytorch_operator": str(top_operators.iloc[0]["operator"]),
            "top_nvidia_kernel": str(top_kernels.iloc[0]["kernel_name"]),
            **nsys_metrics,
            "pytorch_operator_pass_initial_model_sha256": str(
                torch_summary["initial_model_sha256"]
            ),
            "nsight_systems_pass_initial_model_sha256": str(
                nsys_summary["initial_model_sha256"]
            ),
            "pytorch_operator_pass_training_order_sha256": str(
                torch_summary["training_order_sha256"]
            ),
            "nsight_systems_pass_training_order_sha256": str(
                nsys_summary["training_order_sha256"]
            ),
        }
        pd.DataFrame([consolidated]).to_csv(
            output_dir / "profiled_consolidated_summary.csv",
            index=False,
        )
        with (output_dir / "profiled_consolidated_summary.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(consolidated, file, indent=2)

        run_valid = {
            "status": "VALID",
            "timestamp": datetime.now().astimezone().isoformat(),
            "mode": run_mode,
            "primary_metrics_source": "pytorch_operator_pass",
            "pytorch_operator_pass_valid": True,
            "nsight_systems_pass_valid": True,
            "training_window_filter_applied": nsys_metrics[
                "training_window_filter_applied"
            ],
            "initial_model_fingerprints_match": True,
            "training_order_fingerprints_match": True,
            "top_operator": str(top_operators.iloc[0]["operator"]),
            "top_kernel": str(top_kernels.iloc[0]["kernel_name"]),
            "total_kernel_launches": nsys_metrics["total_kernel_launches"],
            "gpu_kernel_idle_s": nsys_metrics["gpu_kernel_idle_s"],
            "gpu_kernel_idle_percent": nsys_metrics["gpu_kernel_idle_percent"],
            "regular_script_sha256": sha256_file(REGULAR_SCRIPT),
            "profile_script_sha256": sha256_file(PROFILE_SCRIPT),
        }
        with (output_dir / "RUN_VALID.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(run_valid, file, indent=2)

        print("=" * 88)
        print("PROFILED WORKFLOW VALID")
        print("=" * 88)
        print(
            "Primary performance metrics: PyTorch autograd CUDA-event "
            "operator profiler pass"
        )
        print(f"Top PyTorch operator: {run_valid['top_operator']}")
        print(f"Top NVIDIA kernel: {run_valid['top_kernel']}")
        print(f"Total kernel launches: {run_valid['total_kernel_launches']}")
        print(
            "GPU kernel idle: "
            f"{run_valid['gpu_kernel_idle_s']:.6f} s "
            f"({run_valid['gpu_kernel_idle_percent']:.3f}%)"
        )
        print(f"Results saved to: {output_dir}")

    except Exception:
        traceback_text = traceback.format_exc()
        print("\nPROFILED WORKFLOW FAILED")
        print(traceback_text)
        if is_gpu_related_error(traceback_text):
            append_gpu_error_history(
                (
                    "profiled smoke test"
                    if smoke
                    else "profiled one-epoch test"
                    if one_epoch
                    else "official profiled run"
                ),
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
            "Automated ResNet-50 NVIDIA profiling workflow using "
            "PyTorch autograd CUDA-event operator profiling and NVIDIA "
            "Nsight Systems"
        )
    )
    public = parser.add_mutually_exclusive_group()
    public.add_argument(
        "--official",
        action="store_true",
        help="Clear resnetprofiledrun and execute the full two-pass workflow.",
    )
    public.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Execute a disposable two-pass five-batch profiling smoke test "
            "outside the official workspace."
        ),
    )
    public.add_argument(
        "--one-epoch-test",
        action="store_true",
        help=(
            "Execute a disposable two-pass full-dataset one-epoch profiling "
            "test outside the official workspace."
        ),
    )

    parser.add_argument(
        "--internal-worker",
        choices=("torch_operator", "nsight_systems"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--internal-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--internal-smoke",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--internal-one-epoch",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()

    if args.internal_worker:
        if args.internal_output is None:
            parser.error("--internal-output is required for internal workers")
        if args.internal_smoke and args.internal_one_epoch:
            parser.error(
                "Internal worker cannot be both smoke and one-epoch mode"
            )
        run_worker(
            profile_kind=args.internal_worker,
            output_dir=args.internal_output,
            smoke=bool(args.internal_smoke),
            one_epoch=bool(args.internal_one_epoch),
        )
        return

    if not args.official and not args.smoke and not args.one_epoch_test:
        parser.error("Choose --smoke, --one-epoch-test, or --official")

    run_parent(
        smoke=bool(args.smoke),
        one_epoch=bool(args.one_epoch_test),
    )


if __name__ == "__main__":
    main()

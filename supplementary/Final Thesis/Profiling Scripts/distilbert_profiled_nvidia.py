#!/usr/bin/env python3

import argparse
import gc
import hashlib
import heapq
import importlib.util
import json
import math
import os
import platform
import shutil
import sqlite3
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.autograd.profiler import profile as autograd_profile
from transformers import AutoModelForSequenceClassification


PROJECT_DIR = Path.home() / "Thesis-Testing"
SCRIPT_DIR = PROJECT_DIR / "scripts"
REGULAR_SCRIPT = SCRIPT_DIR / "distilbert_regular_nvidia.py"
PROFILE_SCRIPT = SCRIPT_DIR / "distilbert_profiled_nvidia.py"
PROFILE_DIR = PROJECT_DIR / "Logs" / "distilbertprofiledrun"
TEMP_ROOT = Path.home() / "Downloads" / "Thesis-Testing-Temporary" / "DistilBERT"
TEMP_PROFILE_DIR = TEMP_ROOT / "profile_smoke"
ONE_EPOCH_PROFILE_DIR = TEMP_ROOT / "profile_one_epoch"
NSYS = Path("/usr/local/bin/nsys")
TRAINING_RANGE_NAME = "DISTILBERT_SST2_TRAINING"
EXPECTED_NVIDIA_DRIVER = "580.167.08"
SCRIPT_REVISION = "2026-06-22 nvidia-distilbert-profiled-v1 representative-window-three-pass"
SQL_CHUNK_ROWS = 250_000
TIMELINE_BATCHES = 4

PYTORCH_PROFILE_WAIT_STEPS_OFFICIAL = 10
PYTORCH_PROFILE_WARMUP_STEPS_OFFICIAL = 10
PYTORCH_PROFILE_ACTIVE_STEPS_OFFICIAL = 100
PYTORCH_PROFILE_WAIT_STEPS_SMOKE = 0
PYTORCH_PROFILE_WARMUP_STEPS_SMOKE = 0
PYTORCH_PROFILE_ACTIVE_STEPS_SMOKE = 4

EXPECTED_CONFIG = {
    "SEED": 55,
    "BATCH_SIZE": 64,
    "EPOCHS": 10,
    "MAX_LENGTH": 128,
    "LEARNING_RATE": 5e-5,
    "NUM_WORKERS": 4,
    "SAMPLE_INTERVAL_S": 0.2,
    "AMP_ENABLED": True,
    "EXPECTED_TRAIN_SAMPLES": 67_349,
    "EXPECTED_VALIDATION_SAMPLES": 872,
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_regular_module():
    if not REGULAR_SCRIPT.is_file():
        raise RuntimeError(f"Required regular script is missing: {REGULAR_SCRIPT}")
    spec = importlib.util.spec_from_file_location(
        "distilbert_regular_nvidia_shared", REGULAR_SCRIPT
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
    if getattr(module, "EXPECTED_DRIVER_VERSION", None) != EXPECTED_NVIDIA_DRIVER:
        mismatches.append(
            "EXPECTED_DRIVER_VERSION: expected "
            f"{EXPECTED_NVIDIA_DRIVER!r}, found "
            f"{getattr(module, 'EXPECTED_DRIVER_VERSION', None)!r}"
        )
    tokenized_name = Path(module.TOKENIZED_DATASET_DIR).name
    if tokenized_name != "tokenized_distilbert_dynamic_maxlen128":
        mismatches.append(
            "TOKENIZED_DATASET_DIR must point to the dynamic-padding dataset; "
            f"found {module.TOKENIZED_DATASET_DIR}"
        )
    if not hasattr(module, "DynamicPaddingCollator"):
        mismatches.append("DynamicPaddingCollator is missing from the regular script")
    if mismatches:
        raise RuntimeError(
            "The profiling and regular scripts no longer use the same fixed "
            "workload/environment:\n" + "\n".join(mismatches)
        )

    required_paths = {
        "MODEL_DIR": module.MODEL_DIR,
        "TOKENIZED_DATASET_DIR": module.TOKENIZED_DATASET_DIR,
    }
    missing = [
        f"{name}: {path}" for name, path in required_paths.items()
        if not Path(path).exists()
    ]
    if missing:
        raise RuntimeError("Required DistilBERT assets are missing:\n" + "\n".join(missing))
    return module


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


def link_or_copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"Expected output is missing: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def is_pytorch_operator(name: str) -> bool:
    return name.startswith("aten::")


def operator_note(name: str) -> str:
    exact = {
        "aten::mm": "Matrix multiplication",
        "aten::bmm": "Batched matrix multiplication",
        "aten::addmm": "Matrix multiplication with bias addition",
        "aten::linear": "Linear layer",
        "aten::layer_norm": "Layer normalization",
        "aten::native_layer_norm": "Layer normalization kernel",
        "aten::native_layer_norm_backward": "Layer normalization backward",
        "aten::_softmax": "Softmax",
        "aten::_softmax_backward_data": "Softmax backward",
        "aten::gelu": "GELU activation",
        "aten::gelu_backward": "GELU activation backward",
        "aten::copy_": "Tensor copy",
        "aten::add_": "In-place tensor addition",
        "aten::_foreach_add_": "In-place multi-tensor addition",
        "aten::_foreach_mul_": "In-place multi-tensor multiplication",
    }
    if name in exact:
        return exact[name]
    lowered = name.lower()
    if "layer_norm_backward" in lowered:
        return "Layer normalization backward"
    if "layer_norm" in lowered:
        return "Layer normalization"
    if "softmax_backward" in lowered:
        return "Softmax backward"
    if "softmax" in lowered:
        return "Softmax"
    if "gelu_backward" in lowered:
        return "GELU activation backward"
    if "gelu" in lowered:
        return "GELU activation"
    if "bmm" in lowered:
        return "Batched matrix multiplication"
    if "addmm" in lowered:
        return "Matrix multiplication with bias addition"
    if "mm" in lowered or "matmul" in lowered:
        return "Matrix multiplication"
    if "copy" in lowered:
        return "Tensor copy"
    if "adam" in lowered:
        return "Adam optimizer update"
    if "dropout" in lowered:
        return "Dropout"
    return ""


def event_device_times(event):
    self_us = getattr(event, "self_device_time_total", None)
    total_us = getattr(event, "device_time_total", None)
    if self_us is None:
        self_us = getattr(event, "self_cuda_time_total", 0.0)
    if total_us is None:
        total_us = getattr(event, "cuda_time_total", 0.0)
    return float(self_us or 0.0), float(total_us or 0.0)


def aggregate_profiler_events(profiler, accumulator: dict) -> None:
    for event in profiler.key_averages():
        name = str(event.key)
        if not is_pytorch_operator(name):
            continue
        self_us, total_us = event_device_times(event)
        if self_us <= 0 and total_us <= 0:
            continue
        row = accumulator.setdefault(
            name,
            {
                "operator": name,
                "self_gpu_time_ms": 0.0,
                "gpu_total_ms": 0.0,
                "calls": 0,
                "notes": operator_note(name),
            },
        )
        row["self_gpu_time_ms"] += self_us / 1000.0
        row["gpu_total_ms"] += total_us / 1000.0
        row["calls"] += int(event.count)


def save_operator_outputs(accumulator: dict, output_dir: Path) -> pd.DataFrame:
    rows = list(accumulator.values())
    rows.sort(key=lambda row: row["self_gpu_time_ms"], reverse=True)
    total_self_ms = sum(row["self_gpu_time_ms"] for row in rows)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["self_gpu_percent"] = (
            row["self_gpu_time_ms"] / total_self_ms * 100.0
            if total_self_ms > 0 else float("nan")
        )
    columns = [
        "rank", "operator", "self_gpu_time_ms", "self_gpu_percent",
        "gpu_total_ms", "calls", "notes",
    ]
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(output_dir / "pytorch_all_operators.csv", index=False)
    frame.head(10).to_csv(output_dir / "pytorch_top10_operators.csv", index=False)
    with (output_dir / "pytorch_profiler_full_table.txt").open("w", encoding="utf-8") as file:
        file.write(frame.to_string(index=False))
        file.write("\n")
    return frame


class LegacyRepresentativeWindowProfiler:
    """Capture one deterministic 100-batch CUDA-event operator window."""

    def __init__(self, output_dir: Path, wait_steps: int, warmup_steps: int, active_steps: int):
        self.output_dir = output_dir
        self.wait_steps = int(wait_steps)
        self.warmup_steps = int(warmup_steps)
        self.active_steps = int(active_steps)
        self.active_start = self.wait_steps + self.warmup_steps
        self.active_end = self.active_start + self.active_steps
        self.total_batches_seen = 0
        self.profiler = None
        self.completed = False
        self.accumulator = {}
        self.aggregation_overhead_s = 0.0

    def before_batch(self) -> None:
        if self.total_batches_seen == self.active_start:
            if self.profiler is not None or self.completed:
                raise RuntimeError("Representative profiler window started more than once.")
            torch.cuda.synchronize()
            self.profiler = autograd_profile(
                use_device="cuda",
                use_kineto=False,
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            )
            self.profiler.__enter__()
            print(
                "Legacy CUDA-event representative window started at training "
                f"batch {self.total_batches_seen + 1}."
            )

    def after_batch(self) -> None:
        self.total_batches_seen += 1
        if self.total_batches_seen == self.active_end:
            if self.profiler is None:
                raise RuntimeError("Representative profiler window was not active at its end.")
            torch.cuda.synchronize()
            self.profiler.__exit__(None, None, None)
            aggregation_start = time.perf_counter()
            aggregate_profiler_events(self.profiler, self.accumulator)
            self.aggregation_overhead_s += time.perf_counter() - aggregation_start
            self.profiler = None
            self.completed = True
            gc.collect()
            print(
                "Legacy CUDA-event representative window completed; "
                f"active={self.active_steps} batches."
            )

    def finalize(self, total_training_batches: int) -> pd.DataFrame:
        if self.profiler is not None:
            self.profiler.__exit__(None, None, None)
            self.profiler = None
        if not self.completed:
            raise RuntimeError(
                "Representative profiler window did not complete. "
                f"Batches seen={self.total_batches_seen}, required={self.active_end}."
            )
        frame = save_operator_outputs(self.accumulator, self.output_dir)
        metadata = {
            "collection_mode": "representative_window_during_full_training",
            "backend": "torch.autograd.profiler legacy CUDA events (Kineto disabled)",
            "wait_batches": self.wait_steps,
            "warmup_batches": self.warmup_steps,
            "active_profiled_batches": self.active_steps,
            "windows_completed": 1,
            "total_profiled_batches": self.active_steps,
            "total_training_batches_executed": int(total_training_batches),
            "operator_scope": "ATen operators only",
            "operator_aggregation_overhead_s": self.aggregation_overhead_s,
            "timeline_source": "separate four-batch Nsight Systems report",
        }
        with (self.output_dir / "pytorch_operator_window_metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        pd.DataFrame([{
            "window": 1,
            "wait_batches": self.wait_steps,
            "warmup_batches": self.warmup_steps,
            "profiled_batches": self.active_steps,
            "training_batches_seen_at_window_end": self.active_end,
            "unique_operators": len(frame),
            "operator_aggregation_overhead_s": self.aggregation_overhead_s,
        }]).to_csv(self.output_dir / "pytorch_operator_window_progress.csv", index=False)
        return frame


def run_worker(
    profile_kind: str,
    output_dir: Path,
    smoke: bool,
    one_epoch: bool,
    timeline: bool,
) -> None:
    regular = load_regular_module()
    reset_directory(output_dir)

    log_file = (output_dir / "training_output.log").open("w", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    monitor = None
    window_profiler = None
    nvtx_active = False

    try:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("WANDB_DISABLED", "true")
        regular.set_seed(regular.SEED)

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot detect the NVIDIA GPU.")
        if torch.version.cuda is None:
            raise RuntimeError("This PyTorch build does not report a CUDA runtime.")

        gpu_name = torch.cuda.get_device_name(0)
        gpu_arch = ".".join(str(part) for part in torch.cuda.get_device_capability(0))
        driver_version = regular.read_nvidia_driver_version()
        if driver_version != EXPECTED_NVIDIA_DRIVER:
            raise RuntimeError(
                f"Unexpected NVIDIA driver: {driver_version}. "
                f"Expected {EXPECTED_NVIDIA_DRIVER}."
            )
        if "5060 Ti" not in gpu_name:
            raise RuntimeError(f"Unexpected GPU selected: {gpu_name}. Expected RTX 5060 Ti.")

        use_smoke_dataset = smoke or timeline
        dataset = regular.verify_assets()
        train_loader, validation_loader = regular.build_dataloaders(
            dataset, smoke=use_smoke_dataset
        )
        epochs_to_run = 1 if (smoke or one_epoch or timeline) else regular.EPOCHS
        train_samples_per_epoch = len(train_loader.dataset)
        validation_samples = len(validation_loader.dataset)

        if profile_kind == "torch_operator":
            if smoke:
                wait_steps = PYTORCH_PROFILE_WAIT_STEPS_SMOKE
                warmup_steps = PYTORCH_PROFILE_WARMUP_STEPS_SMOKE
                active_steps = PYTORCH_PROFILE_ACTIVE_STEPS_SMOKE
            else:
                wait_steps = PYTORCH_PROFILE_WAIT_STEPS_OFFICIAL
                warmup_steps = PYTORCH_PROFILE_WARMUP_STEPS_OFFICIAL
                active_steps = PYTORCH_PROFILE_ACTIVE_STEPS_OFFICIAL
            window_profiler = LegacyRepresentativeWindowProfiler(
                output_dir, wait_steps, warmup_steps, active_steps
            )
        elif profile_kind != "nsight_systems":
            raise RuntimeError(f"Unknown profile kind: {profile_kind}")

        mode_name = (
            "timeline" if timeline else
            "smoke" if smoke else
            "one_epoch" if one_epoch else
            "official_profiled"
        )
        print("=" * 88)
        print(f"DistilBERT SST-2 NVIDIA {profile_kind} profiling ({mode_name})")
        print("=" * 88)
        print(f"Timestamp: {datetime.now().astimezone().isoformat()}")
        print(f"Output directory: {output_dir}")
        print(f"Model directory: {regular.MODEL_DIR}")
        print(f"Dataset directory: {regular.TOKENIZED_DATASET_DIR}")
        print(f"Python: {platform.python_version()}")
        print(f"Platform: {platform.platform()}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"NVIDIA driver: {driver_version}")
        print(f"Transformers: {regular.package_version('transformers')}")
        print(f"Datasets: {regular.package_version('datasets')}")
        print(f"GPU: {gpu_name}")
        print(f"GPU compute capability: {gpu_arch}")
        print()

        config = {
            "mode": mode_name,
            "profile_kind": profile_kind,
            "seed": regular.SEED,
            "model": "DistilBERT base uncased for sequence classification",
            "model_source": regular.MODEL_SOURCE,
            "model_path": str(regular.MODEL_DIR),
            "dataset": "GLUE SST-2",
            "dataset_source": f"{regular.DATASET_SOURCE}/{regular.DATASET_CONFIG}",
            "dataset_path": str(regular.TOKENIZED_DATASET_DIR),
            "training_samples": train_samples_per_epoch,
            "validation_samples": validation_samples,
            "batch_size": regular.BATCH_SIZE,
            "max_length": regular.MAX_LENGTH,
            "padding": "dynamic_per_batch",
            "epochs": epochs_to_run,
            "official_epochs": regular.EPOCHS,
            "optimizer": "torch.optim.Adam",
            "learning_rate": regular.LEARNING_RATE,
            "weight_decay": 0.0,
            "precision": "AMP ON / FP16 autocast",
            "amp_enabled": regular.AMP_ENABLED,
            "workers": regular.NUM_WORKERS,
            "telemetry_interval_s": regular.SAMPLE_INTERVAL_S,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "nvidia_driver": driver_version,
            "transformers_version": regular.package_version("transformers"),
            "datasets_version": regular.package_version("datasets"),
            "gpu": gpu_name,
            "gpu_compute_capability": gpu_arch,
            "training_range_name": TRAINING_RANGE_NAME,
            "pytorch_profile_collection": "single_representative_window",
            "pytorch_profile_backend": "legacy_cuda_events_kineto_disabled",
            "pytorch_profile_wait_steps": (
                PYTORCH_PROFILE_WAIT_STEPS_SMOKE if smoke else PYTORCH_PROFILE_WAIT_STEPS_OFFICIAL
            ),
            "pytorch_profile_warmup_steps": (
                PYTORCH_PROFILE_WARMUP_STEPS_SMOKE if smoke else PYTORCH_PROFILE_WARMUP_STEPS_OFFICIAL
            ),
            "pytorch_profile_active_steps": (
                PYTORCH_PROFILE_ACTIVE_STEPS_SMOKE if smoke else PYTORCH_PROFILE_ACTIVE_STEPS_OFFICIAL
            ),
            "timeline_batches": TIMELINE_BATCHES if timeline else None,
            "script_revision": SCRIPT_REVISION,
        }
        with (output_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

        device = torch.device("cuda:0")
        regular.set_seed(regular.SEED)
        model = AutoModelForSequenceClassification.from_pretrained(
            regular.MODEL_DIR, local_files_only=True
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=regular.LEARNING_RATE)
        scaler = torch.amp.GradScaler("cuda", enabled=regular.AMP_ENABLED)
        if not scaler.is_enabled():
            raise RuntimeError("AMP was requested but GradScaler is disabled.")

        print("Training configuration verification")
        print(f"Batch size: {regular.BATCH_SIZE}")
        print(f"Workers: {regular.NUM_WORKERS}")
        print(f"Epochs: {epochs_to_run}")
        print("Optimizer: Adam")
        print(f"Learning rate: {regular.LEARNING_RATE}")
        print(f"Maximum sequence length: {regular.MAX_LENGTH}")
        print("Padding: dynamic per batch (no pad-to-multiple)")
        print(f"AMP enabled: {scaler.is_enabled()}")
        print(f"Model parameter dtype: {next(model.parameters()).dtype}")
        print()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        monitor = regular.TelemetryMonitor()
        monitor.start()
        if profile_kind == "nsight_systems":
            torch.cuda.nvtx.range_push(TRAINING_RANGE_NAME)
            nvtx_active = True
            print(f"NVTX training range started: {TRAINING_RANGE_NAME}")
        total_start = time.perf_counter()

        epoch_rows = []
        total_batches = 0
        total_samples_processed = 0
        total_valid_tokens = 0
        total_padded_tokens = 0
        stop_after_batches = TIMELINE_BATCHES if timeline else None
        stop_training = False

        for epoch_index in range(epochs_to_run):
            model.train()
            epoch_start = time.perf_counter()
            epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
            epoch_samples = 0
            epoch_batches = 0
            epoch_valid_tokens = 0
            epoch_padded_tokens = 0

            for batch in train_loader:
                if window_profiler is not None:
                    window_profiler.before_batch()

                valid_tokens = int(batch.pop("_valid_tokens"))
                padded_tokens = int(batch.pop("_padded_tokens"))
                batch_size = int(batch["labels"].shape[0])
                batch = regular.move_batch(batch, device)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda", dtype=torch.float16,
                    enabled=regular.AMP_ENABLED,
                ):
                    outputs = model(**batch)
                    loss = outputs.loss
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if window_profiler is not None:
                    window_profiler.after_batch()

                epoch_loss_sum += loss.detach().double() * batch_size
                epoch_samples += batch_size
                epoch_batches += 1
                total_batches += 1
                total_samples_processed += batch_size
                epoch_valid_tokens += valid_tokens
                epoch_padded_tokens += padded_tokens
                total_valid_tokens += valid_tokens
                total_padded_tokens += padded_tokens

                if stop_after_batches is not None and total_batches >= stop_after_batches:
                    stop_training = True
                    break

            torch.cuda.synchronize()
            epoch_end = time.perf_counter()
            epoch_time = epoch_end - epoch_start
            average_loss = float((epoch_loss_sum / epoch_samples).item())
            epoch_rows.append({
                "epoch": epoch_index + 1,
                "epoch_time_s": epoch_time,
                "throughput_samples_s": epoch_samples / epoch_time,
                "samples": epoch_samples,
                "batches": epoch_batches,
                "average_training_loss": average_loss,
                "valid_tokens": epoch_valid_tokens,
                "padded_tokens": epoch_padded_tokens,
                "valid_tokens_s": epoch_valid_tokens / epoch_time,
                "padded_tokens_s": epoch_padded_tokens / epoch_time,
                "average_valid_tokens_per_sample": epoch_valid_tokens / epoch_samples,
                "average_padded_sequence_length": epoch_padded_tokens / epoch_samples,
                "padding_efficiency_percent": epoch_valid_tokens / epoch_padded_tokens * 100.0,
            })
            print(
                f"Epoch {epoch_index + 1}/{epochs_to_run} | "
                f"Time: {epoch_time:.2f}s | "
                f"Throughput: {epoch_samples / epoch_time:.2f} samples/s | "
                f"Batches: {epoch_batches} | Loss: {average_loss:.6f}"
            )
            if stop_training:
                break

        torch.cuda.synchronize()
        total_training_time = time.perf_counter() - total_start
        if nvtx_active:
            torch.cuda.nvtx.range_pop()
            nvtx_active = False
            print(f"NVTX training range ended: {TRAINING_RANGE_NAME}")
        monitor.stop()

        operator_aggregation_overhead_s = 0.0
        if window_profiler is not None:
            operator_frame = window_profiler.finalize(total_batches)
            operator_aggregation_overhead_s = window_profiler.aggregation_overhead_s
            if operator_frame.empty:
                raise RuntimeError("Legacy CUDA-event profiler produced no ATen GPU operator rows.")

        telemetry_df = monitor.dataframe()
        telemetry_df.to_csv(output_dir / "telemetry.csv", index=False)
        with (output_dir / "nvml_api_errors.json").open("w", encoding="utf-8") as file:
            json.dump(monitor.api_errors, file, indent=2)
        regular.validate_telemetry(telemetry_df)

        epoch_df = pd.DataFrame(epoch_rows)
        epoch_df.to_csv(output_dir / "epoch_metrics.csv", index=False)
        validation = regular.validate_model(model, validation_loader, device)

        throughput = total_samples_processed / total_training_time
        valid_tokens_per_second = total_valid_tokens / total_training_time
        padded_tokens_per_second = total_padded_tokens / total_training_time
        average_valid_tokens_per_sample = total_valid_tokens / total_samples_processed
        average_padded_sequence_length = total_padded_tokens / total_samples_processed
        padding_efficiency_percent = total_valid_tokens / total_padded_tokens * 100.0
        batch_latency_ms = total_training_time / total_batches * 1000.0
        average_epoch_time = float(epoch_df["epoch_time_s"].mean())
        average_power_w = regular.series_stat(telemetry_df, "power_w", "mean")
        average_gpu_util = regular.series_stat(telemetry_df, "gpu_util_percent", "mean")
        average_vram_mb = regular.series_stat(telemetry_df, "vram_used_mb", "mean")
        peak_vram_mb = regular.series_stat(telemetry_df, "vram_used_mb", "max")
        average_gpu_temp_c = regular.series_stat(telemetry_df, "gpu_temperature_c", "mean")
        peak_gpu_temp_c = regular.series_stat(telemetry_df, "gpu_temperature_c", "max")
        torch_peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
        performance_per_watt = throughput / average_power_w
        peak_system_ram_mb = regular.series_stat(telemetry_df, "system_ram_used_mb", "max")
        average_system_ram_mb = regular.series_stat(telemetry_df, "system_ram_used_mb", "mean")
        average_cpu_util = regular.series_stat(telemetry_df, "cpu_util_percent", "mean")
        average_cpu_temp_c = regular.series_stat(telemetry_df, "cpu_package_temperature_c", "mean")
        average_disk_read_mb_s = regular.series_stat(telemetry_df, "disk_read_mb_s", "mean")
        average_disk_write_mb_s = regular.series_stat(telemetry_df, "disk_write_mb_s", "mean")
        peak_disk_read_mb_s = regular.series_stat(telemetry_df, "disk_read_mb_s", "max")
        peak_disk_write_mb_s = regular.series_stat(telemetry_df, "disk_write_mb_s", "max")

        expected_samples = (
            TIMELINE_BATCHES * regular.BATCH_SIZE if timeline
            else train_samples_per_epoch * epochs_to_run
        )
        expected_batches = TIMELINE_BATCHES if timeline else len(train_loader) * epochs_to_run
        if total_samples_processed != expected_samples:
            raise RuntimeError(
                f"Processed {total_samples_processed} samples; expected {expected_samples}."
            )
        if total_batches != expected_batches:
            raise RuntimeError(f"Processed {total_batches} batches; expected {expected_batches}.")
        if not all(math.isfinite(row["average_training_loss"]) for row in epoch_rows):
            raise RuntimeError("One or more epoch losses are non-finite.")
        if validation["samples"] != validation_samples:
            raise RuntimeError(
                f"Validated {validation['samples']} samples; expected {validation_samples}."
            )

        temperature_limit_seen = (
            not math.isnan(peak_gpu_temp_c) and peak_gpu_temp_c >= 95.0
        )
        stability_notes = (
            "Completed successfully; no Python, CUDA, cuDNN, Transformers, "
            "Datasets, NVML, GPU, OOM, profiler, or thermal fatal errors detected."
        )
        if temperature_limit_seen:
            stability_notes += " A monitored GPU temperature reached the configured critical threshold."
        else:
            stability_notes += " No monitored GPU temperature reached the configured critical threshold."
        if monitor.api_errors:
            stability_notes += f" NVML had {monitor.api_error_count} recoverable API errors."
        else:
            stability_notes += " All required NVML telemetry queries succeeded."
        if profile_kind == "torch_operator":
            stability_notes += (
                " The operator pass used one legacy CUDA-event representative window "
                f"after {config['pytorch_profile_wait_steps']} wait and "
                f"{config['pytorch_profile_warmup_steps']} warm-up batches."
            )

        summary = {
            "mode": mode_name,
            "profile_kind": profile_kind,
            "throughput_samples_s": throughput,
            "valid_tokens_s": valid_tokens_per_second,
            "padded_tokens_s": padded_tokens_per_second,
            "average_valid_tokens_per_sample": average_valid_tokens_per_sample,
            "average_padded_sequence_length": average_padded_sequence_length,
            "padding_efficiency_percent": padding_efficiency_percent,
            "batch_latency_ms_batch": batch_latency_ms,
            "total_training_time_s": total_training_time,
            "average_epoch_time_s": average_epoch_time,
            "operator_aggregation_overhead_s": operator_aggregation_overhead_s,
            "average_gpu_power_w": average_power_w,
            "performance_per_watt_samples_s_w": performance_per_watt,
            "average_gpu_util_percent": average_gpu_util,
            "average_vram_usage_mb": average_vram_mb,
            "peak_vram_usage_mb": peak_vram_mb,
            "torch_peak_allocated_memory_mb": torch_peak_allocated_mb,
            "average_gpu_temperature_c": average_gpu_temp_c,
            "peak_gpu_temperature_c": peak_gpu_temp_c,
            "thermal_throttle_seen": monitor.thermal_throttle_seen,
            "temperature_critical_threshold_reached": temperature_limit_seen,
            "final_validation_accuracy": validation["accuracy"],
            "final_validation_loss": validation["loss"],
            "final_training_loss": epoch_rows[-1]["average_training_loss"],
            "peak_system_ram_mb": peak_system_ram_mb,
            "average_system_ram_mb": average_system_ram_mb,
            "average_cpu_util_percent": average_cpu_util,
            "average_cpu_package_temperature_c": average_cpu_temp_c,
            "average_disk_read_mb_s": average_disk_read_mb_s,
            "average_disk_write_mb_s": average_disk_write_mb_s,
            "peak_disk_read_mb_s": peak_disk_read_mb_s,
            "peak_disk_write_mb_s": peak_disk_write_mb_s,
            "total_samples_processed": total_samples_processed,
            "total_batches_processed": total_batches,
            "total_valid_tokens": total_valid_tokens,
            "total_padded_tokens": total_padded_tokens,
            "validation_samples": validation["samples"],
            "telemetry_samples": len(telemetry_df),
            "telemetry_interval_s": regular.SAMPLE_INTERVAL_S,
            "workers": regular.NUM_WORKERS,
            "amp_enabled": scaler.is_enabled(),
            "padding": "dynamic_per_batch",
            "model_parameter_dtype": str(next(model.parameters()).dtype),
            "driver_version": driver_version,
            "cuda_runtime": torch.version.cuda,
            "stability_error_notes": stability_notes,
        }
        pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)

        with (output_dir / "summary.txt").open("w", encoding="utf-8") as file:
            file.write(f"DistilBERT SST-2 NVIDIA {profile_kind} profile\n")
            file.write("=" * 72 + "\n")
            file.write(f"Throughput: {regular.format_value(throughput)} samples/s\n")
            file.write(f"Batch Latency: {regular.format_value(batch_latency_ms)} ms/batch\n")
            file.write(f"Total Training Time: {regular.format_value(total_training_time)} s\n")
            file.write(f"Average Epoch Time: {regular.format_value(average_epoch_time)} s\n")
            file.write(f"Average GPU Power Draw: {regular.format_value(average_power_w)} W\n")
            file.write(
                "Performance per Watt: "
                f"{regular.format_value(performance_per_watt, 4)} samples/s/W\n"
            )
            file.write(f"Average GPU Utilization: {regular.format_value(average_gpu_util)} %\n")
            file.write(f"Average VRAM Usage: {regular.format_value(average_vram_mb)} MB\n")
            file.write(f"Peak VRAM Usage: {regular.format_value(peak_vram_mb)} MB\n")
            file.write(f"Average GPU Temperature: {regular.format_value(average_gpu_temp_c)} C\n")
            file.write(f"Final Validation Accuracy: {validation['accuracy']:.4f}\n")
            file.write(f"Peak System RAM: {regular.format_value(peak_system_ram_mb)} MB\n")
            file.write(f"Average System RAM: {regular.format_value(average_system_ram_mb)} MB\n")
            file.write(f"Average CPU Utilization: {regular.format_value(average_cpu_util)} %\n")
            file.write(
                "Average Disk Read / Write: "
                f"{regular.format_value(average_disk_read_mb_s)} / "
                f"{regular.format_value(average_disk_write_mb_s)} MB/s\n"
            )
            file.write(
                "Peak Disk Read / Write: "
                f"{regular.format_value(peak_disk_read_mb_s)} / "
                f"{regular.format_value(peak_disk_write_mb_s)} MB/s\n"
            )
            file.write(f"Stability / Error Notes: {stability_notes}\n")

        worker_valid = {
            "status": "VALID",
            "timestamp": datetime.now().astimezone().isoformat(),
            "mode": mode_name,
            "profile_kind": profile_kind,
            "epochs_completed": len(epoch_rows),
            "total_samples_processed": total_samples_processed,
            "total_batches_processed": total_batches,
            "workers": regular.NUM_WORKERS,
            "amp_enabled": scaler.is_enabled(),
            "padding": "dynamic_per_batch",
            "validation_accuracy": validation["accuracy"],
            "driver_version": driver_version,
        }
        with (output_dir / "WORKER_VALID.json").open("w", encoding="utf-8") as file:
            json.dump(worker_valid, file, indent=2)

        print("=" * 88)
        print("PROFILE WORKER VALID")
        print("=" * 88)
        print(f"Throughput: {throughput:.2f} samples/s")
        print(f"Batch Latency: {batch_latency_ms:.2f} ms/batch")
        print(f"Total Training Time: {total_training_time:.2f} s")
        print(f"Final Validation Accuracy: {validation['accuracy']:.4f}")
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
                partial_df = monitor.dataframe()
                if not partial_df.empty:
                    partial_df.to_csv(output_dir / "telemetry_partial.csv", index=False)
                with (output_dir / "nvml_api_errors.json").open("w", encoding="utf-8") as file:
                    json.dump(monitor.api_errors, file, indent=2)
            except Exception:
                pass
        traceback_text = traceback.format_exc()
        print("\nPROFILE WORKER FAILED")
        print(traceback_text)
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def table_names(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {str(row[0]) for row in rows}


def table_columns(connection: sqlite3.Connection, table: str) -> list[str]:
    rows = connection.execute(
        f"PRAGMA table_info({quote_identifier(table)})"
    ).fetchall()
    return [str(row[1]) for row in rows]


def first_existing(columns, candidates):
    lookup = {str(column).lower(): str(column) for column in columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def load_string_ids(connection: sqlite3.Connection) -> dict:
    tables = table_names(connection)
    if "StringIds" not in tables:
        return {}
    frame = pd.read_sql_query("SELECT id, value FROM StringIds", connection)
    if not {"id", "value"}.issubset(frame.columns):
        return {}
    return dict(zip(frame["id"], frame["value"]))


def find_training_window(connection: sqlite3.Connection, string_ids: dict):
    tables = table_names(connection)
    candidates = [name for name in tables if "NVTX" in name.upper() and "EVENT" in name.upper()]
    if "NVTX_EVENTS" in candidates:
        candidates.remove("NVTX_EVENTS")
        candidates.insert(0, "NVTX_EVENTS")
    diagnostics = []
    for table in candidates:
        columns = table_columns(connection, table)
        start_col = first_existing(columns, ("start", "startNs", "startTime"))
        end_col = first_existing(columns, ("end", "endNs", "endTime"))
        name_col = first_existing(columns, ("text", "message", "name", "textId", "messageId", "nameId"))
        if not start_col or not end_col or not name_col:
            diagnostics.append(f"{table}: missing start/end/name columns")
            continue
        query = (
            f"SELECT {quote_identifier(start_col)} AS s, "
            f"{quote_identifier(end_col)} AS e, "
            f"{quote_identifier(name_col)} AS n FROM {quote_identifier(table)}"
        )
        frame = pd.read_sql_query(query, connection)
        if frame.empty:
            continue
        def resolve(value):
            if value in string_ids:
                return str(string_ids[value])
            return str(value)
        names = frame["n"].map(resolve)
        matches = frame[names.str.contains(TRAINING_RANGE_NAME, regex=False, na=False)].copy()
        if matches.empty:
            diagnostics.append(f"{table}: marker not found")
            continue
        matches["s"] = pd.to_numeric(matches["s"], errors="coerce")
        matches["e"] = pd.to_numeric(matches["e"], errors="coerce")
        matches = matches.dropna(subset=["s", "e"])
        matches = matches[matches["e"] > matches["s"]]
        if matches.empty:
            continue
        matches["duration"] = matches["e"] - matches["s"]
        row = matches.sort_values("duration", ascending=False).iloc[0]
        return int(row["s"]), int(row["e"]), table
    raise RuntimeError(
        f"NVTX range {TRAINING_RANGE_NAME!r} was not found. "
        + "; ".join(diagnostics)
    )


def select_kernel_table(tables: set[str]) -> str:
    for preferred in (
        "CUPTI_ACTIVITY_KIND_KERNEL",
        "CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL",
    ):
        if preferred in tables:
            return preferred
    candidates = [name for name in tables if "CUPTI" in name.upper() and "KERNEL" in name.upper()]
    if not candidates:
        raise RuntimeError("Nsight SQLite export contains no CUDA kernel table.")
    return sorted(candidates)[0]


def select_memcpy_tables(tables: set[str]) -> list[str]:
    preferred = ["CUPTI_ACTIVITY_KIND_MEMCPY", "CUPTI_ACTIVITY_KIND_MEMCPY2"]
    found = [name for name in preferred if name in tables]
    if found:
        return found
    return sorted(name for name in tables if "CUPTI" in name.upper() and "MEMCPY" in name.upper())


def stream_activity_rows(
    connection: sqlite3.Connection,
    table: str,
    window_start: int,
    window_end: int,
    string_ids: dict,
    include_name: bool,
):
    columns = table_columns(connection, table)
    start_col = first_existing(columns, ("start", "startNs", "startTime"))
    end_col = first_existing(columns, ("end", "endNs", "endTime"))
    if not start_col or not end_col:
        raise RuntimeError(f"Table {table} has no start/end columns: {columns}")
    name_col = None
    if include_name:
        name_col = first_existing(
            columns,
            (
                "demangledName",
                "shortName",
                "name",
                "kernelName",
                "nameId",
                "kernelNameId",
            ),
        )
        if not name_col:
            raise RuntimeError(f"Kernel table {table} has no name column: {columns}")

    select_parts = [
        f"{quote_identifier(start_col)} AS s",
        f"{quote_identifier(end_col)} AS e",
    ]
    if name_col:
        select_parts.append(f"{quote_identifier(name_col)} AS n")
    query = (
        f"SELECT {', '.join(select_parts)} FROM {quote_identifier(table)} "
        f"WHERE {quote_identifier(end_col)} > ? AND "
        f"{quote_identifier(start_col)} < ? "
        f"ORDER BY {quote_identifier(start_col)}"
    )
    for chunk in pd.read_sql_query(
        query,
        connection,
        params=(window_start, window_end),
        chunksize=SQL_CHUNK_ROWS,
    ):
        chunk["s"] = pd.to_numeric(chunk["s"], errors="coerce")
        chunk["e"] = pd.to_numeric(chunk["e"], errors="coerce")
        chunk = chunk.dropna(subset=["s", "e"])
        for row in chunk.itertuples(index=False):
            start = max(int(row.s), window_start)
            end = min(int(row.e), window_end)
            if end <= start:
                continue
            if include_name:
                value = row.n
                name = str(string_ids.get(value, value))
                yield start, end, name
            else:
                yield start, end


def union_duration_sorted(intervals) -> int:
    current_start = None
    current_end = None
    total = 0
    for start, end in intervals:
        if current_start is None:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    if current_start is not None:
        total += current_end - current_start
    return int(total)


def parse_nsys_sqlite(sqlite_path: Path, destination: Path) -> dict:
    if not sqlite_path.is_file() or sqlite_path.stat().st_size <= 0:
        raise RuntimeError(f"Nsight SQLite export is missing or empty: {sqlite_path}")

    with sqlite3.connect(sqlite_path) as connection:
        tables = table_names(connection)
        string_ids = load_string_ids(connection)
        training_start, training_end, marker_table = find_training_window(
            connection, string_ids
        )
        training_duration_ns = training_end - training_start
        kernel_table = select_kernel_table(tables)
        memcpy_tables = select_memcpy_tables(tables)

        stats = {}
        kernel_intervals = []
        current_start = None
        current_end = None
        kernel_union_ns = 0
        kernel_rows = 0

        for start, end, name in stream_activity_rows(
            connection,
            kernel_table,
            training_start,
            training_end,
            string_ids,
            include_name=True,
        ):
            duration = end - start
            item = stats.setdefault(
                name,
                {
                    "calls": 0,
                    "total_ns": 0,
                    "sum_sq": 0.0,
                    "min_ns": None,
                    "max_ns": None,
                },
            )
            item["calls"] += 1
            item["total_ns"] += duration
            item["sum_sq"] += float(duration) * float(duration)
            item["min_ns"] = duration if item["min_ns"] is None else min(item["min_ns"], duration)
            item["max_ns"] = duration if item["max_ns"] is None else max(item["max_ns"], duration)
            kernel_rows += 1

            if current_start is None:
                current_start, current_end = start, end
            elif start <= current_end:
                current_end = max(current_end, end)
            else:
                kernel_union_ns += current_end - current_start
                current_start, current_end = start, end
        if current_start is not None:
            kernel_union_ns += current_end - current_start
        if kernel_rows <= 0:
            raise RuntimeError("No CUDA kernels were found inside the training range.")

        total_summed_kernel_ns = sum(item["total_ns"] for item in stats.values())
        rows = []
        for name, item in stats.items():
            calls = item["calls"]
            mean = item["total_ns"] / calls
            variance = (
                (item["sum_sq"] - calls * mean * mean) / (calls - 1)
                if calls > 1
                else 0.0
            )
            rows.append(
                {
                    "kernel_name": name,
                    "calls": calls,
                    "total_duration_ms": item["total_ns"] / 1e6,
                    "average_duration_us": mean / 1e3,
                    "percentage_of_summed_kernel_time": (
                        item["total_ns"] / total_summed_kernel_ns * 100.0
                        if total_summed_kernel_ns > 0
                        else float("nan")
                    ),
                    "minimum_duration_ns": item["min_ns"],
                    "maximum_duration_ns": item["max_ns"],
                    "stddev_duration_ns": math.sqrt(max(variance, 0.0)),
                    "total_duration_ns": item["total_ns"],
                }
            )
        grouped = pd.DataFrame(rows).sort_values("total_duration_ns", ascending=False)
        grouped.insert(0, "rank", range(1, len(grouped) + 1))
        grouped.to_csv(destination / "nsys_parsed_kernel_summary.csv", index=False)
        grouped.head(5).to_csv(destination / "nvidia_top5_kernels.csv", index=False)

        memcpy_count = 0
        memcpy_total_ns = 0
        for table in memcpy_tables:
            for start, end in stream_activity_rows(
                connection,
                table,
                training_start,
                training_end,
                string_ids,
                include_name=False,
            ):
                memcpy_count += 1
                memcpy_total_ns += end - start

        def kernel_plain_generator():
            for start, end, _ in stream_activity_rows(
                connection,
                kernel_table,
                training_start,
                training_end,
                string_ids,
                include_name=True,
            ):
                yield start, end

        memcpy_generators = [
            stream_activity_rows(
                connection,
                table,
                training_start,
                training_end,
                string_ids,
                include_name=False,
            )
            for table in memcpy_tables
        ]
        merged_memcpy = heapq.merge(*memcpy_generators, key=lambda pair: pair[0]) if memcpy_generators else iter(())
        all_activity = heapq.merge(
            kernel_plain_generator(),
            merged_memcpy,
            key=lambda pair: pair[0],
        )
        activity_union_ns = union_duration_sorted(all_activity)

        kernel_idle_ns = max(training_duration_ns - kernel_union_ns, 0)
        activity_idle_ns = max(training_duration_ns - activity_union_ns, 0)
        top = grouped.iloc[0]
        metrics = {
            "training_window_duration_s": training_duration_ns / 1e9,
            "total_kernel_launches": int(kernel_rows),
            "unique_kernel_names": int(len(grouped)),
            "summed_kernel_duration_s": total_summed_kernel_ns / 1e9,
            "kernel_union_busy_time_s": kernel_union_ns / 1e9,
            "gpu_kernel_idle_time_s": kernel_idle_ns / 1e9,
            "gpu_kernel_idle_percent": kernel_idle_ns / training_duration_ns * 100.0,
            "cuda_activity_union_busy_time_s": activity_union_ns / 1e9,
            "cuda_activity_idle_time_s": activity_idle_ns / 1e9,
            "cuda_activity_idle_percent": activity_idle_ns / training_duration_ns * 100.0,
            "memory_copy_calls": int(memcpy_count),
            "memory_copy_total_duration_ms": memcpy_total_ns / 1e6,
            "average_memory_copy_duration_us": (
                memcpy_total_ns / memcpy_count / 1e3 if memcpy_count else float("nan")
            ),
            "top_kernel": str(top["kernel_name"]),
            "top_kernel_calls": int(top["calls"]),
            "top_kernel_cumulative_duration_ms": float(top["total_duration_ms"]),
            "top_kernel_percentage_of_summed_kernel_time": float(
                top["percentage_of_summed_kernel_time"]
            ),
            "nvtx_marker_table": marker_table,
            "kernel_table": kernel_table,
            "memcpy_tables": ",".join(memcpy_tables),
            "training_window_filter_applied": True,
        }
        pd.DataFrame([metrics]).to_csv(destination / "nsys_training_metrics.csv", index=False)
        with (destination / "nsys_training_metrics.json").open("w", encoding="utf-8") as file:
            json.dump(metrics, file, indent=2)
        metadata = {
            "sqlite_file": str(sqlite_path),
            "sqlite_tables": sorted(tables),
            "training_marker": TRAINING_RANGE_NAME,
            "training_start_timestamp_ns": training_start,
            "training_end_timestamp_ns": training_end,
            "kernel_rows_streamed": kernel_rows,
            "unique_training_kernels": len(grouped),
            "memory_copy_rows_streamed": memcpy_count,
            "chunk_rows": SQL_CHUNK_ROWS,
            "raw_per_event_csv_written": False,
            "idle_definition": (
                "kernel idle is training-window time not covered by the union "
                "of CUDA kernel intervals; CUDA activity idle also treats "
                "memory-copy intervals as busy"
            ),
        }
        with (destination / "nsys_parse_metadata.json").open("w", encoding="utf-8") as file:
            json.dump(metadata, file, indent=2)
        return metrics




def validate_worker(worker_dir: Path, kind: str, mode: str) -> None:
    marker_path = worker_dir / "WORKER_VALID.json"
    if not marker_path.is_file():
        raise RuntimeError(f"{kind} did not produce WORKER_VALID.json")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "VALID":
        raise RuntimeError(f"{kind} marker is not VALID: {marker}")
    summary_path = worker_dir / "summary.csv"
    if not summary_path.is_file():
        raise RuntimeError(f"{kind} summary.csv is missing")
    summary = pd.read_csv(summary_path).iloc[0]

    if mode in {"smoke", "timeline"}:
        expected_samples, expected_batches, expected_validation = 256, 4, 256
    elif mode == "one_epoch":
        expected_samples, expected_batches, expected_validation = 67_349, 1_053, 872
    else:
        expected_samples, expected_batches, expected_validation = 673_490, 10_530, 872

    if int(summary["total_samples_processed"]) != expected_samples:
        raise RuntimeError(
            f"{kind} processed {summary['total_samples_processed']} samples; "
            f"expected {expected_samples}."
        )
    if int(summary["total_batches_processed"]) != expected_batches:
        raise RuntimeError(
            f"{kind} processed {summary['total_batches_processed']} batches; "
            f"expected {expected_batches}."
        )
    if int(summary["validation_samples"]) != expected_validation:
        raise RuntimeError(
            f"{kind} validated {summary['validation_samples']} samples; "
            f"expected {expected_validation}."
        )

    def as_bool(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return str(value).strip().lower() in {"true", "1", "yes"}

    if not as_bool(summary["amp_enabled"]):
        raise RuntimeError(f"{kind} did not enable AMP.")
    if int(summary["workers"]) != 4:
        raise RuntimeError(f"{kind} did not use four workers.")
    if str(summary["padding"]) != "dynamic_per_batch":
        raise RuntimeError(f"{kind} did not use dynamic padding.")
    if as_bool(summary["thermal_throttle_seen"]):
        raise RuntimeError(f"{kind} reported thermal throttling.")
    if str(summary["driver_version"]) != EXPECTED_NVIDIA_DRIVER:
        raise RuntimeError(f"{kind} used the wrong driver version.")


def build_profile_summary(output_dir: Path, nsys_metrics: dict) -> None:
    primary = pd.read_csv(output_dir / "pytorch_operator_pass" / "summary.csv").iloc[0]
    nsys_summary = pd.read_csv(output_dir / "nsight_systems_worker_pass" / "summary.csv").iloc[0]
    rows = [
        ("Profiled Throughput", primary["throughput_samples_s"], "samples/s", "PyTorch representative operator pass"),
        ("Batch Latency", primary["batch_latency_ms_batch"], "ms/batch", "PyTorch representative operator pass"),
        ("Total Training Time", primary["total_training_time_s"], "s", "PyTorch representative operator pass"),
        ("Operator Aggregation Overhead", primary["operator_aggregation_overhead_s"], "s", "PyTorch diagnostic; included in full pass timing only when inside training window"),
        ("Average Epoch Time", primary["average_epoch_time_s"], "s", "PyTorch representative operator pass"),
        ("Average GPU Power Draw", primary["average_gpu_power_w"], "W", "PyTorch representative operator pass"),
        ("Performance per Watt", primary["performance_per_watt_samples_s_w"], "samples/s/W", "PyTorch representative operator pass"),
        ("Average GPU Utilization", primary["average_gpu_util_percent"], "%", "PyTorch representative operator pass"),
        ("Average VRAM Usage", primary["average_vram_usage_mb"], "MB", "PyTorch representative operator pass"),
        ("Peak VRAM Usage", primary["peak_vram_usage_mb"], "MB", "PyTorch representative operator pass"),
        ("Average GPU Temperature", primary["average_gpu_temperature_c"], "C", "PyTorch representative operator pass"),
        ("Final Validation Accuracy", primary["final_validation_accuracy"], "accuracy", "PyTorch representative operator pass"),
        ("Nsight Throughput", nsys_summary["throughput_samples_s"], "samples/s", "Nsight full quantitative pass"),
        ("Nsight Training Time", nsys_summary["total_training_time_s"], "s", "Nsight full quantitative pass"),
        ("Nsight Average GPU Power", nsys_summary["average_gpu_power_w"], "W", "Nsight full quantitative pass"),
        ("Nsight Average GPU Utilization", nsys_summary["average_gpu_util_percent"], "%", "Nsight full quantitative pass"),
        ("Nsight Average VRAM", nsys_summary["average_vram_usage_mb"], "MB", "Nsight full quantitative pass"),
        ("Nsight Peak VRAM", nsys_summary["peak_vram_usage_mb"], "MB", "Nsight full quantitative pass"),
        ("Nsight Average GPU Temperature", nsys_summary["average_gpu_temperature_c"], "C", "Nsight full quantitative pass"),
        ("Total Kernel Launches", nsys_metrics["total_kernel_launches"], "count", "Nsight NVTX window"),
        ("Unique Kernel Names", nsys_metrics["unique_kernel_names"], "count", "Nsight NVTX window"),
        ("Summed Kernel Duration", nsys_metrics["summed_kernel_duration_s"], "s", "Nsight NVTX window"),
        ("GPU Kernel Idle Time", nsys_metrics["gpu_kernel_idle_time_s"], "s", "Exact interval union"),
        ("GPU Kernel Idle Percentage", nsys_metrics["gpu_kernel_idle_percent"], "%", "Exact interval union"),
        ("CUDA Activity Idle Time", nsys_metrics["cuda_activity_idle_time_s"], "s", "Kernels plus memory copies"),
        ("CUDA Activity Idle Percentage", nsys_metrics["cuda_activity_idle_percent"], "%", "Kernels plus memory copies"),
        ("Memory Copy Calls", nsys_metrics["memory_copy_calls"], "count", "Nsight NVTX window"),
        ("Memory Copy Total Duration", nsys_metrics["memory_copy_total_duration_ms"], "ms", "Nsight NVTX window"),
        ("Average Memory Copy Duration", nsys_metrics["average_memory_copy_duration_us"], "us/copy", "Nsight NVTX window"),
    ]
    frame = pd.DataFrame(rows, columns=["metric", "value", "unit", "source"])
    frame.to_csv(output_dir / "profile_metrics_summary.csv", index=False)
    (output_dir / "profile_metrics_summary.txt").write_text(
        frame.to_string(index=False), encoding="utf-8"
    )


def run_parent(smoke: bool, one_epoch: bool = False) -> None:
    if smoke:
        output_dir = TEMP_PROFILE_DIR
        display_mode = "smoke test"
        validation_mode = "smoke"
        mode_args = ["--internal-smoke"]
    elif one_epoch:
        output_dir = ONE_EPOCH_PROFILE_DIR
        display_mode = "full-dataset one-epoch test"
        validation_mode = "one_epoch"
        mode_args = ["--internal-one-epoch"]
    else:
        output_dir = PROFILE_DIR
        display_mode = "official"
        validation_mode = "official"
        mode_args = []

    reset_directory(output_dir)
    log_file = (output_dir / "training_output.log").open("w", encoding="utf-8", buffering=1)
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
            raise RuntimeError(f"Nsight Systems is missing: {NSYS}")

        torch_dir = output_dir / "pytorch_operator_pass"
        nsys_worker_dir = output_dir / "nsight_systems_worker_pass"
        nsys_raw_dir = output_dir / "nsight_systems_raw"
        timeline_worker_dir = output_dir / "nsight_timeline_worker_pass"
        timeline_raw_dir = output_dir / "nsight_timeline_raw"
        nsys_raw_dir.mkdir(parents=True, exist_ok=True)
        timeline_raw_dir.mkdir(parents=True, exist_ok=True)

        report_prefix = nsys_raw_dir / "distilbert_full"
        report_path = Path(str(report_prefix) + ".nsys-rep")
        sqlite_path = nsys_raw_dir / "distilbert_full.sqlite"
        timeline_prefix = timeline_raw_dir / "distilbert_timeline"
        timeline_report = Path(str(timeline_prefix) + ".nsys-rep")
        current_hash = sha256_file(PROFILE_SCRIPT)

        expected_training_batches = 4 if smoke else (1_053 if one_epoch else 10_530)
        expected_profiled_batches = 4 if smoke else 100
        parent_config = {
            "mode": validation_mode,
            "workflow": "three_pass_single_command",
            "primary_profile_metrics_source": "pytorch_operator_pass",
            "operator_profiler": (
                "one representative legacy CUDA-event ATen window during a complete training run"
            ),
            "kernel_profiler": "Nsight Systems full quantitative CUDA/NVTX trace",
            "representative_timeline": "separate fixed four-batch Nsight Systems trace",
            "reason_for_separate_passes": (
                "Operator and kernel profilers are separated to prevent profiler interference."
            ),
            "operator_window_used_for_full_run_cumulative_counts": False,
            "timeline_used_for_quantitative_metrics": False,
            "expected_nvidia_driver": EXPECTED_NVIDIA_DRIVER,
            "training_range_name": TRAINING_RANGE_NAME,
            "pytorch_wait_steps": 0 if smoke else 10,
            "pytorch_warmup_steps": 0 if smoke else 10,
            "pytorch_active_profiled_steps": expected_profiled_batches,
            "full_training_batches_executed": expected_training_batches,
            "batch_size": regular.BATCH_SIZE,
            "epochs": 1 if (smoke or one_epoch) else regular.EPOCHS,
            "max_length": regular.MAX_LENGTH,
            "padding": "dynamic_per_batch",
            "amp_enabled": regular.AMP_ENABLED,
            "workers": regular.NUM_WORKERS,
            "regular_script_sha256": sha256_file(REGULAR_SCRIPT),
            "profile_script_sha256": current_hash,
            "profile_script_revision": SCRIPT_REVISION,
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        with (output_dir / "profile_workflow_config.json").open("w", encoding="utf-8") as file:
            json.dump(parent_config, file, indent=2)

        print("=" * 92)
        print(f"DistilBERT NVIDIA profiled workflow ({display_mode})")
        print("=" * 92)
        print(f"Output directory: {output_dir}")
        print(
            "This performs three separate passes: one representative PyTorch "
            "operator window during the complete workload, full Nsight quantitative "
            "tracing, then a four-batch timeline."
        )
        print()

        print("PHASE 1/3: PYTORCH REPRESENTATIVE OPERATOR WINDOW")
        print("-" * 92)
        torch_command = [
            sys.executable, str(PROFILE_SCRIPT),
            "--internal-worker", "torch_operator",
            "--internal-output", str(torch_dir),
            *mode_args,
        ]
        rc = stream_subprocess(torch_command, env=os.environ.copy())
        if rc != 0:
            raise RuntimeError(f"PyTorch operator worker exited with code {rc}.")
        validate_worker(torch_dir, "PyTorch operator worker", validation_mode)
        print()

        print("PHASE 2/3: NSIGHT SYSTEMS FULL QUANTITATIVE TRACE")
        print("-" * 92)
        nsys_command = [
            str(NSYS), "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=true",
            "--output", str(report_prefix),
            sys.executable, str(PROFILE_SCRIPT),
            "--internal-worker", "nsight_systems",
            "--internal-output", str(nsys_worker_dir),
            *mode_args,
        ]
        rc = stream_subprocess(nsys_command, env=os.environ.copy())
        if rc != 0:
            raise RuntimeError(f"Nsight full pass exited with code {rc}.")
        validate_worker(nsys_worker_dir, "Nsight quantitative worker", validation_mode)
        if not report_path.is_file():
            raise RuntimeError(f"Nsight report is missing: {report_path}")
        print()

        print("EXPORTING NSIGHT SYSTEMS SQLITE")
        print("-" * 92)
        export_command = [
            str(NSYS), "export",
            "--type=sqlite",
            "--force-overwrite=true",
            "--output", str(sqlite_path),
            str(report_path),
        ]
        rc = stream_subprocess(export_command, env=os.environ.copy())
        if rc != 0:
            raise RuntimeError(f"Nsight SQLite export exited with code {rc}.")
        print()

        print("PARSING NSIGHT SYSTEMS KERNEL/COPY DATA")
        print("-" * 92)
        nsys_metrics = parse_nsys_sqlite(sqlite_path, output_dir)
        print(
            f"Training kernels parsed: {nsys_metrics['total_kernel_launches']} "
            f"launches across {nsys_metrics['unique_kernel_names']} unique names."
        )
        print(
            f"GPU kernel-idle time: {nsys_metrics['gpu_kernel_idle_time_s']:.6f} s "
            f"({nsys_metrics['gpu_kernel_idle_percent']:.3f}%)."
        )
        print()

        print("PHASE 3/3: FOUR-BATCH NSIGHT SYSTEMS TIMELINE")
        print("-" * 92)
        timeline_command = [
            str(NSYS), "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=true",
            "--output", str(timeline_prefix),
            sys.executable, str(PROFILE_SCRIPT),
            "--internal-worker", "nsight_systems",
            "--internal-output", str(timeline_worker_dir),
            "--internal-timeline",
        ]
        rc = stream_subprocess(timeline_command, env=os.environ.copy())
        if rc != 0:
            raise RuntimeError(f"Nsight timeline pass exited with code {rc}.")
        validate_worker(timeline_worker_dir, "Nsight timeline worker", "timeline")
        if not timeline_report.is_file():
            raise RuntimeError(f"Nsight timeline report is missing: {timeline_report}")
        print(f"Representative timeline: {timeline_report}")
        print()

        primary_files = (
            "config.json", "epoch_metrics.csv", "telemetry.csv",
            "nvml_api_errors.json", "summary.csv", "summary.txt",
            "pytorch_all_operators.csv", "pytorch_top10_operators.csv",
            "pytorch_profiler_full_table.txt",
            "pytorch_operator_window_progress.csv",
            "pytorch_operator_window_metadata.json",
        )
        for name in primary_files:
            link_or_copy(torch_dir / name, output_dir / name)
        link_or_copy(
            torch_dir / "training_output.log",
            output_dir / "pytorch_operator_training_output.log",
        )
        link_or_copy(
            nsys_worker_dir / "training_output.log",
            output_dir / "nsight_worker_training_output.log",
        )
        link_or_copy(
            nsys_worker_dir / "summary.csv",
            output_dir / "nsight_worker_summary.csv",
        )
        link_or_copy(
            nsys_worker_dir / "summary.txt",
            output_dir / "nsight_worker_summary.txt",
        )
        link_or_copy(
            timeline_worker_dir / "summary.txt",
            output_dir / "nsight_timeline_summary.txt",
        )

        top_operators = pd.read_csv(output_dir / "pytorch_top10_operators.csv")
        top_kernels = pd.read_csv(output_dir / "nvidia_top5_kernels.csv")
        if top_operators.empty or top_kernels.empty:
            raise RuntimeError("Final operator or kernel ranking is empty.")

        window_metadata = json.loads(
            (output_dir / "pytorch_operator_window_metadata.json").read_text(encoding="utf-8")
        )
        if int(window_metadata.get("windows_completed", 0)) != 1:
            raise RuntimeError("Unexpected representative operator window count.")
        if int(window_metadata.get("total_profiled_batches", 0)) != expected_profiled_batches:
            raise RuntimeError("Unexpected representative profiled batch count.")
        if int(window_metadata.get("total_training_batches_executed", 0)) != expected_training_batches:
            raise RuntimeError("Unexpected full training batch count in operator metadata.")

        build_profile_summary(output_dir, nsys_metrics)
        marker = {
            "status": "VALID",
            "mode": validation_mode,
            "workflow": "three_pass_single_command",
            "pytorch_operator_pass_valid": True,
            "pytorch_profiled_batches": expected_profiled_batches,
            "pytorch_total_training_batches_executed": expected_training_batches,
            "nsight_quantitative_pass_valid": True,
            "nsight_timeline_pass_valid": True,
            "timeline_used_for_quantitative_metrics": False,
            "top_pytorch_operator": str(top_operators.iloc[0]["operator"]),
            "top_nvidia_kernel": str(top_kernels.iloc[0]["kernel_name"]),
            "total_kernel_launches": int(nsys_metrics["total_kernel_launches"]),
            "gpu_kernel_idle_time_s": float(nsys_metrics["gpu_kernel_idle_time_s"]),
            "gpu_kernel_idle_percent": float(nsys_metrics["gpu_kernel_idle_percent"]),
            "driver_version": EXPECTED_NVIDIA_DRIVER,
            "profile_script_sha256": current_hash,
            "regular_script_sha256": sha256_file(REGULAR_SCRIPT),
            "timeline_report": str(timeline_report),
            "completed_at": datetime.now().astimezone().isoformat(),
        }
        with (output_dir / "PROFILE_VALID.json").open("w", encoding="utf-8") as file:
            json.dump(marker, file, indent=2)

        print("=" * 92)
        print("PROFILED WORKFLOW VALID")
        print("=" * 92)
        print(f"Profiled operator batches: {expected_profiled_batches}")
        print(f"Top PyTorch operator: {marker['top_pytorch_operator']}")
        print(f"Top NVIDIA kernel: {marker['top_nvidia_kernel']}")
        print(f"Total kernel launches: {marker['total_kernel_launches']}")
        print(
            f"GPU kernel idle: {marker['gpu_kernel_idle_time_s']:.6f} s "
            f"({marker['gpu_kernel_idle_percent']:.3f}%)"
        )
        print(f"Representative timeline: {timeline_report}")
        print(f"Results saved to: {output_dir}")

    except Exception:
        traceback_text = traceback.format_exc()
        print("\nPROFILED WORKFLOW FAILED")
        print(traceback_text)
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DistilBERT NVIDIA three-pass profiling workflow"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--one-epoch-test", action="store_true")
    mode.add_argument("--official", action="store_true")
    mode.add_argument(
        "--internal-worker", choices=("torch_operator", "nsight_systems")
    )
    parser.add_argument("--internal-output", type=Path)
    parser.add_argument("--internal-smoke", action="store_true")
    parser.add_argument("--internal-one-epoch", action="store_true")
    parser.add_argument("--internal-timeline", action="store_true")
    args = parser.parse_args()

    if args.internal_worker:
        if args.internal_output is None:
            parser.error("--internal-output is required with --internal-worker")
        run_worker(
            profile_kind=args.internal_worker,
            output_dir=args.internal_output,
            smoke=bool(args.internal_smoke),
            one_epoch=bool(args.internal_one_epoch),
            timeline=bool(args.internal_timeline),
        )
        return
    run_parent(smoke=bool(args.smoke), one_epoch=bool(args.one_epoch_test))


if __name__ == "__main__":
    main()

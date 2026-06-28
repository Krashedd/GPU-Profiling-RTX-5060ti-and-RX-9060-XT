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
import ultralytics
from torch.autograd.profiler import profile as autograd_profile
from ultralytics import YOLO


PROJECT_DIR = Path.home() / "Thesis-Testing"
SCRIPT_DIR = PROJECT_DIR / "scripts"
REGULAR_SCRIPT = SCRIPT_DIR / "yolov8m_regular_nvidia.py"
PROFILE_SCRIPT = SCRIPT_DIR / "yolov8m_profiled_nvidia.py"
PROFILE_DIR = PROJECT_DIR / "Logs" / "yoloprofiledrun"
TEMP_ROOT = Path.home() / "Downloads" / "Thesis-Testing-Temporary" / "YOLOv8m"
TEMP_PROFILE_DIR = TEMP_ROOT / "profile_smoke"
ONE_EPOCH_PROFILE_DIR = TEMP_ROOT / "profile_one_epoch"
NSYS = Path("/usr/local/bin/nsys")
TRAINING_RANGE_NAME = "YOLOV8M_TRAINING"
EXPECTED_NVIDIA_DRIVER = "580.167.08"
SCRIPT_REVISION = "2026-06-22 nvidia-yolov8m-profiled-v1 incremental-safe-three-pass"
MAX_LOG_BYTES = 64 * 1024 * 1024
SQL_CHUNK_ROWS = 250_000

EXPECTED_CONFIG = {
    "SEED": 55,
    "BATCH_SIZE": 16,
    "EPOCHS": 10,
    "IMGSZ": 640,
    "LR0": 0.01,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 1e-4,
    "NUM_WORKERS": 4,
    "SAMPLE_INTERVAL_S": 0.2,
}


class CappedLogFile:
    def __init__(self, path: Path, max_bytes: int = MAX_LOG_BYTES):
        self.path = path
        self.max_bytes = max_bytes
        self.file = path.open("w", encoding="utf-8", buffering=1)
        self.bytes_written = 0
        self.truncated = False

    def write(self, data):
        if self.truncated:
            return len(data)
        encoded = data.encode("utf-8", errors="replace")
        remaining = self.max_bytes - self.bytes_written
        if remaining <= 0:
            self._mark_truncated()
            return len(data)
        if len(encoded) <= remaining:
            self.file.write(data)
            self.bytes_written += len(encoded)
            return len(data)
        partial = encoded[:remaining].decode("utf-8", errors="ignore")
        self.file.write(partial)
        self.bytes_written += len(partial.encode("utf-8"))
        self._mark_truncated()
        return len(data)

    def _mark_truncated(self):
        if not self.truncated:
            self.file.write(
                "\n[LOG TRUNCATED: maximum configured size reached; "
                "benchmark continued.]\n"
            )
            self.file.flush()
            self.truncated = True

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()


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
        "yolov8m_regular_nvidia_shared", REGULAR_SCRIPT
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
    if mismatches:
        raise RuntimeError(
            "The profiling and regular scripts no longer use the same fixed "
            "workload/environment:\n" + "\n".join(mismatches)
        )

    required = {
        "DATASET_ROOT": module.DATASET_ROOT,
        "DATA_YAML": module.DATA_YAML,
        "WEIGHTS_PATH": module.WEIGHTS_PATH,
    }
    missing = [f"{name}: {path}" for name, path in required.items() if not Path(path).exists()]
    if missing:
        raise RuntimeError("Required YOLO assets are missing:\n" + "\n".join(missing))
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


def operator_note(name: str) -> str:
    exact = {
        "aten::convolution_backward": "Convolution backward",
        "aten::cudnn_convolution": "cuDNN convolution forward",
        "aten::cudnn_batch_norm_backward": "cuDNN batch normalization backward",
        "aten::cudnn_batch_norm": "cuDNN batch normalization forward",
        "aten::copy_": "Tensor copy",
        "aten::add_": "In-place tensor addition",
        "aten::_foreach_add_": "In-place multi-tensor addition",
        "aten::_foreach_add": "Multi-tensor addition",
        "Optimizer.step#SGD.step": "SGD optimizer update",
    }
    return exact.get(name, "")


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


def rank_rows(rows):
    ranked = sorted(rows, key=lambda row: row["self_gpu_time_ms"], reverse=True)
    total_self = sum(row["self_gpu_time_ms"] for row in ranked)
    for rank, row in enumerate(ranked, start=1):
        row["rank"] = rank
        row["self_gpu_percent"] = (
            row["self_gpu_time_ms"] / total_self * 100.0
            if total_self > 0
            else float("nan")
        )
    return ranked


def save_operator_outputs(accumulator: dict, output_dir: Path) -> pd.DataFrame:
    columns = [
        "rank",
        "operator",
        "self_gpu_time_ms",
        "self_gpu_percent",
        "gpu_total_ms",
        "calls",
        "notes",
    ]
    all_rows = [dict(row) for row in accumulator.values()]
    all_frame = pd.DataFrame(rank_rows(all_rows), columns=columns)
    all_frame.to_csv(output_dir / "pytorch_all_profile_events.csv", index=False)

    aten_rows = [dict(row) for row in all_rows if str(row["operator"]).startswith("aten::")]
    aten_frame = pd.DataFrame(rank_rows(aten_rows), columns=columns)
    aten_frame.to_csv(output_dir / "pytorch_all_operators.csv", index=False)
    aten_frame.head(10).to_csv(output_dir / "pytorch_top10_operators.csv", index=False)
    (output_dir / "pytorch_profiler_full_table.txt").write_text(
        aten_frame.head(200).to_string(index=False), encoding="utf-8"
    )
    return aten_frame


def merge_intervals(intervals):
    ordered = sorted((float(a), float(b)) for a, b in intervals if b > a)
    if not ordered:
        return []
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [(a, b) for a, b in merged]


def filter_telemetry_intervals(frame: pd.DataFrame, excluded_intervals) -> pd.DataFrame:
    if frame.empty or not excluded_intervals or "time_s" not in frame.columns:
        return frame.copy()
    merged = merge_intervals(excluded_intervals)
    times = pd.to_numeric(frame["time_s"], errors="coerce").to_numpy()
    keep = np.ones(len(frame), dtype=bool)
    interval_index = 0
    for row_index, value in enumerate(times):
        if math.isnan(value):
            continue
        while interval_index < len(merged) and value > merged[interval_index][1]:
            interval_index += 1
        if interval_index < len(merged):
            start, end = merged[interval_index]
            if start <= value <= end:
                keep[row_index] = False
    return frame.loc[keep].reset_index(drop=True)


class ProfileTrainingState:
    def __init__(self, regular, monitor, epochs_to_run: int, profile_kind: str):
        self.regular = regular
        self.monitor = monitor
        self.epochs_to_run = epochs_to_run
        self.profile_kind = profile_kind

        self.epoch_rows = []
        self.epoch_start_time = None
        self.epoch_aggregation_start = 0.0
        self.training_start_time = None
        self.training_end_time = None
        self.dataset_size = None
        self.batches_per_epoch = None
        self.amp_enabled = None
        self.model_dtype = None
        self.window_started = False
        self.window_stopped = False
        self.monitor_started = False
        self.monitor_stopped = False
        self.nvtx_active = False

        self.current_profiler = None
        self.current_batch_profile_start = None
        self.operator_accumulator = {}
        self.operator_aggregation_overhead_s = 0.0
        self.operator_profile_context_time_s = 0.0
        self.aggregation_intervals = []
        self.representative_trace_saved = False
        self.profiled_batches = 0

    def on_train_start(self, trainer):
        self.amp_enabled = bool(trainer.amp)
        try:
            self.model_dtype = str(next(trainer.model.parameters()).dtype)
        except Exception:
            self.model_dtype = "unknown"

        print()
        print("Training configuration verification")
        print(f"Precision: FP32 (AMP disabled: {not self.amp_enabled})")
        print(f"Optimizer: {trainer.args.optimizer}")
        print(f"Batch size: {trainer.args.batch}")
        print(f"Workers: {trainer.args.workers}")
        print(f"Epochs: {trainer.args.epochs}")
        print(f"Model parameter dtype: {self.model_dtype}")

        if self.amp_enabled:
            raise RuntimeError("AMP is enabled, but this workload requires AMP off.")
        if int(trainer.args.workers) != self.regular.NUM_WORKERS:
            raise RuntimeError(
                f"Expected {self.regular.NUM_WORKERS} workers, got {trainer.args.workers}."
            )
        if int(trainer.args.batch) != self.regular.BATCH_SIZE:
            raise RuntimeError(
                f"Expected batch size {self.regular.BATCH_SIZE}, got {trainer.args.batch}."
            )
        if str(trainer.args.optimizer).upper() != "SGD":
            raise RuntimeError(f"Expected SGD optimizer, got {trainer.args.optimizer}.")

    def on_train_epoch_start(self, trainer):
        torch.cuda.synchronize()
        self.epoch_start_time = time.perf_counter()
        self.epoch_aggregation_start = self.operator_aggregation_overhead_s

        if not self.window_started:
            self.dataset_size = len(trainer.train_loader.dataset)
            self.batches_per_epoch = len(trainer.train_loader)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.monitor.start()
            self.monitor_started = True
            self.training_start_time = self.epoch_start_time
            if self.profile_kind == "nsight_systems":
                torch.cuda.nvtx.range_push(TRAINING_RANGE_NAME)
                self.nvtx_active = True
                print(f"NVTX training range started: {TRAINING_RANGE_NAME}")
            self.window_started = True
            print("Monitored profiled training started.")

    def on_train_batch_start(self, trainer):
        if self.profile_kind != "torch_operator":
            return
        if self.current_profiler is not None:
            raise RuntimeError("Previous batch profiler was not finalized.")
        self.current_batch_profile_start = time.perf_counter()
        self.current_profiler = autograd_profile(
            use_device="cuda",
            use_kineto=False,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        )
        self.current_profiler.__enter__()

    def on_train_batch_end(self, trainer):
        if self.profile_kind != "torch_operator":
            return
        if self.current_profiler is None:
            raise RuntimeError("Batch end reached without an active profiler.")

        profiler = self.current_profiler
        self.current_profiler = None
        profiler.__exit__(None, None, None)
        self.operator_profile_context_time_s += (
            time.perf_counter() - self.current_batch_profile_start
        )

        aggregation_start = time.perf_counter()
        start_relative = aggregation_start - self.monitor.start_time
        aggregate_profiler_events(profiler, self.operator_accumulator)
        if not self.representative_trace_saved:
            profiler.export_chrome_trace(
                str(Path(trainer.save_dir).parent / "pytorch_trace_representative_batch.json")
            )
            self.representative_trace_saved = True
        del profiler
        self.profiled_batches += 1
        if self.profiled_batches % 100 == 0:
            gc.collect()
        aggregation_end = time.perf_counter()
        end_relative = aggregation_end - self.monitor.start_time
        self.operator_aggregation_overhead_s += aggregation_end - aggregation_start
        self.aggregation_intervals.append((start_relative, end_relative))

    def on_train_epoch_end(self, trainer):
        torch.cuda.synchronize()
        epoch_end = time.perf_counter()
        raw_epoch_time = epoch_end - self.epoch_start_time
        epoch_aggregation = (
            self.operator_aggregation_overhead_s - self.epoch_aggregation_start
        )
        adjusted_epoch_time = raw_epoch_time - epoch_aggregation
        if adjusted_epoch_time <= 0:
            raise RuntimeError("Adjusted epoch time is not positive.")

        samples = int(self.dataset_size)
        batches = int(self.batches_per_epoch)
        throughput = samples / adjusted_epoch_time

        losses = {}
        try:
            values = trainer.tloss.detach().float().cpu().tolist()
            if not isinstance(values, list):
                values = [values]
            names = ["box_loss", "cls_loss", "dfl_loss"]
            losses = {
                names[index] if index < len(names) else f"loss_{index}": float(value)
                for index, value in enumerate(values)
            }
        except Exception:
            pass

        row = {
            "epoch": int(trainer.epoch) + 1,
            "epoch_time_s": adjusted_epoch_time,
            "raw_epoch_wall_time_s": raw_epoch_time,
            "operator_aggregation_time_s": epoch_aggregation,
            "throughput_images_s": throughput,
            "samples": samples,
            "batches": batches,
            **losses,
        }
        self.epoch_rows.append(row)

        loss_text = " | ".join(f"{k}: {v:.4f}" for k, v in losses.items())
        if loss_text:
            loss_text = " | " + loss_text
        print(
            f"Epoch {trainer.epoch + 1}/{trainer.args.epochs} | "
            f"Adjusted Time: {adjusted_epoch_time:.2f}s | "
            f"Raw Wall: {raw_epoch_time:.2f}s | "
            f"Throughput: {throughput:.2f} images/s | "
            f"Batches: {batches}{loss_text}"
        )

        if int(trainer.epoch) + 1 == int(trainer.args.epochs):
            self.training_end_time = epoch_end
            self.stop_window()

    def on_train_end(self, trainer):
        if self.training_end_time is None and self.window_started:
            torch.cuda.synchronize()
            self.training_end_time = time.perf_counter()
        self.stop_window()

    def stop_window(self):
        if self.window_stopped:
            return
        if self.current_profiler is not None:
            profiler = self.current_profiler
            self.current_profiler = None
            profiler.__exit__(None, None, None)
        if self.nvtx_active:
            torch.cuda.nvtx.range_pop()
            self.nvtx_active = False
            print(f"NVTX training range ended: {TRAINING_RANGE_NAME}")
        if self.monitor_started and not self.monitor_stopped:
            self.monitor.stop()
            self.monitor_stopped = True
        if self.window_started:
            self.window_stopped = True

    @property
    def raw_training_time(self) -> float:
        if self.training_start_time is None or self.training_end_time is None:
            return float("nan")
        return self.training_end_time - self.training_start_time

    @property
    def adjusted_training_time(self) -> float:
        raw = self.raw_training_time
        if math.isnan(raw):
            return raw
        return raw - self.operator_aggregation_overhead_s


def series_stat(regular, frame, column, operation):
    return regular.series_stat(frame, column, operation)


def run_worker(
    profile_kind: str,
    output_dir: Path,
    smoke: bool,
    one_epoch: bool = False,
    timeline: bool = False,
) -> None:
    regular = load_regular_module()
    reset_directory(output_dir)

    log_file = CappedLogFile(output_dir / "training_output.log")
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    monitor = None
    state = None

    try:
        os.environ.setdefault("WANDB_DISABLED", "true")
        os.environ.setdefault("COMET_DISABLE_AUTO_LOGGING", "1")
        os.environ.setdefault("YOLO_VERBOSE", "true")
        regular.set_seed(regular.SEED)
        counts = regular.verify_dataset()
        weights_hash = regular.file_sha256(regular.WEIGHTS_PATH)
        if weights_hash != regular.EXPECTED_WEIGHTS_SHA256:
            raise RuntimeError(
                "YOLOv8m weights checksum mismatch: "
                f"expected {regular.EXPECTED_WEIGHTS_SHA256}, got {weights_hash}."
            )
        yaml_hash = regular.file_sha256(regular.DATA_YAML)
        if yaml_hash != regular.EXPECTED_DATA_YAML_SHA256:
            raise RuntimeError(
                "Dataset YAML checksum mismatch: "
                f"expected {regular.EXPECTED_DATA_YAML_SHA256}, got {yaml_hash}."
            )

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot detect the NVIDIA GPU.")
        if torch.version.cuda is None:
            raise RuntimeError("This PyTorch build does not report a CUDA runtime.")
        gpu_name = torch.cuda.get_device_name(0)
        compute_capability = torch.cuda.get_device_capability(0)
        if regular.EXPECTED_GPU_NAME_FRAGMENT not in gpu_name:
            raise RuntimeError(
                f"Unexpected GPU selected: {gpu_name}. Expected "
                f"{regular.EXPECTED_GPU_NAME_FRAGMENT}."
            )
        if ultralytics.__version__ != regular.EXPECTED_ULTRALYTICS_VERSION:
            raise RuntimeError(
                f"Expected Ultralytics {regular.EXPECTED_ULTRALYTICS_VERSION}, "
                f"got {ultralytics.__version__}."
            )

        if smoke or timeline:
            display_mode = "smoke test" if smoke else "four-batch timeline pass"
            run_mode = "smoke" if smoke else "timeline_four_batch"
            epochs_to_run = 1
            selected_yaml, training_images, validation_images = regular.create_smoke_yaml(
                output_dir
            )
        elif one_epoch:
            display_mode = "one-epoch full-dataset test"
            run_mode = "one_epoch_profile_test"
            epochs_to_run = 1
            selected_yaml = regular.DATA_YAML
            training_images = counts["train_images"]
            validation_images = counts["val_images"]
        else:
            display_mode = "official pass"
            run_mode = "official_profiled"
            epochs_to_run = regular.EPOCHS
            selected_yaml = regular.DATA_YAML
            training_images = counts["train_images"]
            validation_images = counts["val_images"]

        monitor = regular.TelemetryMonitor(interval_s=regular.SAMPLE_INTERVAL_S)
        if monitor.driver_version != EXPECTED_NVIDIA_DRIVER:
            raise RuntimeError(
                "NVIDIA driver changed from the frozen profiling baseline: "
                f"expected {EXPECTED_NVIDIA_DRIVER}, found {monitor.driver_version}."
            )

        print("=" * 88)
        print(
            f"YOLOv8m COCO MiniTrain 10K NVIDIA {profile_kind} profiling "
            f"{display_mode}"
        )
        print("=" * 88)
        print(f"Timestamp: {datetime.now().astimezone().isoformat()}")
        print(f"Output directory: {output_dir}")
        print(f"Dataset root: {regular.DATASET_ROOT}")
        print(f"Dataset YAML: {selected_yaml}")
        print(f"Weights: {regular.WEIGHTS_PATH}")
        print(f"Python: {platform.python_version()}")
        print(f"Platform: {platform.platform()}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"Ultralytics: {ultralytics.__version__}")
        print(f"NVIDIA driver: {monitor.driver_version}")
        print(f"GPU: {gpu_name}")
        print(
            "Compute capability: "
            f"{compute_capability[0]}.{compute_capability[1]}"
        )
        print(f"Profile script revision: {SCRIPT_REVISION}")
        if profile_kind == "torch_operator":
            print(
                "Operator capture: one legacy CUDA-event profiler context per "
                "training batch; DataLoader next() occurs before the callback."
            )
        print()

        config = {
            "mode": run_mode,
            "profile_kind": profile_kind,
            "seed": regular.SEED,
            "model": "YOLOv8m",
            "weights": str(regular.WEIGHTS_PATH),
            "weights_sha256": weights_hash,
            "dataset": "COCO 2017 MiniTrain 10K",
            "dataset_root": str(regular.DATASET_ROOT),
            "dataset_yaml": str(selected_yaml),
            "training_images": training_images,
            "validation_images": validation_images,
            "batch_size": regular.BATCH_SIZE,
            "image_size": regular.IMGSZ,
            "epochs": epochs_to_run,
            "official_epochs": regular.EPOCHS,
            "optimizer": "SGD",
            "learning_rate": regular.LR0,
            "momentum": regular.MOMENTUM,
            "weight_decay": regular.WEIGHT_DECAY,
            "precision": "FP32 / AMP OFF",
            "workers": regular.NUM_WORKERS,
            "telemetry_interval_s": regular.SAMPLE_INTERVAL_S,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "ultralytics_version": ultralytics.__version__,
            "nvidia_driver": monitor.driver_version,
            "gpu": gpu_name,
            "compute_capability": f"{compute_capability[0]}.{compute_capability[1]}",
            "training_range_name": TRAINING_RANGE_NAME,
            "operator_profiler_backend": (
                "torch.autograd.profiler CUDA-event timing "
                "(use_device=cuda, use_kineto=False)"
                if profile_kind == "torch_operator"
                else None
            ),
            "operator_capture_strategy": (
                "one context per batch through Ultralytics on_train_batch_start/"
                "on_train_batch_end callbacks; immediate compact aggregation"
                if profile_kind == "torch_operator"
                else None
            ),
            "primary_timing_definition": (
                "training wall time minus post-batch operator-table aggregation; "
                "DataLoader, profiler context, preprocessing, forward, backward, "
                "optimizer, and synchronization remain included"
                if profile_kind == "torch_operator"
                else "NVTX-bounded complete training wall time"
            ),
            "regular_script_sha256": sha256_file(REGULAR_SCRIPT),
            "profile_script_sha256": sha256_file(Path(__file__).resolve()),
            "profile_script_revision": SCRIPT_REVISION,
            "log_size_cap_bytes": MAX_LOG_BYTES,
        }
        with (output_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

        state = ProfileTrainingState(
            regular=regular,
            monitor=monitor,
            epochs_to_run=epochs_to_run,
            profile_kind=profile_kind,
        )
        model = YOLO(str(regular.WEIGHTS_PATH))
        model.add_callback("on_train_start", state.on_train_start)
        model.add_callback("on_train_epoch_start", state.on_train_epoch_start)
        model.add_callback("on_train_batch_start", state.on_train_batch_start)
        model.add_callback("on_train_batch_end", state.on_train_batch_end)
        model.add_callback("on_train_epoch_end", state.on_train_epoch_end)
        model.add_callback("on_train_end", state.on_train_end)

        train_output_dir = output_dir / "ultralytics_train"
        print("Starting monitored profiled training...")
        model.train(
            data=str(selected_yaml),
            epochs=epochs_to_run,
            imgsz=regular.IMGSZ,
            batch=regular.BATCH_SIZE,
            device=0,
            workers=regular.NUM_WORKERS,
            optimizer="SGD",
            lr0=regular.LR0,
            momentum=regular.MOMENTUM,
            weight_decay=regular.WEIGHT_DECAY,
            amp=False,
            seed=regular.SEED,
            deterministic=True,
            val=False,
            pretrained=True,
            cache=False,
            rect=False,
            resume=False,
            save=True,
            save_period=-1,
            plots=False,
            verbose=True,
            project=str(output_dir),
            name="ultralytics_train",
            exist_ok=True,
        )

        state.stop_window()
        raw_training_time = state.raw_training_time
        adjusted_training_time = state.adjusted_training_time
        if math.isnan(adjusted_training_time) or adjusted_training_time <= 0:
            raise RuntimeError("Training callbacks did not produce valid timing.")

        epoch_df = pd.DataFrame(state.epoch_rows)
        epoch_df.to_csv(output_dir / "epoch_metrics.csv", index=False)

        telemetry_raw = monitor.dataframe()
        telemetry_raw.to_csv(output_dir / "telemetry_raw.csv", index=False)
        telemetry_df = filter_telemetry_intervals(
            telemetry_raw, state.aggregation_intervals
        )
        telemetry_df.to_csv(output_dir / "telemetry.csv", index=False)
        regular.validate_telemetry(telemetry_df)
        with (output_dir / "nvml_api_errors.json").open("w", encoding="utf-8") as file:
            json.dump(monitor.api_errors, file, indent=2)
        regular.write_nvml_error_log(output_dir, monitor)

        if profile_kind == "torch_operator":
            print("Writing incrementally aggregated PyTorch operator statistics...")
            operator_df = save_operator_outputs(
                state.operator_accumulator, output_dir
            )
            if operator_df.empty:
                raise RuntimeError("No GPU operator statistics were produced.")
            if not state.representative_trace_saved:
                raise RuntimeError("Representative PyTorch batch trace was not created.")
            print(f"Top PyTorch operator: {operator_df.iloc[0]['operator']}")
            print(
                "Post-batch operator aggregation excluded from primary timing: "
                f"{state.operator_aggregation_overhead_s:.3f}s"
            )
            print()

        if len(state.epoch_rows) != epochs_to_run:
            raise RuntimeError(
                f"Incomplete pass: expected {epochs_to_run} epochs, "
                f"got {len(state.epoch_rows)}."
            )
        batches_per_epoch = int(state.batches_per_epoch)
        total_batches = batches_per_epoch * epochs_to_run
        total_images = training_images * epochs_to_run
        expected_batches = math.ceil(training_images / regular.BATCH_SIZE) * epochs_to_run
        if total_batches != expected_batches:
            raise RuntimeError(
                f"Expected {expected_batches} batches, got {total_batches}."
            )
        if profile_kind == "torch_operator" and state.profiled_batches != total_batches:
            raise RuntimeError(
                f"Expected {total_batches} profiled batches, got {state.profiled_batches}."
            )

        last_weights = train_output_dir / "weights" / "last.pt"
        if not last_weights.is_file():
            raise RuntimeError(f"Expected trained checkpoint is missing: {last_weights}")

        print("Starting final validation outside the profiling window...")
        validation_model = YOLO(str(last_weights))
        validation_results = validation_model.val(
            data=str(selected_yaml),
            imgsz=regular.IMGSZ,
            batch=regular.BATCH_SIZE,
            device=0,
            workers=regular.NUM_WORKERS,
            amp=False,
            plots=False,
            save_json=False,
            verbose=True,
            project=str(output_dir),
            name="ultralytics_val",
            exist_ok=True,
        )
        validation = regular.extract_validation_metrics(validation_results)

        throughput = total_images / adjusted_training_time
        raw_workflow_throughput = total_images / raw_training_time
        batch_latency_ms = adjusted_training_time / total_batches * 1000.0
        average_epoch_time = float(epoch_df["epoch_time_s"].mean())

        average_power_w = series_stat(regular, telemetry_df, "power_w", "mean")
        average_gpu_util = series_stat(regular, telemetry_df, "gpu_util_percent", "mean")
        average_vram_mb = series_stat(regular, telemetry_df, "vram_used_mb", "mean")
        peak_vram_mb = series_stat(regular, telemetry_df, "vram_used_mb", "max")
        average_gpu_temp_c = series_stat(regular, telemetry_df, "gpu_temperature_c", "mean")
        peak_gpu_temp_c = series_stat(regular, telemetry_df, "gpu_temperature_c", "max")
        average_cpu_util = series_stat(regular, telemetry_df, "cpu_util_percent", "mean")
        average_system_ram_mb = series_stat(regular, telemetry_df, "system_ram_used_mb", "mean")
        peak_system_ram_mb = series_stat(regular, telemetry_df, "system_ram_used_mb", "max")
        performance_per_watt = (
            throughput / average_power_w
            if not math.isnan(average_power_w) and average_power_w > 0
            else float("nan")
        )

        summary = {
            "mode": run_mode,
            "profile_kind": profile_kind,
            "throughput_images_s": throughput,
            "raw_workflow_throughput_images_s": raw_workflow_throughput,
            "batch_latency_ms_batch": batch_latency_ms,
            "total_training_time_s": adjusted_training_time,
            "raw_training_wall_time_s": raw_training_time,
            "operator_aggregation_overhead_s": state.operator_aggregation_overhead_s,
            "operator_profile_context_time_s": state.operator_profile_context_time_s,
            "average_epoch_time_s": average_epoch_time,
            "average_gpu_power_w": average_power_w,
            "performance_per_watt_images_s_w": performance_per_watt,
            "average_gpu_util_percent": average_gpu_util,
            "average_vram_usage_mb": average_vram_mb,
            "peak_vram_usage_mb": peak_vram_mb,
            "average_gpu_temperature_c": average_gpu_temp_c,
            "peak_gpu_temperature_c": peak_gpu_temp_c,
            "final_validation_map50_95": validation["map50_95"],
            "final_validation_map50": validation["map50"],
            "final_validation_map75": validation["map75"],
            "final_validation_mean_precision": validation["mean_precision"],
            "final_validation_mean_recall": validation["mean_recall"],
            "validation_images": validation_images,
            "total_images_processed": total_images,
            "total_batches_processed": total_batches,
            "epochs_completed": len(state.epoch_rows),
            "workers": regular.NUM_WORKERS,
            "amp_enabled": bool(state.amp_enabled),
            "model_dtype": state.model_dtype,
            "thermal_throttling_observed": monitor.thermal_throttle_seen,
            "average_cpu_util_percent": average_cpu_util,
            "average_system_ram_mb": average_system_ram_mb,
            "peak_system_ram_mb": peak_system_ram_mb,
            "telemetry_samples_raw": len(telemetry_raw),
            "telemetry_samples_primary": len(telemetry_df),
            "driver_version": monitor.driver_version,
            "gpu": gpu_name,
            "script_revision": SCRIPT_REVISION,
            "profile_script_sha256": sha256_file(Path(__file__).resolve()),
            "regular_script_sha256": sha256_file(REGULAR_SCRIPT),
        }
        pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)
        with (output_dir / "summary.txt").open("w", encoding="utf-8") as file:
            for key, value in summary.items():
                file.write(f"{key}: {value}\n")

        worker_marker = {
            "status": "VALID",
            "mode": run_mode,
            "profile_kind": profile_kind,
            "total_images_processed": total_images,
            "total_batches_processed": total_batches,
            "validation_images": validation_images,
            "final_validation_map50_95": validation["map50_95"],
            "driver_version": monitor.driver_version,
            "profile_script_sha256": sha256_file(Path(__file__).resolve()),
            "regular_script_sha256": sha256_file(REGULAR_SCRIPT),
            "completed_at": datetime.now().astimezone().isoformat(),
        }
        with (output_dir / "WORKER_VALID.json").open("w", encoding="utf-8") as file:
            json.dump(worker_marker, file, indent=2)

        print("=" * 84)
        print("PROFILE WORKER VALID")
        print("=" * 84)
        print(f"Profile kind: {profile_kind}")
        print(f"Throughput: {throughput:.2f} images/s")
        print(f"Batch Latency: {batch_latency_ms:.2f} ms/batch")
        print(f"Adjusted Training Time: {adjusted_training_time:.2f} s")
        print(f"Raw Training Wall Time: {raw_training_time:.2f} s")
        print(f"Final Validation mAP50-95: {validation['map50_95']:.4f}")
        print(f"Results saved to: {output_dir}")

    except Exception:
        if state is not None:
            try:
                state.stop_window()
            except Exception:
                pass
        elif monitor is not None:
            try:
                monitor.stop()
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
        expected_images, expected_batches, expected_validation = 64, 4, 32
    elif mode == "one_epoch":
        expected_images, expected_batches, expected_validation = 10000, 625, 5000
    else:
        expected_images, expected_batches, expected_validation = 100000, 6250, 5000
    if int(summary["total_images_processed"]) != expected_images:
        raise RuntimeError(
            f"{kind} processed {summary['total_images_processed']} images; "
            f"expected {expected_images}."
        )
    if int(summary["total_batches_processed"]) != expected_batches:
        raise RuntimeError(
            f"{kind} processed {summary['total_batches_processed']} batches; "
            f"expected {expected_batches}."
        )
    if int(summary["validation_images"]) != expected_validation:
        raise RuntimeError(
            f"{kind} validated {summary['validation_images']} images; "
            f"expected {expected_validation}."
        )
    def as_bool(value):
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return str(value).strip().lower() in {"true", "1", "yes"}

    if as_bool(summary["amp_enabled"]):
        raise RuntimeError(f"{kind} unexpectedly enabled AMP.")
    if as_bool(summary["thermal_throttling_observed"]):
        raise RuntimeError(f"{kind} reported thermal throttling.")
    if str(summary["driver_version"]) != EXPECTED_NVIDIA_DRIVER:
        raise RuntimeError(f"{kind} used the wrong driver version.")


def verify_prerequisite_marker(path: Path, expected_mode: str, script_hash: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"Required prerequisite marker is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "VALID" or data.get("mode") != expected_mode:
        raise RuntimeError(f"Invalid prerequisite marker: {path}")
    if data.get("profile_script_sha256") != script_hash:
        raise RuntimeError(
            "The profiling script changed after prerequisite testing. Re-run "
            f"smoke and one-epoch profile tests. Marker: {path}"
        )
    if data.get("driver_version") != EXPECTED_NVIDIA_DRIVER:
        raise RuntimeError(f"Driver mismatch in prerequisite marker: {path}")


def build_profile_summary(output_dir: Path, nsys_metrics: dict) -> None:
    primary = pd.read_csv(output_dir / "pytorch_operator_pass" / "summary.csv").iloc[0]
    nsys_summary = pd.read_csv(output_dir / "nsight_systems_worker_pass" / "summary.csv").iloc[0]
    rows = [
        ("Profiled Throughput", primary["throughput_images_s"], "images/s", "PyTorch operator pass"),
        ("Batch Latency", primary["batch_latency_ms_batch"], "ms/batch", "PyTorch operator pass"),
        ("Total Training Time", primary["total_training_time_s"], "s", "PyTorch operator pass"),
        ("Raw Training Wall Time", primary["raw_training_wall_time_s"], "s", "PyTorch operator pass diagnostic"),
        ("Operator Aggregation Overhead", primary["operator_aggregation_overhead_s"], "s", "Excluded from primary timing"),
        ("Average Epoch Time", primary["average_epoch_time_s"], "s", "PyTorch operator pass"),
        ("Average GPU Power Draw", primary["average_gpu_power_w"], "W", "PyTorch operator pass"),
        ("Performance per Watt", primary["performance_per_watt_images_s_w"], "images/s/W", "PyTorch operator pass"),
        ("Average GPU Utilization", primary["average_gpu_util_percent"], "%", "PyTorch operator pass"),
        ("Average VRAM Usage", primary["average_vram_usage_mb"], "MB", "PyTorch operator pass"),
        ("Peak VRAM Usage", primary["peak_vram_usage_mb"], "MB", "PyTorch operator pass"),
        ("Average GPU Temperature", primary["average_gpu_temperature_c"], "C", "PyTorch operator pass"),
        ("Final Validation mAP50-95", primary["final_validation_map50_95"], "mAP50-95", "PyTorch operator pass"),
        ("Final Validation mAP50", primary["final_validation_map50"], "mAP50", "PyTorch operator pass"),
        ("Nsight Throughput", nsys_summary["throughput_images_s"], "images/s", "Nsight full quantitative pass"),
        ("Nsight Training Time", nsys_summary["total_training_time_s"], "s", "Nsight full quantitative pass"),
        ("Total Kernel Launches", nsys_metrics["total_kernel_launches"], "count", "Nsight NVTX window"),
        ("Unique Kernel Names", nsys_metrics["unique_kernel_names"], "count", "Nsight NVTX window"),
        ("Summed Kernel Duration", nsys_metrics["summed_kernel_duration_s"], "s", "Nsight NVTX window"),
        ("GPU Kernel Idle Time", nsys_metrics["gpu_kernel_idle_time_s"], "s", "Exact interval union"),
        ("GPU Kernel Idle Percentage", nsys_metrics["gpu_kernel_idle_percent"], "%", "Exact interval union"),
        ("CUDA Activity Idle Time", nsys_metrics["cuda_activity_idle_time_s"], "s", "Kernels plus memory copies"),
        ("CUDA Activity Idle Percentage", nsys_metrics["cuda_activity_idle_percent"], "%", "Kernels plus memory copies"),
        ("Memory Copy Calls", nsys_metrics["memory_copy_calls"], "count", "Nsight NVTX window"),
        ("Average Memory Copy Duration", nsys_metrics["average_memory_copy_duration_us"], "us", "Nsight NVTX window"),
    ]
    frame = pd.DataFrame(rows, columns=["metric", "value", "unit", "source"])
    frame.to_csv(output_dir / "profile_metrics_summary.csv", index=False)
    (output_dir / "profile_metrics_summary.txt").write_text(
        frame.to_string(index=False), encoding="utf-8"
    )


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

    current_hash = sha256_file(Path(__file__).resolve())
    if not smoke and not one_epoch:
        verify_prerequisite_marker(
            TEMP_PROFILE_DIR / "PROFILE_VALID.json", "smoke", current_hash
        )
        verify_prerequisite_marker(
            ONE_EPOCH_PROFILE_DIR / "PROFILE_VALID.json",
            "one_epoch_profile_test",
            current_hash,
        )

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
        nsys_raw_dir = output_dir / "nsight_systems_raw"
        timeline_worker_dir = output_dir / "nsight_timeline_worker_pass"
        timeline_raw_dir = output_dir / "nsight_timeline_raw"
        nsys_raw_dir.mkdir(parents=True, exist_ok=True)
        timeline_raw_dir.mkdir(parents=True, exist_ok=True)

        report_prefix = nsys_raw_dir / "yolov8m_nsys"
        report_path = Path(str(report_prefix) + ".nsys-rep")
        sqlite_path = nsys_raw_dir / "yolov8m_nsys.sqlite"
        timeline_prefix = timeline_raw_dir / "yolov8m_timeline"
        timeline_report = Path(str(timeline_prefix) + ".nsys-rep")

        parent_config = {
            "mode": run_mode,
            "workflow": "three_pass_single_command",
            "primary_profile_metrics_source": "pytorch_operator_pass",
            "operator_profiler": (
                "legacy torch.autograd CUDA-event timing, one context per batch, "
                "immediate aggregation"
            ),
            "kernel_profiler": (
                "Nsight Systems CUDA+NVTX full quantitative trace, parsed in "
                "streaming SQLite chunks"
            ),
            "representative_timeline": "separate fixed four-batch Nsight Systems trace",
            "passes_are_separate": True,
            "timeline_used_for_quantitative_metrics": False,
            "expected_nvidia_driver": EXPECTED_NVIDIA_DRIVER,
            "training_range_name": TRAINING_RANGE_NAME,
            "regular_script_sha256": sha256_file(REGULAR_SCRIPT),
            "profile_script_sha256": current_hash,
            "profile_script_revision": SCRIPT_REVISION,
            "log_size_cap_bytes": MAX_LOG_BYTES,
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        with (output_dir / "profile_workflow_config.json").open("w", encoding="utf-8") as file:
            json.dump(parent_config, file, indent=2)

        mode_args = []
        validation_mode = "official"
        if smoke:
            mode_args = ["--internal-smoke"]
            validation_mode = "smoke"
        elif one_epoch:
            mode_args = ["--internal-one-epoch"]
            validation_mode = "one_epoch"

        print("=" * 92)
        print(f"YOLOv8m NVIDIA profiled workflow ({display_mode})")
        print("=" * 92)
        print(f"Output directory: {output_dir}")
        print(
            "This performs three separate passes: full operator profiling, "
            "full Nsight quantitative tracing, then a four-batch timeline."
        )
        print()

        print("PHASE 1/3: PYTORCH OPERATOR PROFILING")
        print("-" * 92)
        torch_command = [
            sys.executable,
            str(PROFILE_SCRIPT),
            "--internal-worker",
            "torch_operator",
            "--internal-output",
            str(torch_dir),
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
            str(NSYS),
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            "--output",
            str(sqlite_path),
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
            str(NSYS),
            "profile",
            "--trace=cuda,nvtx",
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=true",
            "--output",
            str(timeline_prefix),
            sys.executable,
            str(PROFILE_SCRIPT),
            "--internal-worker",
            "nsight_systems",
            "--internal-output",
            str(timeline_worker_dir),
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

        root_primary_files = (
            "config.json",
            "epoch_metrics.csv",
            "telemetry.csv",
            "telemetry_raw.csv",
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
        for name in root_primary_files:
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

        build_profile_summary(output_dir, nsys_metrics)
        marker = {
            "status": "VALID",
            "mode": run_mode,
            "workflow": "three_pass_single_command",
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
        print("Primary performance metrics: PyTorch legacy CUDA-event operator pass")
        print(f"Top PyTorch operator: {top_operators.iloc[0]['operator']}")
        print(f"Top NVIDIA kernel: {top_kernels.iloc[0]['kernel_name']}")
        print(f"Total kernel launches: {nsys_metrics['total_kernel_launches']}")
        print(
            f"GPU kernel idle: {nsys_metrics['gpu_kernel_idle_time_s']:.6f} s "
            f"({nsys_metrics['gpu_kernel_idle_percent']:.3f}%)"
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
        description="YOLOv8m NVIDIA three-pass profiling workflow"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--one-epoch-test", action="store_true")
    mode.add_argument("--official", action="store_true")
    mode.add_argument(
        "--internal-worker",
        choices=("torch_operator", "nsight_systems"),
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

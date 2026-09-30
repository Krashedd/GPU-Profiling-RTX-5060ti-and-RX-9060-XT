#!/usr/bin/env python3

import argparse
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

import pandas as pd
import torch
import ultralytics
from torch.profiler import ProfilerActivity, profile
from ultralytics import YOLO


PROJECT_DIR = Path.home() / "Thesis-Testing"
SCRIPT_DIR = PROJECT_DIR / "scripts"
REGULAR_SCRIPT = SCRIPT_DIR / "yolov8m_regular_amd.py"
PROFILE_SCRIPT = SCRIPT_DIR / "yolov8m_profiled_amd.py"
PROFILE_DIR = PROJECT_DIR / "Logs" / "yoloprofiledrun"
GPU_ERROR_HISTORY = PROJECT_DIR / "Logs" / "yolo_gpu_error_history.log"
ROCPROFV3 = Path("/opt/rocm/bin/rocprofv3")
TRAINING_RANGE_NAME = "YOLOV8M_TRAINING"
PYTORCH_PROFILE_CHUNK_STEPS_OFFICIAL = 50
PYTORCH_PROFILE_CHUNK_STEPS_SMOKE = 4
ROCPROF_CSV_CHUNK_ROWS = 250_000

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
        "yolov8m_regular_amd_shared", REGULAR_SCRIPT
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

    required_paths = {
        "DATASET_ROOT": module.DATASET_ROOT,
        "DATA_YAML": module.DATA_YAML,
        "WEIGHTS_PATH": module.WEIGHTS_PATH,
    }
    missing = [f"{name}: {path}" for name, path in required_paths.items() if not Path(path).exists()]
    if missing:
        raise RuntimeError("Required YOLO files are missing:\n" + "\n".join(missing))

    return module


def append_gpu_error_history(label: str, traceback_text: str) -> None:
    GPU_ERROR_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with GPU_ERROR_HISTORY.open("a", encoding="utf-8") as file:
        file.write("\n" + "=" * 88 + "\n")
        file.write(f"Timestamp: {datetime.now().astimezone().isoformat()}\n")
        file.write(f"Workload: YOLOv8m COCO MiniTrain 10K AMD {label}\n")
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


class RoctxController:
    """Minimal ROCTx range wrapper used to mark the training window."""

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


def is_pytorch_operator(name: str) -> bool:
    return name.startswith(
        (
            "aten::",
            "Optimizer.",
            "autograd::engine::evaluate_function",
            "torch::autograd",
        )
    )


def operator_note(name: str) -> str:
    exact = {
        "aten::convolution_backward": "Convolution backward",
        "aten::miopen_convolution": "MIOpen convolution forward",
        "aten::miopen_batch_norm_backward": "MIOpen batch normalization backward",
        "aten::miopen_batch_norm": "MIOpen batch normalization forward",
        "aten::copy_": "Tensor copy",
        "aten::add_": "In-place tensor addition",
        "aten::_foreach_add_": "In-place multi-tensor addition",
        "aten::_foreach_add": "Multi-tensor addition",
        "Optimizer.step#SGD.step": "SGD optimizer update",
    }
    if name in exact:
        return exact[name]
    lowered = name.lower()
    if "convolution_backward" in lowered:
        return "Convolution backward"
    if "convolution" in lowered:
        return "Convolution forward"
    if "batch_norm_backward" in lowered:
        return "Batch normalization backward"
    if "batch_norm" in lowered:
        return "Batch normalization forward"
    if "upsample" in lowered:
        return "Upsampling"
    if "cat" in lowered:
        return "Tensor concatenation"
    if "copy" in lowered:
        return "Tensor copy"
    if "sgd" in lowered:
        return "SGD optimizer update"
    return ""


def event_device_times(event):
    self_us = getattr(event, "self_device_time_total", None)
    total_us = getattr(event, "device_time_total", None)
    if self_us is None:
        self_us = getattr(event, "self_cuda_time_total", 0.0)
    if total_us is None:
        total_us = getattr(event, "cuda_time_total", 0.0)
    return float(self_us or 0.0), float(total_us or 0.0)


def save_operator_outputs(profiler_object, output_dir: Path) -> pd.DataFrame:
    rows = []
    for event in profiler_object.key_averages():
        name = str(event.key)
        if not is_pytorch_operator(name):
            continue
        self_us, total_us = event_device_times(event)
        if self_us <= 0 and total_us <= 0:
            continue
        rows.append(
            {
                "operator": name,
                "self_gpu_time_ms": self_us / 1000.0,
                "gpu_total_ms": total_us / 1000.0,
                "calls": int(event.count),
                "notes": operator_note(name),
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
            table_text = profiler_object.key_averages().table(
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
    profiler_object.export_chrome_trace(str(output_dir / "pytorch_trace.json"))
    return frame


class ChunkedOperatorAccumulator:
    """Aggregate PyTorch operator statistics while releasing trace events in chunks."""

    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.totals = {}
        self.chunks_completed = 0
        self.representative_trace_exported = False

    def __call__(self, profiler_object):
        chunk_rows = 0
        for event in profiler_object.key_averages():
            name = str(event.key)
            if not is_pytorch_operator(name):
                continue
            self_us, total_us = event_device_times(event)
            if self_us <= 0 and total_us <= 0:
                continue
            row = self.totals.setdefault(
                name,
                {
                    "self_gpu_time_ms": 0.0,
                    "gpu_total_ms": 0.0,
                    "calls": 0,
                    "notes": operator_note(name),
                },
            )
            row["self_gpu_time_ms"] += self_us / 1000.0
            row["gpu_total_ms"] += total_us / 1000.0
            row["calls"] += int(event.count)
            chunk_rows += 1

        if chunk_rows == 0:
            return

        self.chunks_completed += 1
        self._build_frame().to_csv(
            self.output_dir / "pytorch_operator_chunk_progress.csv", index=False
        )

        if not self.representative_trace_exported:
            profiler_object.export_chrome_trace(
                str(self.output_dir / "pytorch_trace.json")
            )
            self.representative_trace_exported = True

        print(
            f"torch.profiler chunk {self.chunks_completed} flushed; "
            f"{len(self.totals)} cumulative PyTorch operators."
        )

    def _build_frame(self) -> pd.DataFrame:
        rows = []
        for name, values in self.totals.items():
            rows.append(
                {
                    "operator": name,
                    "self_gpu_time_ms": float(values["self_gpu_time_ms"]),
                    "gpu_total_ms": float(values["gpu_total_ms"]),
                    "calls": int(values["calls"]),
                    "notes": values["notes"],
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
        return pd.DataFrame(rows, columns=columns)

    def finalize(self) -> pd.DataFrame:
        frame = self._build_frame()
        frame.to_csv(self.output_dir / "pytorch_all_operators.csv", index=False)
        frame.head(10).to_csv(
            self.output_dir / "pytorch_top10_operators.csv", index=False
        )
        (self.output_dir / "pytorch_profiler_full_table.txt").write_text(
            frame.head(100).to_string(index=False), encoding="utf-8"
        )
        metadata = {
            "collection_mode": "chunked_full_workload",
            "chunks_completed": self.chunks_completed,
            "representative_trace_only": True,
            "representative_trace_file": "pytorch_trace.json",
            "all_operator_totals_cover_full_profiled_training": True,
        }
        with (self.output_dir / "pytorch_operator_chunk_metadata.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(metadata, file, indent=2)
        return frame


class ProfileTrainingState:
    def __init__(
        self,
        regular,
        monitor,
        epochs_to_run: int,
        profiler_object=None,
        roctx=None,
    ):
        self.regular = regular
        self.monitor = monitor
        self.epochs_to_run = epochs_to_run
        self.profiler_object = profiler_object
        self.roctx = roctx

        self.epoch_rows = []
        self.epoch_start_time = None
        self.training_start_time = None
        self.training_end_time = None
        self.dataset_size = None
        self.batches_per_epoch = None
        self.amp_enabled = None
        self.model_dtype = None
        self.window_started = False
        self.window_stopped = False
        self.profiler_started = False
        self.profiler_stopped = False
        self.monitor_started = False
        self.monitor_stopped = False

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
                f"Expected {self.regular.NUM_WORKERS} workers, "
                f"got {trainer.args.workers}."
            )
        if int(trainer.args.batch) != self.regular.BATCH_SIZE:
            raise RuntimeError(
                f"Expected batch {self.regular.BATCH_SIZE}, got {trainer.args.batch}."
            )

    def on_train_epoch_start(self, trainer):
        torch.cuda.synchronize()
        self.epoch_start_time = time.perf_counter()

        if not self.window_started:
            self.dataset_size = len(trainer.train_loader.dataset)
            self.batches_per_epoch = len(trainer.train_loader)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

            if self.profiler_object is not None:
                self.profiler_object.start()
                self.profiler_started = True
                print("torch.profiler collection started for training.")

            if self.roctx is not None:
                self.roctx.push(TRAINING_RANGE_NAME)
                print(f"ROCTx training range started: {TRAINING_RANGE_NAME}")

            self.monitor.start()
            self.monitor_started = True
            self.training_start_time = self.epoch_start_time
            self.window_started = True
            print("Monitored profiled training started.")

    def on_train_batch_end(self, trainer):
        if self.profiler_started and not self.profiler_stopped:
            self.profiler_object.step()

    def on_train_epoch_end(self, trainer):
        torch.cuda.synchronize()
        epoch_end = time.perf_counter()
        epoch_time = epoch_end - self.epoch_start_time
        samples = int(self.dataset_size)
        batches = int(self.batches_per_epoch)
        throughput = samples / epoch_time

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
            "epoch_time_s": epoch_time,
            "throughput_images_s": throughput,
            "samples": samples,
            "batches": batches,
            **losses,
        }
        self.epoch_rows.append(row)

        loss_text = " | ".join(
            f"{key}: {value:.4f}" for key, value in losses.items()
        )
        if loss_text:
            loss_text = " | " + loss_text

        print(
            f"Epoch {trainer.epoch + 1}/{trainer.args.epochs} | "
            f"Time: {epoch_time:.2f}s | "
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

        if self.roctx is not None:
            self.roctx.pop()

        if self.profiler_started and not self.profiler_stopped:
            self.profiler_object.stop()
            self.profiler_stopped = True

        if self.monitor_started and not self.monitor_stopped:
            self.monitor.stop()
            self.monitor_stopped = True

        if self.window_started:
            self.window_stopped = True

    @property
    def total_training_time(self) -> float:
        if self.training_start_time is None or self.training_end_time is None:
            return float("nan")
        return self.training_end_time - self.training_start_time


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
    state = None
    profiler_object = None
    operator_accumulator = None
    roctx = None

    try:
        os.environ.setdefault("WANDB_DISABLED", "true")
        os.environ.setdefault("COMET_DISABLE_AUTO_LOGGING", "1")
        regular.set_seed(regular.SEED)
        counts = regular.verify_dataset()

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot detect the AMD GPU.")
        if torch.version.hip is None:
            raise RuntimeError("This PyTorch build does not report a HIP runtime.")

        gpu_name = torch.cuda.get_device_name(0)
        gpu_arch = torch.cuda.get_device_properties(0).gcnArchName
        if "9060 XT" not in gpu_name:
            raise RuntimeError(
                f"Unexpected GPU selected: {gpu_name}. Expected RX 9060 XT."
            )

        epochs_to_run = 1 if smoke else regular.EPOCHS
        if smoke:
            selected_yaml, training_images, validation_images = regular.create_smoke_yaml(
                output_dir
            )
        else:
            selected_yaml = regular.DATA_YAML
            training_images = counts["train_images"]
            validation_images = counts["val_images"]

        print("=" * 88)
        print(
            f"YOLOv8m COCO MiniTrain 10K AMD {profile_kind} profiling "
            f"{'smoke test' if smoke else 'official pass'}"
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
        print(f"HIP runtime: {torch.version.hip}")
        print(f"System ROCm: {regular.read_rocm_version()}")
        print(f"Ultralytics: {ultralytics.__version__}")
        print(f"GPU: {gpu_name}")
        print(f"GPU architecture: {gpu_arch}")
        print()

        if profile_kind == "torch_operator":
            chunk_steps = (
                PYTORCH_PROFILE_CHUNK_STEPS_SMOKE
                if smoke
                else PYTORCH_PROFILE_CHUNK_STEPS_OFFICIAL
            )
            operator_accumulator = ChunkedOperatorAccumulator(output_dir)
            profiler_object = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
                schedule=torch.profiler.schedule(
                    wait=0,
                    warmup=0,
                    active=chunk_steps,
                    repeat=(1 if smoke else 125),
                ),
                on_trace_ready=operator_accumulator,
                acc_events=False,
            )
            print(
                f"torch.profiler will flush operator events every "
                f"{chunk_steps} training batches."
            )
        elif profile_kind == "rocprofv3":
            roctx = RoctxController()
            print(f"ROCTx library: {roctx.library_path}")
            print()
        else:
            raise RuntimeError(f"Unknown profile kind: {profile_kind}")

        config = {
            "mode": "smoke" if smoke else "official_profiled",
            "profile_kind": profile_kind,
            "seed": regular.SEED,
            "model": "YOLOv8m",
            "weights": str(regular.WEIGHTS_PATH),
            "dataset": "COCO 2017 MiniTrain 10K",
            "dataset_source": "Kaggle banuprasadb/coco-minitrain-10k",
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
            "device": 0,
            "telemetry_interval_s": regular.SAMPLE_INTERVAL_S,
            "torch_version": torch.__version__,
            "hip_runtime": torch.version.hip,
            "system_rocm": regular.read_rocm_version(),
            "ultralytics_version": ultralytics.__version__,
            "gpu": gpu_name,
            "gpu_architecture": gpu_arch,
            "training_range_name": TRAINING_RANGE_NAME,
            "pytorch_profile_collection": "chunked_full_workload",
            "pytorch_profile_chunk_steps": (
                PYTORCH_PROFILE_CHUNK_STEPS_SMOKE
                if smoke
                else PYTORCH_PROFILE_CHUNK_STEPS_OFFICIAL
            ),
            "script_revision": "2026-06-16 chunked-profiler-disk-sorted-intervals-v3",
        }
        with (output_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

        monitor = regular.TelemetryMonitor(
            interval_s=regular.SAMPLE_INTERVAL_S
        )
        state = ProfileTrainingState(
            regular=regular,
            monitor=monitor,
            epochs_to_run=epochs_to_run,
            profiler_object=profiler_object,
            roctx=roctx,
        )

        model = YOLO(str(regular.WEIGHTS_PATH))
        model.add_callback("on_train_start", state.on_train_start)
        model.add_callback("on_train_epoch_start", state.on_train_epoch_start)
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
        total_training_time = state.total_training_time
        if math.isnan(total_training_time) or total_training_time <= 0:
            raise RuntimeError("Training callbacks did not produce valid timing.")

        if profiler_object is not None:
            if operator_accumulator is None:
                raise RuntimeError("PyTorch operator accumulator was not initialized.")
            operator_frame = operator_accumulator.finalize()
            if operator_frame.empty:
                raise RuntimeError("PyTorch operator table is empty.")

        epoch_df = pd.DataFrame(state.epoch_rows)
        epoch_df.to_csv(output_dir / "epoch_metrics.csv", index=False)

        telemetry_df = monitor.dataframe()
        telemetry_df.to_csv(output_dir / "telemetry.csv", index=False)
        regular.validate_telemetry(telemetry_df)
        with (output_dir / "amdsmi_api_errors.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(monitor.api_errors, file, indent=2)

        if len(state.epoch_rows) != epochs_to_run:
            raise RuntimeError(
                f"Incomplete run: expected {epochs_to_run} epochs, "
                f"got {len(state.epoch_rows)}."
            )

        batches_per_epoch = int(state.batches_per_epoch)
        total_batches = batches_per_epoch * epochs_to_run
        total_images = training_images * epochs_to_run
        expected_batches = math.ceil(training_images / regular.BATCH_SIZE) * epochs_to_run
        expected_images = training_images * epochs_to_run
        if total_batches != expected_batches:
            raise RuntimeError(
                f"Expected {expected_batches} batches, got {total_batches}."
            )
        if total_images != expected_images:
            raise RuntimeError(
                f"Expected {expected_images} processed images, got {total_images}."
            )

        last_weights = train_output_dir / "weights" / "last.pt"
        if not last_weights.exists():
            raise RuntimeError(
                f"Expected trained checkpoint was not created: {last_weights}"
            )

        print()
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

        throughput = total_images / total_training_time
        batch_latency_ms = total_training_time / total_batches * 1000.0
        average_epoch_time = statistics.mean(
            row["epoch_time_s"] for row in state.epoch_rows
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
        peak_edge_temp_c = regular.series_stat(
            telemetry_df, "gpu_edge_temperature_c", "max"
        )
        peak_hotspot_temp_c = regular.series_stat(
            telemetry_df, "gpu_hotspot_temperature_c", "max"
        )
        peak_memory_temp_c = regular.series_stat(
            telemetry_df, "gpu_memory_temperature_c", "max"
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
        torch_peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)

        performance_per_watt = (
            throughput / average_power_w
            if not math.isnan(average_power_w) and average_power_w > 0
            else float("nan")
        )
        temperature_limit_seen = (
            (not math.isnan(peak_edge_temp_c) and peak_edge_temp_c >= 100.0)
            or (
                not math.isnan(peak_hotspot_temp_c)
                and peak_hotspot_temp_c >= 110.0
            )
            or (
                not math.isnan(peak_memory_temp_c)
                and peak_memory_temp_c >= 105.0
            )
        )

        stability_notes = (
            "Completed successfully; no Python, HIP, ROCm, Ultralytics, "
            "AMD SMI, profiler, or GPU fatal errors detected."
        )
        if temperature_limit_seen:
            stability_notes += (
                " A monitored GPU temperature reached its configured "
                "critical threshold."
            )
        else:
            stability_notes += (
                " No monitored GPU temperature reached the configured "
                "critical thresholds."
            )
        if monitor.api_errors:
            error_count = sum(item["count"] for item in monitor.api_errors.values())
            stability_notes += (
                f" AMD SMI had {error_count} recoverable telemetry API errors."
            )
        else:
            stability_notes += " All AMD SMI telemetry queries succeeded."

        summary = {
            "mode": "smoke" if smoke else "official_profiled",
            "profile_kind": profile_kind,
            "throughput_images_s": throughput,
            "batch_latency_ms_batch": batch_latency_ms,
            "total_training_time_s": total_training_time,
            "average_epoch_time_s": average_epoch_time,
            "average_gpu_power_w": average_power_w,
            "performance_per_watt_images_s_w": performance_per_watt,
            "average_gpu_util_percent": average_gpu_util,
            "average_vram_usage_mb": average_vram_mb,
            "peak_vram_usage_mb": peak_vram_mb,
            "torch_peak_allocated_memory_mb": torch_peak_allocated_mb,
            "average_gpu_edge_temperature_c": average_edge_temp_c,
            "average_gpu_hotspot_temperature_c": average_hotspot_temp_c,
            "average_gpu_memory_temperature_c": average_memory_temp_c,
            "peak_gpu_edge_temperature_c": peak_edge_temp_c,
            "peak_gpu_hotspot_temperature_c": peak_hotspot_temp_c,
            "peak_gpu_memory_temperature_c": peak_memory_temp_c,
            "temperature_critical_threshold_reached": temperature_limit_seen,
            "final_validation_map50_95": validation["map50_95"],
            "final_validation_map50": validation["map50"],
            "final_validation_map75": validation["map75"],
            "final_validation_mean_precision": validation["mean_precision"],
            "final_validation_mean_recall": validation["mean_recall"],
            "validation_images": validation_images,
            "peak_system_ram_mb": peak_system_ram_mb,
            "average_system_ram_mb": average_system_ram_mb,
            "average_cpu_util_percent": average_cpu_util,
            "average_cpu_package_temperature_c": average_cpu_temp_c,
            "average_disk_read_mb_s": average_disk_read_mb_s,
            "average_disk_write_mb_s": average_disk_write_mb_s,
            "peak_disk_read_mb_s": peak_disk_read_mb_s,
            "peak_disk_write_mb_s": peak_disk_write_mb_s,
            "total_images_processed": total_images,
            "total_batches_processed": total_batches,
            "telemetry_samples": len(telemetry_df),
            "telemetry_interval_s": regular.SAMPLE_INTERVAL_S,
            "workers": regular.NUM_WORKERS,
            "amp_enabled": bool(state.amp_enabled),
            "model_parameter_dtype": state.model_dtype,
            "training_range_name": TRAINING_RANGE_NAME,
            "stability_error_notes": stability_notes,
        }
        pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)

        with (output_dir / "summary.txt").open("w", encoding="utf-8") as file:
            file.write(
                f"YOLOv8m AMD {'Smoke' if smoke else 'Official'} "
                f"{profile_kind} Profile Pass\n"
            )
            file.write("=" * 72 + "\n")
            file.write(f"Throughput: {throughput:.2f} images/s\n")
            file.write(f"Batch Latency: {batch_latency_ms:.2f} ms/batch\n")
            file.write(f"Total Training Time: {total_training_time:.2f} s\n")
            file.write(f"Average Epoch Time: {average_epoch_time:.2f} s\n")
            file.write(f"Average GPU Power Draw: {average_power_w:.2f} W\n")
            file.write(
                f"Performance per Watt: {performance_per_watt:.4f} images/s/W\n"
            )
            file.write(f"Average GPU Utilization: {average_gpu_util:.2f} %\n")
            file.write(f"Average VRAM Usage: {average_vram_mb:.2f} MB\n")
            file.write(f"Peak VRAM Usage: {peak_vram_mb:.2f} MB\n")
            file.write(
                f"Average GPU Edge Temperature: {average_edge_temp_c:.2f} C\n"
            )
            file.write(
                f"Average GPU Hotspot Temperature: {average_hotspot_temp_c:.2f} C\n"
            )
            file.write(
                f"Average GPU Memory Temperature: {average_memory_temp_c:.2f} C\n"
            )
            file.write(
                f"Final Validation mAP50-95: {validation['map50_95']:.4f}\n"
            )
            file.write(f"Final Validation mAP50: {validation['map50']:.4f}\n")
            file.write(
                f"Final Validation Precision: {validation['mean_precision']:.4f}\n"
            )
            file.write(
                f"Final Validation Recall: {validation['mean_recall']:.4f}\n"
            )
            file.write(f"Peak System RAM: {peak_system_ram_mb:.2f} MB\n")
            file.write(f"Average System RAM: {average_system_ram_mb:.2f} MB\n")
            file.write(f"Average CPU Utilization: {average_cpu_util:.2f} %\n")
            file.write(
                f"Average CPU Package Temperature: {average_cpu_temp_c:.2f} C\n"
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
            file.write("Precision: FP32; AMP enabled: False\n")
            file.write(f"Workers: {regular.NUM_WORKERS}\n")
            file.write(f"Stability / Error Notes: {stability_notes}\n")

        worker_valid = {
            "status": "VALID",
            "timestamp": datetime.now().astimezone().isoformat(),
            "mode": "smoke" if smoke else "official_profiled",
            "profile_kind": profile_kind,
            "workers": regular.NUM_WORKERS,
            "amp_enabled": bool(state.amp_enabled),
            "total_images_processed": total_images,
            "total_batches_processed": total_batches,
        }
        with (output_dir / "WORKER_VALID.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(worker_valid, file, indent=2)

        print()
        print("=" * 88)
        print("PROFILE WORKER VALID")
        print("=" * 88)
        print(f"Profile kind: {profile_kind}")
        print(f"Throughput: {throughput:.2f} images/s")
        print(f"Batch Latency: {batch_latency_ms:.2f} ms/batch")
        print(f"Total Training Time: {total_training_time:.2f} s")
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

        if monitor is not None:
            try:
                partial = monitor.dataframe()
                if not partial.empty:
                    partial.to_csv(
                        output_dir / "telemetry_partial.csv", index=False
                    )
                with (output_dir / "amdsmi_api_errors.json").open(
                    "w", encoding="utf-8"
                ) as file:
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


def _read_marker_window(marker_paths) -> tuple[int, int]:
    marker_df = read_trace_csvs(marker_paths)
    if marker_df.empty:
        raise RuntimeError("rocprofv3 marker trace is empty.")

    required = {"Function", "Start_Timestamp", "End_Timestamp"}
    missing = required.difference(marker_df.columns)
    if missing:
        raise RuntimeError(
            "Marker trace is missing columns: " + ", ".join(sorted(missing))
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

    marker_matches["Start_Timestamp"] = pd.to_numeric(
        marker_matches["Start_Timestamp"], errors="coerce"
    )
    marker_matches["End_Timestamp"] = pd.to_numeric(
        marker_matches["End_Timestamp"], errors="coerce"
    )
    marker_matches = marker_matches.dropna(
        subset=["Start_Timestamp", "End_Timestamp"]
    )
    marker_matches["duration_ns"] = (
        marker_matches["End_Timestamp"]
        - marker_matches["Start_Timestamp"]
    )
    marker_row = marker_matches.sort_values(
        "duration_ns", ascending=False
    ).iloc[0]
    training_start = int(marker_row["Start_Timestamp"])
    training_end = int(marker_row["End_Timestamp"])
    if training_end <= training_start:
        raise RuntimeError("The ROCTx training marker has an invalid duration.")
    return training_start, training_end


def _iter_csv_chunks(paths, chunksize: int = ROCPROF_CSV_CHUNK_ROWS):
    for path in paths:
        try:
            reader = pd.read_csv(path, chunksize=chunksize)
            for chunk in reader:
                yield path, chunk
        except pd.errors.EmptyDataError:
            continue


def parse_rocprof_outputs(raw_dir: Path, destination: Path) -> dict:
    """Parse full-run rocprofv3 CSVs without loading the entire trace into RAM."""

    kernel_paths = find_trace_files(raw_dir, "kernel_trace.csv")
    marker_paths = find_trace_files(raw_dir, "marker_api_trace.csv")
    memory_copy_paths = find_trace_files(raw_dir, "memory_copy_trace.csv")

    if not kernel_paths:
        raise RuntimeError("rocprofv3 produced no kernel_trace.csv file.")
    if not marker_paths:
        raise RuntimeError("rocprofv3 produced no marker_api_trace.csv file.")

    training_start, training_end = _read_marker_window(marker_paths)
    training_duration_ns = float(training_end - training_start)

    aggregate = {}
    raw_kernel_rows = 0
    training_kernel_rows = 0

    # rocprofv3 does not guarantee that kernel rows are emitted in timestamp
    # order.  Exact busy/idle time therefore requires a disk-backed sort of
    # the filtered intervals before computing their union.  This keeps RAM
    # bounded even for a full 10-epoch trace.
    unsorted_intervals = destination / ".kernel_intervals_unsorted.tsv"
    sorted_intervals = destination / ".kernel_intervals_sorted.tsv"
    for temporary_path in (unsorted_intervals, sorted_intervals):
        temporary_path.unlink(missing_ok=True)

    required_kernel_columns = {
        "Kernel_Name",
        "Start_Timestamp",
        "End_Timestamp",
    }

    try:
        with unsorted_intervals.open("w", encoding="utf-8", buffering=1024 * 1024) as interval_file:
            for source_path, chunk in _iter_csv_chunks(kernel_paths):
                raw_kernel_rows += int(len(chunk))
                missing = required_kernel_columns.difference(chunk.columns)
                if missing:
                    raise RuntimeError(
                        f"Kernel trace {source_path} is missing columns: "
                        + ", ".join(sorted(missing))
                    )

                work = chunk[["Kernel_Name", "Start_Timestamp", "End_Timestamp"]].copy()
                work["Start_Timestamp"] = pd.to_numeric(
                    work["Start_Timestamp"], errors="coerce"
                )
                work["End_Timestamp"] = pd.to_numeric(
                    work["End_Timestamp"], errors="coerce"
                )
                work = work.dropna(
                    subset=["Kernel_Name", "Start_Timestamp", "End_Timestamp"]
                )
                work = work[
                    (work["Start_Timestamp"] >= training_start)
                    & (work["End_Timestamp"] <= training_end)
                ].copy()
                if work.empty:
                    continue

                work["Start_Timestamp"] = work["Start_Timestamp"].astype("int64")
                work["End_Timestamp"] = work["End_Timestamp"].astype("int64")
                work["duration_ns"] = (
                    work["End_Timestamp"] - work["Start_Timestamp"]
                )
                work = work[work["duration_ns"] >= 0]
                if work.empty:
                    continue

                training_kernel_rows += int(len(work))

                # Write only the two integer columns needed for the exact
                # interval union.  Sorting is performed by GNU sort on disk.
                work[["Start_Timestamp", "End_Timestamp"]].to_csv(
                    interval_file,
                    sep="\t",
                    header=False,
                    index=False,
                    lineterminator="\n",
                )

                work["duration_sq"] = work["duration_ns"].astype("float64") ** 2
                grouped = work.groupby("Kernel_Name", dropna=False)["duration_ns"].agg(
                    ["count", "sum", "min", "max"]
                )
                sq_grouped = work.groupby("Kernel_Name", dropna=False)["duration_sq"].sum()

                for kernel_name, row in grouped.iterrows():
                    key = str(kernel_name)
                    values = aggregate.setdefault(
                        key,
                        {
                            "count": 0,
                            "sum": 0.0,
                            "sumsq": 0.0,
                            "min": float("inf"),
                            "max": float("-inf"),
                        },
                    )
                    values["count"] += int(row["count"])
                    values["sum"] += float(row["sum"])
                    values["sumsq"] += float(sq_grouped.loc[kernel_name])
                    values["min"] = min(values["min"], float(row["min"]))
                    values["max"] = max(values["max"], float(row["max"]))

        if training_kernel_rows == 0 or not aggregate:
            raise RuntimeError("No kernels were found inside the ROCTx training range.")

        sort_env = os.environ.copy()
        sort_env["LC_ALL"] = "C"
        sort_command = [
            "sort",
            "-k1,1n",
            "-k2,2n",
            "--buffer-size=25%",
            "--temporary-directory",
            str(destination),
            "--output",
            str(sorted_intervals),
            str(unsorted_intervals),
        ]
        sort_result = subprocess.run(
            sort_command,
            env=sort_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if sort_result.returncode != 0:
            raise RuntimeError(
                "Disk-backed timestamp sort failed: "
                + (sort_result.stderr.strip() or f"exit code {sort_result.returncode}")
            )

        current_union_start = None
        current_union_end = None
        busy_union_ns = 0.0
        sorted_interval_rows = 0

        with sorted_intervals.open("r", encoding="utf-8") as interval_file:
            for line_number, line in enumerate(interval_file, start=1):
                fields = line.rstrip("\n").split("\t")
                if len(fields) != 2:
                    raise RuntimeError(
                        f"Malformed sorted interval at line {line_number}: {line!r}"
                    )
                start_ns = int(fields[0])
                end_ns = int(fields[1])
                sorted_interval_rows += 1

                if current_union_start is None:
                    current_union_start = start_ns
                    current_union_end = end_ns
                elif start_ns > current_union_end:
                    busy_union_ns += float(current_union_end - current_union_start)
                    current_union_start = start_ns
                    current_union_end = end_ns
                elif end_ns > current_union_end:
                    current_union_end = end_ns

        if sorted_interval_rows != training_kernel_rows:
            raise RuntimeError(
                "The disk-backed interval sort changed the number of kernel rows: "
                f"expected {training_kernel_rows}, found {sorted_interval_rows}."
            )

        if current_union_start is not None:
            busy_union_ns += float(current_union_end - current_union_start)
    finally:
        unsorted_intervals.unlink(missing_ok=True)
        sorted_intervals.unlink(missing_ok=True)

    rows = []
    total_kernel_duration_ns = sum(values["sum"] for values in aggregate.values())
    for kernel_name, values in aggregate.items():
        count = int(values["count"])
        total_ns = float(values["sum"])
        mean_ns = total_ns / count if count else float("nan")
        if count > 1:
            variance = max(
                0.0,
                (values["sumsq"] - (total_ns * total_ns) / count) / (count - 1),
            )
            stddev_ns = math.sqrt(variance)
        else:
            stddev_ns = float("nan")
        rows.append(
            {
                "kernel_name": kernel_name,
                "calls": count,
                "total_duration_ns": total_ns,
                "average_duration_ns": mean_ns,
                "minimum_duration_ns": values["min"],
                "maximum_duration_ns": values["max"],
                "stddev_duration_ns": stddev_ns,
            }
        )

    rows.sort(key=lambda row: row["total_duration_ns"], reverse=True)
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
        row["total_duration_ms"] = row["total_duration_ns"] / 1e6
        row["average_duration_us"] = row["average_duration_ns"] / 1e3
        row["percentage"] = (
            row["total_duration_ns"] / total_kernel_duration_ns * 100.0
            if total_kernel_duration_ns > 0
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
    grouped_frame = pd.DataFrame(rows)[columns]
    grouped_frame.to_csv(
        destination / "rocprofv3_parsed_summary.csv", index=False
    )
    grouped_frame.head(5).to_csv(
        destination / "amd_top5_kernels.csv", index=False
    )

    memory_copy_calls = 0
    memory_copy_total_ns = 0.0
    required_copy_columns = {"Start_Timestamp", "End_Timestamp"}
    for source_path, chunk in _iter_csv_chunks(memory_copy_paths):
        missing = required_copy_columns.difference(chunk.columns)
        if missing:
            raise RuntimeError(
                f"Memory-copy trace {source_path} is missing columns: "
                + ", ".join(sorted(missing))
            )
        work = chunk[["Start_Timestamp", "End_Timestamp"]].copy()
        work["Start_Timestamp"] = pd.to_numeric(
            work["Start_Timestamp"], errors="coerce"
        )
        work["End_Timestamp"] = pd.to_numeric(
            work["End_Timestamp"], errors="coerce"
        )
        work = work.dropna(subset=["Start_Timestamp", "End_Timestamp"])
        work = work[
            (work["Start_Timestamp"] >= training_start)
            & (work["End_Timestamp"] <= training_end)
        ].copy()
        if work.empty:
            continue
        work["duration_ns"] = (
            work["End_Timestamp"] - work["Start_Timestamp"]
        )
        work = work[work["duration_ns"] >= 0]
        memory_copy_calls += int(len(work))
        memory_copy_total_ns += float(work["duration_ns"].sum())

    memory_copy_total_ms = (
        memory_copy_total_ns / 1e6 if memory_copy_calls else float("nan")
    )
    memory_copy_average_us = (
        memory_copy_total_ns / memory_copy_calls / 1e3
        if memory_copy_calls
        else float("nan")
    )
    pd.DataFrame(
        [
            {
                "calls": memory_copy_calls,
                "total_duration_ms": memory_copy_total_ms,
                "average_duration_us": memory_copy_average_us,
            }
        ]
    ).to_csv(destination / "rocprofv3_memory_copy_summary.csv", index=False)

    gpu_idle_ns = max(0.0, training_duration_ns - busy_union_ns)
    gpu_idle_percent = (
        gpu_idle_ns / training_duration_ns * 100.0
        if training_duration_ns > 0
        else float("nan")
    )

    metadata = {
        "parser_mode": "streaming_chunked_csv_with_disk_backed_interval_sort",
        "csv_chunk_rows": ROCPROF_CSV_CHUNK_ROWS,
        "kernel_trace_files": [str(path) for path in kernel_paths],
        "marker_trace_files": [str(path) for path in marker_paths],
        "memory_copy_trace_files": [str(path) for path in memory_copy_paths],
        "training_marker": TRAINING_RANGE_NAME,
        "training_start_timestamp_ns": training_start,
        "training_end_timestamp_ns": training_end,
        "training_marker_duration_s": training_duration_ns / 1e9,
        "raw_kernel_rows": raw_kernel_rows,
        "training_kernel_rows": training_kernel_rows,
        "unique_training_kernels": int(len(grouped_frame)),
        "sum_training_kernel_duration_ms": total_kernel_duration_ns / 1e6,
        "gpu_busy_time_union_ms": busy_union_ns / 1e6,
        "gpu_idle_time_ms": gpu_idle_ns / 1e6,
        "gpu_idle_percent": gpu_idle_percent,
        "kernel_trace_ordering": "disk_sorted_by_start_then_end_timestamp",
        "memory_copy_calls": memory_copy_calls,
        "memory_copy_total_ms": memory_copy_total_ms,
        "memory_copy_average_us": memory_copy_average_us,
        "training_window_filter_applied": True,
    }
    with (destination / "rocprofv3_parse_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(metadata, file, indent=2)

    return metadata


def build_profile_metrics_summary(output_dir: Path, parse_metadata: dict) -> None:
    primary = pd.read_csv(output_dir / "summary.csv").iloc[0]
    operators = pd.read_csv(output_dir / "pytorch_top10_operators.csv")
    kernels = pd.read_csv(output_dir / "amd_top5_kernels.csv")

    top_operator = str(operators.iloc[0]["operator"])
    top_kernel = str(kernels.iloc[0]["kernel_name"])
    top_kernel_duration_ms = float(kernels.iloc[0]["total_duration_ms"])

    util = float(primary["average_gpu_util_percent"])
    idle_percent = float(parse_metadata["gpu_idle_percent"])
    if util >= 90.0 and idle_percent <= 10.0:
        bottleneck = "GPU compute-bound (utilization/idle-time heuristic)"
    elif idle_percent > 25.0:
        bottleneck = "Pipeline/input-limited (idle-time heuristic)"
    else:
        bottleneck = "Mixed/undetermined"

    rows = [
        ("Throughput", float(primary["throughput_images_s"]), "images/s", "PyTorch profiled full run"),
        ("Batch Latency", float(primary["batch_latency_ms_batch"]), "ms/batch", "PyTorch profiled full run"),
        ("Total Training Time", float(primary["total_training_time_s"]), "s", "PyTorch profiled full run"),
        ("Average Epoch Time", float(primary["average_epoch_time_s"]), "s", "PyTorch profiled full run"),
        ("Average GPU Power Draw", float(primary["average_gpu_power_w"]), "W", "PyTorch profiled full run"),
        ("Performance per Watt", float(primary["performance_per_watt_images_s_w"]), "images/s/W", "PyTorch profiled full run"),
        ("Average GPU Utilization", util, "%", "PyTorch profiled full run"),
        ("Average VRAM Usage", float(primary["average_vram_usage_mb"]), "MB", "PyTorch profiled full run"),
        ("Peak VRAM Usage", float(primary["peak_vram_usage_mb"]), "MB", "PyTorch profiled full run"),
        ("Average GPU Temperature", float(primary["average_gpu_edge_temperature_c"]), "C", "PyTorch profiled full run"),
        ("Top PyTorch Operator", top_operator, "", "Chunk-aggregated all 10 epochs"),
        ("Top AMD Kernel", top_kernel, "", "rocprofv3 full run"),
        ("Top Kernel Duration", top_kernel_duration_ms, "ms cumulative", "rocprofv3 full run"),
        ("Kernel Launch Count", int(parse_metadata["training_kernel_rows"]), "dispatches", "rocprofv3 full run"),
        ("GPU Idle Time", float(parse_metadata["gpu_idle_time_ms"]) / 1000.0, "s", "Exact union of full-run kernel intervals"),
        ("GPU Idle Percentage", idle_percent, "%", "Exact union of full-run kernel intervals"),
        ("Bottleneck Type", bottleneck, "", "Heuristic; interpret with operators/kernels"),
        ("Final Validation Score", float(primary["final_validation_map50_95"]), "mAP50-95", "PyTorch profiled full run"),
        ("Average GPU Memory Copy Time", float(parse_metadata["memory_copy_average_us"]), "us/copy", "rocprofv3 full run"),
        ("Peak System RAM", float(primary["peak_system_ram_mb"]), "MB", "PyTorch profiled full run"),
        ("Average System RAM", float(primary["average_system_ram_mb"]), "MB", "PyTorch profiled full run"),
        ("CPU Utilization", float(primary["average_cpu_util_percent"]), "%", "PyTorch profiled full run"),
        ("Average Disk Read", float(primary["average_disk_read_mb_s"]), "MB/s", "PyTorch profiled full run"),
        ("Average Disk Write", float(primary["average_disk_write_mb_s"]), "MB/s", "PyTorch profiled full run"),
        ("Peak Disk Read", float(primary["peak_disk_read_mb_s"]), "MB/s", "PyTorch profiled full run"),
        ("Peak Disk Write", float(primary["peak_disk_write_mb_s"]), "MB/s", "PyTorch profiled full run"),
    ]
    frame = pd.DataFrame(rows, columns=["metric", "value", "unit", "source"])
    frame.to_csv(output_dir / "profile_metrics_summary.csv", index=False)

    with (output_dir / "profile_metrics_summary.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write("YOLOv8m AMD Consolidated Profile Metrics\n")
        file.write("=" * 72 + "\n")
        for row in rows:
            metric, value, unit, source = row
            suffix = f" {unit}" if unit else ""
            file.write(f"{metric}: {value}{suffix} [{source}]\n")


def validate_worker(worker_dir: Path, kind: str, smoke: bool) -> None:
    marker_path = worker_dir / "WORKER_VALID.json"
    if not marker_path.is_file():
        raise RuntimeError(f"{kind} worker did not produce WORKER_VALID.json")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if marker.get("status") != "VALID":
        raise RuntimeError(f"{kind} worker status is not VALID: {marker}")

    summary_path = worker_dir / "summary.csv"
    if not summary_path.is_file():
        raise RuntimeError(f"{kind} worker summary.csv is missing.")
    summary = pd.read_csv(summary_path).iloc[0]

    expected_images = 64 if smoke else 100000
    expected_batches = 4 if smoke else 6250
    if int(summary["total_images_processed"]) != expected_images:
        raise RuntimeError(
            f"{kind} worker processed {summary['total_images_processed']} "
            f"images; expected {expected_images}."
        )
    if int(summary["total_batches_processed"]) != expected_batches:
        raise RuntimeError(
            f"{kind} worker processed {summary['total_batches_processed']} "
            f"batches; expected {expected_batches}."
        )
    if bool(summary["amp_enabled"]):
        raise RuntimeError(f"{kind} worker unexpectedly enabled AMP.")
    if int(summary["workers"]) != 4:
        raise RuntimeError(f"{kind} worker did not use four workers.")
    if not math.isfinite(float(summary["final_validation_map50_95"])):
        raise RuntimeError(f"{kind} worker produced an invalid mAP50-95.")


def run_parent(smoke: bool) -> None:
    output_dir = (
        Path("/tmp/yolov8m_amd_profile_smoke") if smoke else PROFILE_DIR
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
        timeline_worker_dir = output_dir / "rocprofv3_timeline_worker_pass"
        timeline_raw_dir = output_dir / "rocprofv3_timeline_raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        timeline_raw_dir.mkdir(parents=True, exist_ok=True)

        parent_config = {
            "mode": "smoke" if smoke else "official_profiled",
            "workflow": "three_pass_single_command",
            "primary_profile_metrics_source": "pytorch_operator_pass",
            "operator_profiler": "torch.profiler chunked over complete workload",
            "kernel_profiler": (
                "rocprofv3 kernel/memory-copy/marker traces with ROCTx "
                "training-window filtering"
            ),
            "representative_timeline": (
                "fixed four-batch rocprofv3 Perfetto trace for visualization only"
            ),
            "reason_for_separate_passes": (
                "PyTorch operator profiling and rocprofv3 tracing are separated "
                "to avoid profiler interference; the short third pass prevents "
                "Perfetto ring-buffer truncation while retaining a readable timeline."
            ),
            "both_quantitative_passes_use_identical_fixed_workload": True,
            "timeline_used_for_quantitative_metrics": False,
            "training_range_name": TRAINING_RANGE_NAME,
            "pytorch_chunk_steps": (
                PYTORCH_PROFILE_CHUNK_STEPS_SMOKE
                if smoke
                else PYTORCH_PROFILE_CHUNK_STEPS_OFFICIAL
            ),
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        with (output_dir / "profile_workflow_config.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(parent_config, file, indent=2)

        worker_smoke_arg = ["--internal-smoke"] if smoke else []

        print("=" * 92)
        print(
            "YOLOv8m AMD profiled workflow "
            f"({'smoke test' if smoke else 'official'})"
        )
        print("=" * 92)
        print(f"Output directory: {output_dir}")
        print(
            "This command performs three controlled passes: full PyTorch "
            "operator profiling, full rocprofv3 aggregate tracing, then a "
            "short representative Perfetto timeline."
        )
        print()

        print("PHASE 1/3: PYTORCH OPERATOR PROFILING (FULL WORKLOAD)")
        print("-" * 92)
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

        print("PHASE 2/3: ROCPROFV3 AGGREGATE TRACING (FULL WORKLOAD)")
        print("-" * 92)
        kernel_command = [
            str(ROCPROFV3),
            "--disable-signal-handlers",
            "--kernel-trace",
            "--memory-copy-trace",
            "--marker-trace",
            "--stats",
            "--summary",
            "--summary-units",
            "msec",
            "--summary-output-file",
            "rocprofv3_summary",
            "--output-format",
            "csv",
            "--output-directory",
            str(raw_dir),
            "--output-file",
            "yolov8m_profile",
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
            raise RuntimeError(f"rocprofv3 aggregate pass exited with code {return_code}.")
        validate_worker(kernel_worker_dir, "rocprofv3 aggregate", smoke)
        print()

        print("PARSING FULL ROCPROFV3 DATA IN STREAMING CHUNKS")
        print("-" * 92)
        parse_metadata = parse_rocprof_outputs(raw_dir, output_dir)
        print(
            f"Training kernels parsed: {parse_metadata['training_kernel_rows']} "
            f"dispatches across {parse_metadata['unique_training_kernels']} "
            "unique kernel names."
        )
        print(
            f"Exact kernel-union idle time: "
            f"{parse_metadata['gpu_idle_time_ms'] / 1000.0:.3f} s "
            f"({parse_metadata['gpu_idle_percent']:.3f}%)."
        )
        print()

        print("PHASE 3/3: REPRESENTATIVE ROCPROFV3 PERFETTO TIMELINE")
        print("-" * 92)
        timeline_command = [
            str(ROCPROFV3),
            "--disable-signal-handlers",
            "--kernel-trace",
            "--memory-copy-trace",
            "--marker-trace",
            "--output-format",
            "csv",
            "pftrace",
            "--output-directory",
            str(timeline_raw_dir),
            "--output-file",
            "yolov8m_timeline",
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
            str(timeline_worker_dir),
            "--internal-smoke",
        ]
        return_code = stream_subprocess(timeline_command, env=env)
        if return_code != 0:
            raise RuntimeError(f"rocprofv3 timeline pass exited with code {return_code}.")
        validate_worker(timeline_worker_dir, "rocprofv3 timeline", True)
        timeline_files = find_trace_files(timeline_raw_dir, "results.pftrace")
        if not timeline_files:
            raise RuntimeError("Representative Perfetto timeline was not produced.")
        print(f"Representative timeline: {timeline_files[0]}")
        print()

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
            "pytorch_operator_chunk_progress.csv",
            "pytorch_operator_chunk_metadata.json",
        )
        for name in primary_files:
            link_or_copy(torch_dir / name, output_dir / name)

        link_or_copy(
            torch_dir / "training_output.log",
            output_dir / "pytorch_operator_training_output.log",
        )
        link_or_copy(
            kernel_worker_dir / "training_output.log",
            output_dir / "rocprofv3_worker_training_output.log",
        )
        link_or_copy(
            kernel_worker_dir / "summary.csv",
            output_dir / "rocprofv3_worker_summary.csv",
        )
        link_or_copy(
            kernel_worker_dir / "summary.txt",
            output_dir / "rocprofv3_worker_summary.txt",
        )
        link_or_copy(
            timeline_worker_dir / "training_output.log",
            output_dir / "rocprofv3_timeline_training_output.log",
        )
        link_or_copy(
            timeline_worker_dir / "summary.txt",
            output_dir / "rocprofv3_timeline_summary.txt",
        )

        top_operators = pd.read_csv(output_dir / "pytorch_top10_operators.csv")
        top_kernels = pd.read_csv(output_dir / "amd_top5_kernels.csv")
        if top_operators.empty:
            raise RuntimeError("Top PyTorch operator table is empty.")
        if top_kernels.empty:
            raise RuntimeError("Top AMD kernel table is empty.")

        build_profile_metrics_summary(output_dir, parse_metadata)

        chunk_metadata = json.loads(
            (output_dir / "pytorch_operator_chunk_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        expected_chunks = 1 if smoke else 125
        if int(chunk_metadata.get("chunks_completed", 0)) != expected_chunks:
            raise RuntimeError(
                "Unexpected PyTorch profiler chunk count: "
                f"{chunk_metadata.get('chunks_completed')} instead of "
                f"{expected_chunks}."
            )

        run_valid = {
            "status": "VALID",
            "timestamp": datetime.now().astimezone().isoformat(),
            "mode": "smoke" if smoke else "official_profiled",
            "primary_metrics_source": "pytorch_operator_pass",
            "pytorch_operator_pass_valid": True,
            "pytorch_operator_chunks_completed": int(
                chunk_metadata["chunks_completed"]
            ),
            "rocprofv3_aggregate_pass_valid": True,
            "rocprofv3_timeline_pass_valid": True,
            "timeline_used_for_quantitative_metrics": False,
            "representative_timeline": str(timeline_files[0]),
            "training_window_filter_applied": parse_metadata[
                "training_window_filter_applied"
            ],
            "training_kernel_dispatches": parse_metadata[
                "training_kernel_rows"
            ],
            "unique_training_kernels": parse_metadata[
                "unique_training_kernels"
            ],
            "gpu_idle_time_ms": parse_metadata["gpu_idle_time_ms"],
            "gpu_idle_percent": parse_metadata["gpu_idle_percent"],
            "top_operator": str(top_operators.iloc[0]["operator"]),
            "top_kernel": str(top_kernels.iloc[0]["kernel_name"]),
        }
        with (output_dir / "RUN_VALID.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(run_valid, file, indent=2)

        print("=" * 92)
        print("PROFILED WORKFLOW VALID")
        print("=" * 92)
        print("Primary performance metrics: PyTorch operator profiler pass")
        print(f"PyTorch chunks completed: {run_valid['pytorch_operator_chunks_completed']}")
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
            "Automated YOLOv8m AMD profiling workflow using torch.profiler "
            "and rocprofv3"
        )
    )
    public = parser.add_mutually_exclusive_group()
    public.add_argument(
        "--official",
        action="store_true",
        help="Clear yoloprofiledrun and execute the full profiled workflow.",
    )
    public.add_argument(
        "--smoke",
        action="store_true",
        help="Execute a disposable three-pass profiling smoke test in /tmp.",
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

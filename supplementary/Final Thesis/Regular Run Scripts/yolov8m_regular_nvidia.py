#!/usr/bin/env python3

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pynvml
import torch
import ultralytics
from ultralytics import YOLO


SEED = 55
BATCH_SIZE = 16
EPOCHS = 10
IMGSZ = 640
LR0 = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
SAMPLE_INTERVAL_S = 0.2
WAIT_BETWEEN_RUNS_S = 10

PROJECT_DIR = Path.home() / "Thesis-Testing"
DATASET_ROOT = (
    PROJECT_DIR
    / "Dataset"
    / "COCO2017_10k"
    / "coco_minitrain_10k"
    / "coco_minitrain_10k"
)
DATA_YAML = DATASET_ROOT / "coco_minitrain_10k.yaml"
WEIGHTS_PATH = PROJECT_DIR / "scripts" / "yolov8m.pt"
LOGS_DIR = PROJECT_DIR / "Logs"
TEMP_ROOT = Path.home() / "Downloads" / "Thesis-Testing-Temporary" / "YOLOv8m"
SMOKE_DIR = TEMP_ROOT / "smoke"
ONE_EPOCH_DIR = TEMP_ROOT / "one_epoch"
GPU_ERROR_HISTORY = LOGS_DIR / "yolo_gpu_error_history.log"
ALL_RUNS_STATUS_JSON = LOGS_DIR / "yolo_all_runs_status.json"
ALL_RUNS_STATUS_LOG = LOGS_DIR / "yolo_all_runs_status.log"

EXPECTED_GPU_NAME_FRAGMENT = "RTX 5060 Ti"
EXPECTED_DRIVER_VERSION = "580.167.08"
EXPECTED_CUDA_RUNTIME = "13.0"
EXPECTED_ULTRALYTICS_VERSION = "8.4.61"
EXPECTED_WEIGHTS_SHA256 = (
    "5d4a90cdc7a21786cc59cd19778e9eafff836df9e2da32524737c7ee6efe4fe5"
)
EXPECTED_DATA_YAML_SHA256 = (
    "0b75a1916b6fb5af1b8269cf6d1392d76d001a58a218f4483b17df4ee8c32da9"
)
SCRIPT_REVISION = "2026-06-22 nvidia-yolov8m-regular-v1"

EXPECTED_OUTPUT_FILES = (
    "config.json",
    "epoch_metrics.csv",
    "telemetry.csv",
    "summary.csv",
    "summary.txt",
    "training_output.log",
    "nvml_api_errors.json",
    "nvml_error_log.txt",
    "RUN_VALID.json",
)

COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
    "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


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


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def numeric(value, default=float("nan")) -> float:
    try:
        if value is None or value == "N/A":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def format_value(value, decimals: int = 2) -> str:
    value = numeric(value)
    if math.isnan(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def text_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "Unavailable"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def script_sha256() -> str:
    return file_sha256(Path(__file__).resolve())


def cpu_package_temperature_c() -> float:
    try:
        groups = psutil.sensors_temperatures(fahrenheit=False)
    except Exception:
        return float("nan")

    preferred_groups = ["coretemp", "k10temp", "zenpower"]
    preferred_labels = (
        "package id 0",
        "tctl",
        "tdie",
        "cpu",
        "package",
    )

    for group_name in preferred_groups:
        entries = groups.get(group_name, [])
        for preferred in preferred_labels:
            for entry in entries:
                label = (entry.label or "").strip().lower()
                if preferred in label:
                    return numeric(entry.current)
        values = [numeric(entry.current) for entry in entries]
        values = [value for value in values if not math.isnan(value)]
        if values:
            return max(values)

    return float("nan")


def _clock_event_constant(*names: str) -> int:
    for name in names:
        value = getattr(pynvml, name, None)
        if value is not None:
            return int(value)
    return 0


THERMAL_CLOCK_EVENT_MASK = (
    _clock_event_constant(
        "nvmlClocksEventReasonSwThermalSlowdown",
        "nvmlClocksThrottleReasonSwThermalSlowdown",
    )
    | _clock_event_constant(
        "nvmlClocksEventReasonHwThermalSlowdown",
        "nvmlClocksThrottleReasonHwThermalSlowdown",
    )
)


def current_clock_event_reasons(handle) -> int:
    function = getattr(pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None)
    if function is None:
        function = getattr(
            pynvml,
            "nvmlDeviceGetCurrentClocksThrottleReasons",
            None,
        )
    if function is None:
        raise RuntimeError("NVML clock-event reason query is unavailable.")
    return int(function(handle))


class TelemetryMonitor:
    def __init__(self, interval_s: float = SAMPLE_INTERVAL_S):
        self.interval_s = interval_s
        self.rows = []
        self.stop_event = threading.Event()
        self.thread = None
        self.start_time = None
        self.running = False
        self.initialized = False
        self.api_errors = {}
        self.thermal_throttle_seen = False
        self.slowdown_temperature_c = float("nan")
        self.shutdown_temperature_c = float("nan")

        pynvml.nvmlInit()
        self.initialized = True
        count = pynvml.nvmlDeviceGetCount()
        if count < 1:
            self.close()
            raise RuntimeError("NVML returned no GPU handles.")

        self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.gpu_name = text_value(pynvml.nvmlDeviceGetName(self.gpu))
        self.driver_version = text_value(pynvml.nvmlSystemGetDriverVersion())

        threshold_function = getattr(
            pynvml,
            "nvmlDeviceGetTemperatureThreshold",
            None,
        )
        if threshold_function is not None:
            slowdown_kind = getattr(
                pynvml,
                "NVML_TEMPERATURE_THRESHOLD_SLOWDOWN",
                None,
            )
            shutdown_kind = getattr(
                pynvml,
                "NVML_TEMPERATURE_THRESHOLD_SHUTDOWN",
                None,
            )
            if slowdown_kind is not None:
                self.slowdown_temperature_c = numeric(
                    self._safe_api(
                        "temperature_slowdown_threshold",
                        lambda: threshold_function(self.gpu, slowdown_kind),
                        float("nan"),
                    )
                )
            if shutdown_kind is not None:
                self.shutdown_temperature_c = numeric(
                    self._safe_api(
                        "temperature_shutdown_threshold",
                        lambda: threshold_function(self.gpu, shutdown_kind),
                        float("nan"),
                    )
                )

    def _safe_api(self, name, function, default):
        try:
            return function()
        except Exception as error:
            item = self.api_errors.setdefault(
                name,
                {"count": 0, "last_error": ""},
            )
            item["count"] += 1
            item["last_error"] = f"{type(error).__name__}: {error}"
            return default

    @property
    def api_error_count(self) -> int:
        return sum(item["count"] for item in self.api_errors.values())

    def start(self) -> None:
        if self.running:
            return
        psutil.cpu_percent(interval=None)
        self.start_time = time.perf_counter()
        self.stop_event.clear()
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        previous_disk = psutil.disk_io_counters()
        previous_time = time.perf_counter()

        while not self.stop_event.wait(self.interval_s):
            now_perf = time.perf_counter()
            elapsed = max(now_perf - previous_time, 1e-9)
            disk = psutil.disk_io_counters()

            if previous_disk is not None and disk is not None:
                read_mb_s = (
                    disk.read_bytes - previous_disk.read_bytes
                ) / elapsed / (1024**2)
                write_mb_s = (
                    disk.write_bytes - previous_disk.write_bytes
                ) / elapsed / (1024**2)
            else:
                read_mb_s = float("nan")
                write_mb_s = float("nan")

            utilization = self._safe_api(
                "utilization",
                lambda: pynvml.nvmlDeviceGetUtilizationRates(self.gpu),
                None,
            )
            memory = self._safe_api(
                "memory_info",
                lambda: pynvml.nvmlDeviceGetMemoryInfo(self.gpu),
                None,
            )
            power_mw = self._safe_api(
                "power_usage",
                lambda: pynvml.nvmlDeviceGetPowerUsage(self.gpu),
                float("nan"),
            )
            temperature_c = numeric(
                self._safe_api(
                    "gpu_temperature",
                    lambda: pynvml.nvmlDeviceGetTemperature(
                        self.gpu,
                        pynvml.NVML_TEMPERATURE_GPU,
                    ),
                    float("nan"),
                )
            )
            graphics_clock_mhz = numeric(
                self._safe_api(
                    "graphics_clock",
                    lambda: pynvml.nvmlDeviceGetClockInfo(
                        self.gpu,
                        pynvml.NVML_CLOCK_GRAPHICS,
                    ),
                    float("nan"),
                )
            )
            memory_clock_mhz = numeric(
                self._safe_api(
                    "memory_clock",
                    lambda: pynvml.nvmlDeviceGetClockInfo(
                        self.gpu,
                        pynvml.NVML_CLOCK_MEM,
                    ),
                    float("nan"),
                )
            )
            pstate = self._safe_api(
                "performance_state",
                lambda: pynvml.nvmlDeviceGetPerformanceState(self.gpu),
                None,
            )
            reasons = self._safe_api(
                "clock_event_reasons",
                lambda: current_clock_event_reasons(self.gpu),
                0,
            )
            reasons = int(reasons or 0)
            thermal_event_active = int(
                bool(THERMAL_CLOCK_EVENT_MASK and reasons & THERMAL_CLOCK_EVENT_MASK)
            )
            temperature_threshold_active = int(
                not math.isnan(temperature_c)
                and not math.isnan(self.slowdown_temperature_c)
                and self.slowdown_temperature_c > 0
                and temperature_c >= self.slowdown_temperature_c
            )
            thermal_throttle_active = int(
                thermal_event_active or temperature_threshold_active
            )
            self.thermal_throttle_seen |= bool(thermal_throttle_active)

            ram = psutil.virtual_memory()
            gpu_util_percent = (
                numeric(utilization.gpu)
                if utilization is not None
                else float("nan")
            )
            gpu_memory_util_percent = (
                numeric(utilization.memory)
                if utilization is not None
                else float("nan")
            )
            vram_used_mb = (
                numeric(memory.used) / (1024**2)
                if memory is not None
                else float("nan")
            )
            vram_total_mb = (
                numeric(memory.total) / (1024**2)
                if memory is not None
                else float("nan")
            )
            power_w = numeric(power_mw)
            if not math.isnan(power_w):
                power_w /= 1000.0

            self.rows.append(
                {
                    "timestamp_iso": datetime.now().astimezone().isoformat(),
                    "time_s": now_perf - self.start_time,
                    "gpu_util_percent": gpu_util_percent,
                    "gpu_memory_util_percent": gpu_memory_util_percent,
                    "vram_used_mb": vram_used_mb,
                    "vram_total_mb": vram_total_mb,
                    "power_w": power_w,
                    "gpu_temperature_c": temperature_c,
                    "gpu_slowdown_temperature_c": self.slowdown_temperature_c,
                    "gpu_shutdown_temperature_c": self.shutdown_temperature_c,
                    "graphics_clock_mhz": graphics_clock_mhz,
                    "memory_clock_mhz": memory_clock_mhz,
                    "performance_state": (
                        f"P{int(pstate)}" if pstate is not None else "N/A"
                    ),
                    "clock_event_reasons_bitmask": reasons,
                    "clock_event_reasons_hex": hex(reasons),
                    "thermal_clock_event_active": thermal_event_active,
                    "temperature_threshold_active": temperature_threshold_active,
                    "thermal_throttle_active": thermal_throttle_active,
                    "cpu_util_percent": psutil.cpu_percent(interval=None),
                    "cpu_package_temperature_c": cpu_package_temperature_c(),
                    "system_ram_used_mb": ram.used / (1024**2),
                    "system_ram_percent": ram.percent,
                    "disk_read_mb_s": read_mb_s,
                    "disk_write_mb_s": write_mb_s,
                    "nvml_api_error_count_cumulative": self.api_error_count,
                }
            )

            previous_disk = disk
            previous_time = now_perf

    def stop(self) -> None:
        if self.running:
            self.stop_event.set()
            if self.thread is not None:
                self.thread.join(timeout=max(self.interval_s * 5, 2.0))
            self.running = False
        self.close()

    def close(self) -> None:
        if self.initialized:
            try:
                pynvml.nvmlShutdown()
            finally:
                self.initialized = False

    def dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def series_stat(df: pd.DataFrame, column: str, operation: str) -> float:
    if column not in df.columns:
        return float("nan")
    values = pd.to_numeric(df[column], errors="coerce").dropna()
    if values.empty:
        return float("nan")
    if operation == "mean":
        return float(values.mean())
    if operation == "max":
        return float(values.max())
    if operation == "min":
        return float(values.min())
    raise ValueError(f"Unsupported operation: {operation}")


def validate_telemetry(df: pd.DataFrame) -> None:
    if df.empty:
        raise RuntimeError("No telemetry samples were collected.")

    required = (
        "gpu_util_percent",
        "vram_used_mb",
        "vram_total_mb",
        "power_w",
        "gpu_temperature_c",
    )
    missing = []
    for column in required:
        if column not in df.columns:
            missing.append(column)
            continue
        values = pd.to_numeric(df[column], errors="coerce").dropna()
        if values.empty:
            missing.append(column)
    if missing:
        raise RuntimeError(
            "Required telemetry fields contain no valid samples: "
            + ", ".join(missing)
        )

    gpu_util = pd.to_numeric(df["gpu_util_percent"], errors="coerce").dropna()
    vram_used = pd.to_numeric(df["vram_used_mb"], errors="coerce").dropna()
    vram_total = pd.to_numeric(df["vram_total_mb"], errors="coerce").dropna()
    power = pd.to_numeric(df["power_w"], errors="coerce").dropna()
    temperature = pd.to_numeric(
        df["gpu_temperature_c"], errors="coerce"
    ).dropna()

    problems = []
    if ((gpu_util < 0) | (gpu_util > 100)).any():
        problems.append("GPU utilization outside 0-100%")
    if vram_total.median() < 1000 or vram_total.median() > 200000:
        problems.append(f"implausible total VRAM ({vram_total.median():.2f} MB)")
    if vram_used.max() < 100:
        problems.append(f"implausible used VRAM ({vram_used.max():.2f} MB maximum)")
    if power.mean() <= 0 or power.mean() > 1000:
        problems.append(f"implausible GPU power ({power.mean():.2f} W mean)")
    if temperature.mean() < 10 or temperature.mean() > 120:
        problems.append(
            f"implausible GPU temperature ({temperature.mean():.2f} C mean)"
        )
    if problems:
        raise RuntimeError("Telemetry validation failed: " + "; ".join(problems))


def is_gpu_related_error(text: str) -> bool:
    lowered = text.lower()
    terms = (
        "cuda",
        "cudnn",
        "device-side",
        "gpu",
        "nvml",
        "nvidia",
        "out of memory",
        "pynvml",
    )
    return any(term in lowered for term in terms)


def append_gpu_error_history(
    run_label: str,
    traceback_text: str,
    official: bool,
) -> Path:
    history_path = (
        GPU_ERROR_HISTORY
        if official
        else TEMP_ROOT / "yolo_gpu_error_history.log"
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as file:
        file.write("\n" + "=" * 80 + "\n")
        file.write(f"Timestamp: {datetime.now().astimezone().isoformat()}\n")
        file.write(f"Workload: YOLOv8m COCO MiniTrain 10K NVIDIA {run_label}\n")
        file.write(traceback_text.rstrip() + "\n")
    return history_path


def count_manifest_entries(path: Path) -> int:
    return sum(
        1
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )


def count_dataset_files() -> dict:
    paths = {
        "train_images": DATASET_ROOT / "images" / "train2017",
        "train_labels": DATASET_ROOT / "labels" / "train2017",
        "val_images": DATASET_ROOT / "images" / "val2017",
        "val_labels": DATASET_ROOT / "labels" / "val2017",
    }
    return {
        "train_images": len(list(paths["train_images"].glob("*.jpg"))),
        "train_labels": len(list(paths["train_labels"].glob("*.txt"))),
        "val_images": len(list(paths["val_images"].glob("*.jpg"))),
        "val_labels": len(list(paths["val_labels"].glob("*.txt"))),
        "train_manifest": count_manifest_entries(DATASET_ROOT / "train2017.txt"),
        "val_manifest": count_manifest_entries(DATASET_ROOT / "val2017.txt"),
    }


def verify_dataset() -> dict:
    if not DATASET_ROOT.is_dir():
        raise RuntimeError(f"Dataset root does not exist: {DATASET_ROOT}")
    if not DATA_YAML.is_file():
        raise RuntimeError(f"Dataset YAML does not exist: {DATA_YAML}")

    counts = count_dataset_files()
    expected = {
        "train_images": 10000,
        "train_labels": 10000,
        "val_images": 5000,
        "val_labels": 4952,
        "train_manifest": 10000,
        "val_manifest": 5000,
    }
    if counts != expected:
        raise RuntimeError(
            f"Dataset count mismatch. Expected {expected}, received {counts}."
        )

    yaml_hash = file_sha256(DATA_YAML)
    if yaml_hash != EXPECTED_DATA_YAML_SHA256:
        raise RuntimeError(
            "Dataset YAML checksum mismatch. "
            f"Expected {EXPECTED_DATA_YAML_SHA256}, got {yaml_hash}."
        )

    for manifest_name in ("train2017.txt", "val2017.txt"):
        manifest = DATASET_ROOT / manifest_name
        lines = [
            line.strip()
            for line in manifest.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        for entry in (lines[0], lines[-1]):
            path = Path(entry)
            if not path.is_absolute():
                path = DATASET_ROOT / path
            if not path.resolve().is_file():
                raise RuntimeError(
                    f"Manifest path does not exist in {manifest_name}: {path}"
                )

    return counts


def resolve_manifest_entry(line: str) -> Path:
    entry = Path(line.strip())
    if not entry.is_absolute():
        entry = DATASET_ROOT / entry
    return entry.resolve()


def create_smoke_yaml(run_dir: Path) -> tuple[Path, int, int]:
    train_lines = [
        line.strip()
        for line in (DATASET_ROOT / "train2017.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ][:64]
    val_lines = [
        line.strip()
        for line in (DATASET_ROOT / "val2017.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ][:32]

    train_paths = [str(resolve_manifest_entry(line)) for line in train_lines]
    val_paths = [str(resolve_manifest_entry(line)) for line in val_lines]

    train_manifest = run_dir / "smoke_train.txt"
    val_manifest = run_dir / "smoke_val.txt"
    train_manifest.write_text("\n".join(train_paths) + "\n", encoding="utf-8")
    val_manifest.write_text("\n".join(val_paths) + "\n", encoding="utf-8")

    yaml_path = run_dir / "smoke_dataset.yaml"
    names_text = "\n".join(
        f"  {index}: {json.dumps(name)}"
        for index, name in enumerate(COCO_NAMES)
    )
    yaml_path.write_text(
        f"path: {DATASET_ROOT}\n"
        f"train: {train_manifest}\n"
        f"val: {val_manifest}\n"
        "names:\n"
        f"{names_text}\n",
        encoding="utf-8",
    )
    return yaml_path, len(train_paths), len(val_paths)


class TrainingState:
    def __init__(self, monitor: TelemetryMonitor):
        self.monitor = monitor
        self.epoch_rows = []
        self.epoch_start_time = None
        self.training_start_time = None
        self.training_end_time = None
        self.dataset_size = None
        self.batches_per_epoch = None
        self.amp_enabled = None
        self.model_dtype = None
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
        if int(trainer.args.workers) != NUM_WORKERS:
            raise RuntimeError(
                f"Expected {NUM_WORKERS} workers, got {trainer.args.workers}."
            )
        if int(trainer.args.batch) != BATCH_SIZE:
            raise RuntimeError(
                f"Expected batch size {BATCH_SIZE}, got {trainer.args.batch}."
            )
        if str(trainer.args.optimizer).upper() != "SGD":
            raise RuntimeError(
                f"Expected SGD optimizer, got {trainer.args.optimizer}."
            )

    def on_train_epoch_start(self, trainer):
        torch.cuda.synchronize()
        self.epoch_start_time = time.perf_counter()

        if not self.monitor_started:
            self.dataset_size = len(trainer.train_loader.dataset)
            self.batches_per_epoch = len(trainer.train_loader)
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            self.monitor.start()
            self.monitor_started = True
            self.training_start_time = self.epoch_start_time
            print("Monitored training started.")

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
            self.stop_monitor()

    def on_train_end(self, trainer):
        if self.training_end_time is None:
            torch.cuda.synchronize()
            self.training_end_time = time.perf_counter()
        self.stop_monitor()

    def stop_monitor(self):
        if self.monitor_started and not self.monitor_stopped:
            self.monitor.stop()
            self.monitor_stopped = True

    @property
    def total_training_time(self) -> float:
        if self.training_start_time is None or self.training_end_time is None:
            return float("nan")
        return self.training_end_time - self.training_start_time


def extract_validation_metrics(metrics) -> dict:
    box = getattr(metrics, "box", None)
    if box is None:
        raise RuntimeError("Ultralytics validation did not return box metrics.")

    return {
        "map50_95": numeric(getattr(box, "map", None)),
        "map50": numeric(getattr(box, "map50", None)),
        "map75": numeric(getattr(box, "map75", None)),
        "mean_precision": numeric(getattr(box, "mp", None)),
        "mean_recall": numeric(getattr(box, "mr", None)),
    }


def output_files_are_nonempty(run_dir: Path) -> tuple[bool, list[str]]:
    missing_or_empty = []
    for filename in EXPECTED_OUTPUT_FILES:
        path = run_dir / filename
        if not path.is_file() or path.stat().st_size <= 0:
            missing_or_empty.append(filename)
    return not missing_or_empty, missing_or_empty


def write_nvml_error_log(run_dir: Path, monitor: TelemetryMonitor) -> None:
    path = run_dir / "nvml_error_log.txt"
    with path.open("w", encoding="utf-8") as file:
        file.write("NVIDIA/NVML telemetry status\n")
        file.write("=" * 48 + "\n")
        file.write(f"Recoverable NVML API errors: {monitor.api_error_count}\n")
        file.write(f"Thermal throttling observed: {monitor.thermal_throttle_seen}\n")
        file.write(f"Driver version: {monitor.driver_version}\n")
        if monitor.api_errors:
            file.write(json.dumps(monitor.api_errors, indent=2, sort_keys=True))
            file.write("\n")
        else:
            file.write("All required NVML telemetry queries succeeded.\n")


def verify_prerequisite_marker(path: Path, expected_mode: str, sha256: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"Required prerequisite marker is missing: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("status") != "VALID":
        raise RuntimeError(f"Prerequisite marker is not VALID: {path}")
    if data.get("mode") != expected_mode:
        raise RuntimeError(
            f"Prerequisite marker mode mismatch at {path}: "
            f"expected {expected_mode}, got {data.get('mode')}"
        )
    if data.get("script_sha256") != sha256:
        raise RuntimeError(
            "The script changed after the prerequisite test. Re-run smoke and "
            f"one-epoch tests before --all-runs. Marker: {path}"
        )
    if data.get("driver_version") != EXPECTED_DRIVER_VERSION:
        raise RuntimeError(
            f"Prerequisite driver mismatch at {path}: "
            f"expected {EXPECTED_DRIVER_VERSION}, got {data.get('driver_version')}"
        )
    if data.get("weights_sha256") != EXPECTED_WEIGHTS_SHA256:
        raise RuntimeError(f"Prerequisite weights checksum mismatch at {path}")
    if data.get("dataset_yaml_sha256") != EXPECTED_DATA_YAML_SHA256:
        raise RuntimeError(f"Prerequisite dataset YAML checksum mismatch at {path}")


def run_all_official() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    current_sha256 = script_sha256()
    verify_prerequisite_marker(SMOKE_DIR / "RUN_VALID.json", "smoke", current_sha256)
    verify_prerequisite_marker(
        ONE_EPOCH_DIR / "RUN_VALID.json",
        "full_dataset_one_epoch_test",
        current_sha256,
    )

    ALL_RUNS_STATUS_LOG.write_text("", encoding="utf-8")
    statuses = []

    def record(message: str) -> None:
        line = f"[{datetime.now().astimezone().isoformat()}] {message}"
        print(line, flush=True)
        with ALL_RUNS_STATUS_LOG.open("a", encoding="utf-8") as file:
            file.write(line + "\n")

    record("Smoke and one-epoch prerequisite markers verified.")
    record("Starting sequential official YOLOv8m runs 1, 2, and 3.")
    script_path = Path(__file__).resolve()

    for index, run_number in enumerate((1, 2, 3), start=1):
        record(f"Starting official run {run_number}.")
        started = time.perf_counter()
        result = subprocess.run(
            [sys.executable, str(script_path), "--run", str(run_number)],
            check=False,
        )
        elapsed_s = time.perf_counter() - started
        run_dir = LOGS_DIR / f"yolorun{run_number}"
        marker_path = run_dir / "RUN_VALID.json"

        marker_valid = False
        marker_error = ""
        if result.returncode == 0 and marker_path.is_file():
            try:
                marker_data = json.loads(marker_path.read_text(encoding="utf-8"))
                marker_valid = (
                    marker_data.get("status") == "VALID"
                    and marker_data.get("mode") == "official_regular"
                    and marker_data.get("run") == run_number
                    and marker_data.get("script_sha256") == current_sha256
                    and marker_data.get("driver_version") == EXPECTED_DRIVER_VERSION
                )
                if not marker_valid:
                    marker_error = "RUN_VALID.json contents did not match."
            except Exception as error:
                marker_error = f"Could not parse RUN_VALID.json: {error}"
        elif result.returncode != 0:
            marker_error = f"Process exited with code {result.returncode}."
        else:
            marker_error = "RUN_VALID.json was not created."

        statuses.append(
            {
                "run": run_number,
                "return_code": result.returncode,
                "elapsed_s": elapsed_s,
                "valid": marker_valid,
                "error": marker_error,
                "results_directory": str(run_dir),
            }
        )
        ALL_RUNS_STATUS_JSON.write_text(
            json.dumps(
                {"status": "IN_PROGRESS_OR_FAILED", "runs": statuses},
                indent=2,
            ),
            encoding="utf-8",
        )

        if not marker_valid:
            record(
                f"Run {run_number} failed or did not produce a valid marker. "
                "Later runs were not started."
            )
            raise SystemExit(result.returncode if result.returncode != 0 else 1)

        record(
            f"Official run {run_number} completed successfully in "
            f"{elapsed_s / 60.0:.2f} minutes."
        )
        if index < 3:
            record(f"Waiting {WAIT_BETWEEN_RUNS_S} seconds before the next run.")
            time.sleep(WAIT_BETWEEN_RUNS_S)

    ALL_RUNS_STATUS_JSON.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "completed_at": datetime.now().astimezone().isoformat(),
                "runs": statuses,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    record(
        "All three official runs completed successfully. Results are in "
        "yolorun1, yolorun2, and yolorun3."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLOv8m COCO MiniTrain 10K NVIDIA regular benchmark"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", type=int, choices=(1, 2, 3))
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--one-epoch-test", action="store_true")
    mode.add_argument("--all-runs", action="store_true")
    args = parser.parse_args()

    if args.all_runs:
        run_all_official()
        return

    smoke = bool(args.smoke)
    one_epoch_test = bool(args.one_epoch_test)

    if smoke:
        run_mode = "smoke"
        run_label = "smoke test"
        run_dir = SMOKE_DIR
    elif one_epoch_test:
        run_mode = "full_dataset_one_epoch_test"
        run_label = "full-dataset one-epoch test"
        run_dir = ONE_EPOCH_DIR
    else:
        run_mode = "official_regular"
        run_label = f"regular run {args.run}"
        run_dir = LOGS_DIR / f"yolorun{args.run}"

    reset_directory(run_dir)
    log_file = (run_dir / "training_output.log").open(
        "w", encoding="utf-8", buffering=1
    )
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
        set_seed(SEED)

        counts = verify_dataset()
        if not WEIGHTS_PATH.is_file():
            raise RuntimeError(f"YOLOv8m weights do not exist: {WEIGHTS_PATH}")
        weights_hash = file_sha256(WEIGHTS_PATH)
        if weights_hash != EXPECTED_WEIGHTS_SHA256:
            raise RuntimeError(
                "YOLOv8m weights checksum mismatch. "
                f"Expected {EXPECTED_WEIGHTS_SHA256}, got {weights_hash}."
            )
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot detect the NVIDIA GPU.")
        if torch.version.hip is not None:
            raise RuntimeError("This PyTorch build unexpectedly reports a HIP runtime.")
        if str(torch.version.cuda) != EXPECTED_CUDA_RUNTIME:
            raise RuntimeError(
                f"Expected CUDA runtime {EXPECTED_CUDA_RUNTIME}, "
                f"got {torch.version.cuda}."
            )
        if ultralytics.__version__ != EXPECTED_ULTRALYTICS_VERSION:
            raise RuntimeError(
                f"Expected Ultralytics {EXPECTED_ULTRALYTICS_VERSION}, "
                f"got {ultralytics.__version__}."
            )

        gpu_name = torch.cuda.get_device_name(0)
        compute_capability = torch.cuda.get_device_capability(0)
        if EXPECTED_GPU_NAME_FRAGMENT not in gpu_name:
            raise RuntimeError(
                f"Unexpected GPU selected: {gpu_name}. "
                f"Expected a name containing {EXPECTED_GPU_NAME_FRAGMENT}."
            )

        monitor = TelemetryMonitor()
        if EXPECTED_GPU_NAME_FRAGMENT not in monitor.gpu_name:
            raise RuntimeError(
                f"Unexpected NVML GPU selected: {monitor.gpu_name}."
            )
        if monitor.driver_version != EXPECTED_DRIVER_VERSION:
            raise RuntimeError(
                f"Expected NVIDIA driver {EXPECTED_DRIVER_VERSION}, "
                f"got {monitor.driver_version}."
            )

        epochs_to_run = 1 if (smoke or one_epoch_test) else EPOCHS
        if smoke:
            selected_yaml, training_images, validation_images = create_smoke_yaml(
                run_dir
            )
        else:
            selected_yaml = DATA_YAML
            training_images = counts["train_images"]
            validation_images = counts["val_images"]

        current_script_sha256 = script_sha256()
        yaml_sha256 = file_sha256(DATA_YAML)
        train_manifest_sha256 = file_sha256(DATASET_ROOT / "train2017.txt")
        val_manifest_sha256 = file_sha256(DATASET_ROOT / "val2017.txt")

        print("=" * 84)
        print(f"YOLOv8m COCO MiniTrain 10K NVIDIA {run_label}")
        print("=" * 84)
        print(f"Timestamp: {datetime.now().astimezone().isoformat()}")
        print(f"Output directory: {run_dir}")
        print(f"Dataset root: {DATASET_ROOT}")
        print(f"Dataset YAML: {selected_yaml}")
        print(f"Weights: {WEIGHTS_PATH}")
        print(f"Python: {platform.python_version()}")
        print(f"Platform: {platform.platform()}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"Ultralytics: {ultralytics.__version__}")
        print(f"NVIDIA ML Python: {package_version('nvidia-ml-py')}")
        print(f"NVIDIA driver: {monitor.driver_version}")
        print(f"GPU: {gpu_name}")
        print(
            "Compute capability: "
            f"{compute_capability[0]}.{compute_capability[1]}"
        )
        print(f"Script revision: {SCRIPT_REVISION}")
        print()

        config = {
            "mode": run_mode,
            "run": args.run if run_mode == "official_regular" else None,
            "seed": SEED,
            "model": "YOLOv8m",
            "weights": str(WEIGHTS_PATH),
            "weights_sha256": weights_hash,
            "dataset": "COCO 2017 MiniTrain 10K",
            "dataset_source": "Kaggle banuprasadb/coco-minitrain-10k",
            "dataset_archive_sha256": (
                "4ec3d60f4e74164d03d23a846f7cd06c0f7b4260618b6f37135d34526cdc1607"
            ),
            "dataset_root": str(DATASET_ROOT),
            "dataset_yaml": str(selected_yaml),
            "official_dataset_yaml_sha256": yaml_sha256,
            "train_manifest_sha256": train_manifest_sha256,
            "val_manifest_sha256": val_manifest_sha256,
            "dataset_counts": counts,
            "training_images": training_images,
            "validation_images": validation_images,
            "batch_size": BATCH_SIZE,
            "image_size": IMGSZ,
            "epochs": epochs_to_run,
            "official_epochs": EPOCHS,
            "optimizer": "SGD",
            "learning_rate": LR0,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "precision": "FP32 / AMP OFF",
            "workers": NUM_WORKERS,
            "device": 0,
            "telemetry_interval_s": SAMPLE_INTERVAL_S,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "ultralytics_version": ultralytics.__version__,
            "nvidia_ml_py_version": package_version("nvidia-ml-py"),
            "driver_version": monitor.driver_version,
            "gpu": gpu_name,
            "compute_capability": (
                f"{compute_capability[0]}.{compute_capability[1]}"
            ),
            "validation_metric_primary": "mAP50-95",
            "validation_metric_secondary": "mAP50",
            "script_revision": SCRIPT_REVISION,
            "script_sha256": current_script_sha256,
        }
        with (run_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

        state = TrainingState(monitor)
        model = YOLO(str(WEIGHTS_PATH))
        model.add_callback("on_train_start", state.on_train_start)
        model.add_callback("on_train_epoch_start", state.on_train_epoch_start)
        model.add_callback("on_train_epoch_end", state.on_train_epoch_end)
        model.add_callback("on_train_end", state.on_train_end)

        train_output_dir = run_dir / "ultralytics_train"

        print("Starting monitored training...")
        model.train(
            data=str(selected_yaml),
            epochs=epochs_to_run,
            imgsz=IMGSZ,
            batch=BATCH_SIZE,
            device=0,
            workers=NUM_WORKERS,
            optimizer="SGD",
            lr0=LR0,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
            amp=False,
            seed=SEED,
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
            project=str(run_dir),
            name="ultralytics_train",
            exist_ok=True,
        )

        state.stop_monitor()
        total_training_time = state.total_training_time
        if math.isnan(total_training_time) or total_training_time <= 0:
            raise RuntimeError("Training callbacks did not produce valid timing.")

        epoch_df = pd.DataFrame(state.epoch_rows)
        epoch_df.to_csv(run_dir / "epoch_metrics.csv", index=False)

        telemetry_df = monitor.dataframe()
        telemetry_df.to_csv(run_dir / "telemetry.csv", index=False)
        validate_telemetry(telemetry_df)

        with (run_dir / "nvml_api_errors.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(monitor.api_errors, file, indent=2)
        write_nvml_error_log(run_dir, monitor)

        if len(state.epoch_rows) != epochs_to_run:
            raise RuntimeError(
                f"Incomplete run: expected {epochs_to_run} epochs, "
                f"got {len(state.epoch_rows)}."
            )

        batches_per_epoch = int(state.batches_per_epoch)
        total_batches = batches_per_epoch * epochs_to_run
        total_samples = training_images * epochs_to_run

        if not smoke:
            expected_batches = math.ceil(training_images / BATCH_SIZE) * epochs_to_run
            expected_samples = training_images * epochs_to_run
            if total_batches != expected_batches:
                raise RuntimeError(
                    f"Expected {expected_batches} batches, got {total_batches}."
                )
            if total_samples != expected_samples:
                raise RuntimeError(
                    f"Expected {expected_samples} processed images, "
                    f"got {total_samples}."
                )

        last_weights = train_output_dir / "weights" / "last.pt"
        if not last_weights.is_file():
            raise RuntimeError(
                f"Expected trained checkpoint was not created: {last_weights}"
            )

        print()
        print("Starting final validation outside timed training...")
        validation_model = YOLO(str(last_weights))
        validation_results = validation_model.val(
            data=str(selected_yaml),
            imgsz=IMGSZ,
            batch=BATCH_SIZE,
            device=0,
            workers=NUM_WORKERS,
            amp=False,
            plots=False,
            save_json=False,
            verbose=True,
            project=str(run_dir),
            name="ultralytics_val",
            exist_ok=True,
        )
        validation = extract_validation_metrics(validation_results)

        throughput = total_samples / total_training_time
        batch_latency_ms = total_training_time / total_batches * 1000.0
        average_epoch_time = float(epoch_df["epoch_time_s"].mean())

        average_power_w = series_stat(telemetry_df, "power_w", "mean")
        average_gpu_util = series_stat(telemetry_df, "gpu_util_percent", "mean")
        average_vram_mb = series_stat(telemetry_df, "vram_used_mb", "mean")
        peak_vram_mb = series_stat(telemetry_df, "vram_used_mb", "max")
        average_gpu_temp_c = series_stat(
            telemetry_df, "gpu_temperature_c", "mean"
        )
        peak_gpu_temp_c = series_stat(
            telemetry_df, "gpu_temperature_c", "max"
        )
        average_graphics_clock_mhz = series_stat(
            telemetry_df, "graphics_clock_mhz", "mean"
        )
        average_memory_clock_mhz = series_stat(
            telemetry_df, "memory_clock_mhz", "mean"
        )
        average_cpu_util = series_stat(
            telemetry_df, "cpu_util_percent", "mean"
        )
        average_cpu_temp_c = series_stat(
            telemetry_df, "cpu_package_temperature_c", "mean"
        )
        average_system_ram_mb = series_stat(
            telemetry_df, "system_ram_used_mb", "mean"
        )
        peak_system_ram_mb = series_stat(
            telemetry_df, "system_ram_used_mb", "max"
        )
        average_disk_read_mb_s = series_stat(
            telemetry_df, "disk_read_mb_s", "mean"
        )
        average_disk_write_mb_s = series_stat(
            telemetry_df, "disk_write_mb_s", "mean"
        )
        peak_disk_read_mb_s = series_stat(
            telemetry_df, "disk_read_mb_s", "max"
        )
        peak_disk_write_mb_s = series_stat(
            telemetry_df, "disk_write_mb_s", "max"
        )
        torch_peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)

        performance_per_watt = (
            throughput / average_power_w
            if not math.isnan(average_power_w) and average_power_w > 0
            else float("nan")
        )

        stability_notes = (
            "Completed successfully; no Python, CUDA, cuDNN, Ultralytics, "
            "NVML, or GPU fatal errors detected."
        )
        if monitor.thermal_throttle_seen:
            stability_notes += " NVML reported thermal throttling."
        else:
            stability_notes += " NVML reported no thermal throttling."
        if monitor.api_errors:
            stability_notes += (
                f" NVML had {monitor.api_error_count} recoverable telemetry "
                "API errors."
            )
        else:
            stability_notes += " All required NVML telemetry queries succeeded."

        summary = {
            "mode": run_mode,
            "run": args.run if run_mode == "official_regular" else None,
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
            "average_gpu_temperature_c": average_gpu_temp_c,
            "peak_gpu_temperature_c": peak_gpu_temp_c,
            "average_graphics_clock_mhz": average_graphics_clock_mhz,
            "average_memory_clock_mhz": average_memory_clock_mhz,
            "thermal_throttling_observed": monitor.thermal_throttle_seen,
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
            "total_images_processed": total_samples,
            "total_batches_processed": total_batches,
            "batches_per_epoch": batches_per_epoch,
            "telemetry_samples": len(telemetry_df),
            "telemetry_interval_s": SAMPLE_INTERVAL_S,
            "workers": NUM_WORKERS,
            "amp_enabled": bool(state.amp_enabled),
            "model_parameter_dtype": state.model_dtype,
            "driver_version": monitor.driver_version,
            "script_sha256": current_script_sha256,
            "weights_sha256": weights_hash,
            "dataset_yaml_sha256": yaml_sha256,
            "stability_error_notes": stability_notes,
        }
        pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)

        with (run_dir / "summary.txt").open("w", encoding="utf-8") as file:
            file.write(f"YOLOv8m NVIDIA {run_label.title()}\n")
            file.write("=" * 64 + "\n")
            file.write(f"Throughput: {format_value(throughput)} images/s\n")
            file.write(f"Batch Latency: {format_value(batch_latency_ms)} ms/batch\n")
            file.write(f"Total Training Time: {format_value(total_training_time)} s\n")
            file.write(f"Average Epoch Time: {format_value(average_epoch_time)} s\n")
            file.write(f"Average GPU Power Draw: {format_value(average_power_w)} W\n")
            file.write(
                "Performance per Watt: "
                f"{format_value(performance_per_watt)} images/s/W\n"
            )
            file.write(
                f"Average GPU Utilization: {format_value(average_gpu_util)} %\n"
            )
            file.write(f"Average VRAM Usage: {format_value(average_vram_mb)} MB\n")
            file.write(f"Peak VRAM Usage: {format_value(peak_vram_mb)} MB\n")
            file.write(
                f"Average GPU Temperature: {format_value(average_gpu_temp_c)} C\n"
            )
            file.write(f"Peak GPU Temperature: {format_value(peak_gpu_temp_c)} C\n")
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
            file.write(f"Peak System RAM: {format_value(peak_system_ram_mb)} MB\n")
            file.write(
                f"Average System RAM: {format_value(average_system_ram_mb)} MB\n"
            )
            file.write(
                f"Average CPU Utilization: {format_value(average_cpu_util)} %\n"
            )
            file.write(
                "Average CPU Package Temperature: "
                f"{format_value(average_cpu_temp_c)} C\n"
            )
            file.write(
                "Average Disk Read / Write: "
                f"{format_value(average_disk_read_mb_s)} / "
                f"{format_value(average_disk_write_mb_s)} MB/s\n"
            )
            file.write(
                "Peak Disk Read / Write: "
                f"{format_value(peak_disk_read_mb_s)} / "
                f"{format_value(peak_disk_write_mb_s)} MB/s\n"
            )
            file.write(f"Precision: FP32; AMP enabled: {bool(state.amp_enabled)}\n")
            file.write(f"Workers: {NUM_WORKERS}\n")
            file.write(f"Driver: {monitor.driver_version}\n")
            file.write(f"Stability / Error Notes: {stability_notes}\n")

        valid_marker = {
            "status": "VALID",
            "timestamp": datetime.now().astimezone().isoformat(),
            "mode": run_mode,
            "run": args.run if run_mode == "official_regular" else None,
            "workers": NUM_WORKERS,
            "amp_enabled": bool(state.amp_enabled),
            "epochs_completed": len(state.epoch_rows),
            "total_batches_processed": total_batches,
            "total_images_processed": total_samples,
            "driver_version": monitor.driver_version,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "ultralytics_version": ultralytics.__version__,
            "script_sha256": current_script_sha256,
            "weights_sha256": weights_hash,
            "dataset_yaml_sha256": yaml_sha256,
            "nvml_api_error_count": monitor.api_error_count,
            "thermal_throttling_observed": monitor.thermal_throttle_seen,
        }
        with (run_dir / "RUN_VALID.json").open("w", encoding="utf-8") as file:
            json.dump(valid_marker, file, indent=2)

        outputs_valid, missing_outputs = output_files_are_nonempty(run_dir)
        if not outputs_valid:
            raise RuntimeError(
                "Run output validation failed; missing or empty files: "
                + ", ".join(missing_outputs)
            )

        print()
        print("=" * 84)
        if smoke:
            print("SMOKE TEST PASSED")
        elif one_epoch_test:
            print("ONE-EPOCH FULL-DATASET TEST VALID")
        else:
            print("OFFICIAL RUN VALID")
        print("=" * 84)
        print(f"Throughput: {throughput:.2f} images/s")
        print(f"Batch Latency: {batch_latency_ms:.2f} ms/batch")
        print(f"Total Training Time: {total_training_time:.2f} s")
        print(f"Average GPU Power Draw: {average_power_w:.2f} W")
        print(f"Average GPU Utilization: {average_gpu_util:.2f} %")
        print(f"Average VRAM Usage: {average_vram_mb:.2f} MB")
        print(f"Peak VRAM Usage: {peak_vram_mb:.2f} MB")
        print(f"Average GPU Temperature: {average_gpu_temp_c:.2f} C")
        print(f"Final Validation mAP50-95: {validation['map50_95']:.4f}")
        print(f"Final Validation mAP50: {validation['map50']:.4f}")
        print(f"Workers: {NUM_WORKERS}")
        print(f"AMP enabled: {bool(state.amp_enabled)}")
        print(f"Stability / Error Notes: {stability_notes}")
        print(f"Results saved to: {run_dir}")

    except Exception:
        if state is not None:
            try:
                state.stop_monitor()
            except Exception:
                pass
        elif monitor is not None:
            try:
                monitor.stop()
            except Exception:
                pass

        if monitor is not None:
            try:
                partial_df = monitor.dataframe()
                if not partial_df.empty:
                    partial_df.to_csv(
                        run_dir / "telemetry_partial.csv", index=False
                    )
                with (run_dir / "nvml_api_errors.json").open(
                    "w", encoding="utf-8"
                ) as file:
                    json.dump(monitor.api_errors, file, indent=2)
                write_nvml_error_log(run_dir, monitor)
            except Exception:
                pass

        traceback_text = traceback.format_exc()
        print()
        print("RUN FAILED")
        print(traceback_text)

        try:
            (run_dir / "RUN_FAILED.txt").write_text(
                traceback_text, encoding="utf-8"
            )
        except Exception:
            pass

        if is_gpu_related_error(traceback_text):
            try:
                history = append_gpu_error_history(
                    run_label,
                    traceback_text,
                    official=(run_mode == "official_regular"),
                )
                print(f"GPU-related failure appended to: {history}")
            except Exception as history_error:
                print(f"Could not append GPU error history: {history_error}")
        else:
            print("Failure was not automatically classified as GPU-related.")

        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


if __name__ == "__main__":
    main()

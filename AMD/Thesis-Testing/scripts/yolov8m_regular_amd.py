#!/usr/bin/env python3

import argparse
import importlib.metadata
import json
import math
import os
import platform
import random
import shutil
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import amdsmi
import numpy as np
import pandas as pd
import psutil
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
GPU_ERROR_HISTORY = LOGS_DIR / "yolo_gpu_error_history.log"

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


def first_numeric(mapping, keys, default=float("nan")) -> float:
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        value = numeric(mapping.get(key), default=float("nan"))
        if not math.isnan(value):
            return value
    return default


def cpu_package_temperature_c() -> float:
    try:
        groups = psutil.sensors_temperatures(fahrenheit=False)
    except Exception:
        return float("nan")

    preferred_groups = ["coretemp", "k10temp", "zenpower"]
    preferred_labels = ("package id 0", "tctl", "tdie", "cpu", "package")

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

        amdsmi.amdsmi_init()
        self.initialized = True
        devices = amdsmi.amdsmi_get_processor_handles()
        if not devices:
            self.close()
            raise RuntimeError("AMD SMI returned no GPU handles.")
        self.gpu = devices[0]

    def _safe_api(self, name, function, default):
        try:
            return function()
        except Exception as error:
            item = self.api_errors.setdefault(
                name, {"count": 0, "last_error": ""}
            )
            item["count"] += 1
            item["last_error"] = f"{type(error).__name__}: {error}"
            return default

    def _temperature_c(self, sensor_type) -> float:
        raw = self._safe_api(
            f"temperature_{sensor_type}",
            lambda: amdsmi.amdsmi_get_temp_metric(
                self.gpu,
                sensor_type,
                amdsmi.AmdSmiTemperatureMetric.CURRENT,
            ),
            float("nan"),
        )
        value = numeric(raw)
        if math.isnan(value):
            return value
        return value / 1000.0 if value > 1000.0 else value

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

            activity = self._safe_api(
                "gpu_activity",
                lambda: amdsmi.amdsmi_get_gpu_activity(self.gpu),
                {},
            )
            vram = self._safe_api(
                "vram_usage",
                lambda: amdsmi.amdsmi_get_gpu_vram_usage(self.gpu),
                {},
            )
            power = self._safe_api(
                "power_info",
                lambda: amdsmi.amdsmi_get_power_info(self.gpu),
                {},
            )
            gfx_clock = self._safe_api(
                "gfx_clock",
                lambda: amdsmi.amdsmi_get_clock_info(
                    self.gpu, amdsmi.AmdSmiClkType.GFX
                ),
                {},
            )
            mem_clock = self._safe_api(
                "mem_clock",
                lambda: amdsmi.amdsmi_get_clock_info(
                    self.gpu, amdsmi.AmdSmiClkType.MEM
                ),
                {},
            )
            vram_used = numeric(vram.get("vram_used"))
            vram_total = numeric(vram.get("vram_total"))
            if not math.isnan(vram_used) and vram_used > 1024**2:
                vram_used /= 1024**2
            if not math.isnan(vram_total) and vram_total > 1024**2:
                vram_total /= 1024**2

            ram = psutil.virtual_memory()
            self.rows.append(
                {
                    "timestamp_iso": datetime.now().astimezone().isoformat(),
                    "time_s": now_perf - self.start_time,
                    "gpu_util_percent": numeric(activity.get("gfx_activity")),
                    "gpu_memory_util_percent": numeric(
                        activity.get("umc_activity")
                    ),
                    "gpu_mm_util_percent": numeric(activity.get("mm_activity")),
                    "vram_used_mb": vram_used,
                    "vram_total_mb": vram_total,
                    "power_w": first_numeric(
                        power,
                        (
                            "socket_power",
                            "average_socket_power",
                            "current_socket_power",
                        ),
                    ),
                    "gpu_edge_temperature_c": self._temperature_c(
                        amdsmi.AmdSmiTemperatureType.EDGE
                    ),
                    "gpu_hotspot_temperature_c": self._temperature_c(
                        amdsmi.AmdSmiTemperatureType.HOTSPOT
                    ),
                    "gpu_memory_temperature_c": self._temperature_c(
                        amdsmi.AmdSmiTemperatureType.VRAM
                    ),
                    "graphics_clock_mhz": numeric(gfx_clock.get("clk")),
                    "memory_clock_mhz": numeric(mem_clock.get("clk")),
                    "cpu_util_percent": psutil.cpu_percent(interval=None),
                    "cpu_package_temperature_c": cpu_package_temperature_c(),
                    "system_ram_used_mb": ram.used / (1024**2),
                    "system_ram_percent": ram.percent,
                    "disk_read_mb_s": read_mb_s,
                    "disk_write_mb_s": write_mb_s,
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
                amdsmi.amdsmi_shut_down()
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
    raise ValueError(f"Unsupported operation: {operation}")


def format_value(value, decimals: int = 2) -> str:
    value = numeric(value)
    if math.isnan(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def read_rocm_version() -> str:
    version_file = Path("/opt/rocm/.info/version")
    try:
        return version_file.read_text(encoding="utf-8").strip()
    except OSError:
        return "Unavailable"


def validate_telemetry(df: pd.DataFrame) -> None:
    if df.empty:
        raise RuntimeError("No telemetry samples were collected.")

    required = (
        "gpu_util_percent",
        "vram_used_mb",
        "power_w",
        "gpu_edge_temperature_c",
    )
    invalid = []
    for column in required:
        values = pd.to_numeric(df.get(column), errors="coerce").dropna()
        if values.empty:
            invalid.append(column)
    if invalid:
        raise RuntimeError(
            "Required telemetry fields contain no valid samples: "
            + ", ".join(invalid)
        )

    edge = series_stat(df, "gpu_edge_temperature_c", "mean")
    vram = series_stat(df, "vram_used_mb", "max")
    if not math.isnan(edge) and not 5.0 <= edge <= 120.0:
        raise RuntimeError(f"Implausible GPU temperature telemetry: {edge}")
    if not math.isnan(vram) and vram < 100.0:
        raise RuntimeError(f"Implausible peak VRAM telemetry: {vram}")


def is_gpu_related_error(text: str) -> bool:
    lowered = text.lower()
    terms = (
        "amd-smi", "amdsmi", "amdgpu", "cuda", "gfx1200", "gpu", "hip",
        "hsa", "memory access fault", "miopen", "out of memory", "rocblas",
        "rocm", "rocprof",
    )
    return any(term in lowered for term in terms)


def append_gpu_error_history(run_label: str, traceback_text: str) -> None:
    GPU_ERROR_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with GPU_ERROR_HISTORY.open("a", encoding="utf-8") as file:
        file.write("\n" + "=" * 80 + "\n")
        file.write(f"Timestamp: {datetime.now().astimezone().isoformat()}\n")
        file.write(f"Workload: YOLOv8m COCO MiniTrain 10K AMD {run_label}\n")
        file.write(traceback_text.rstrip() + "\n")


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
        "train_manifest": sum(
            1 for _ in (DATASET_ROOT / "train2017.txt").open(
                "r", encoding="utf-8"
            )
        ),
        "val_manifest": sum(
            1 for _ in (DATASET_ROOT / "val2017.txt").open(
                "r", encoding="utf-8"
            )
        ),
    }


def verify_dataset() -> dict:
    if not DATASET_ROOT.exists():
        raise RuntimeError(f"Dataset root does not exist: {DATASET_ROOT}")
    if not DATA_YAML.exists():
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="YOLOv8m COCO MiniTrain 10K AMD regular benchmark"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", type=int, choices=(1, 2, 3))
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--one-epoch-test", action="store_true")
    mode.add_argument("--all-runs", action="store_true")
    args = parser.parse_args()

    if args.all_runs:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        overall_log = LOGS_DIR / "yolo_all_runs_status.log"
        overall_json = LOGS_DIR / "yolo_all_runs_status.json"
        overall_log.write_text("", encoding="utf-8")
        run_statuses = []

        def record(message: str) -> None:
            line = f"[{datetime.now().astimezone().isoformat()}] {message}"
            print(line, flush=True)
            with overall_log.open("a", encoding="utf-8") as file:
                file.write(line + "\n")

        record("Starting sequential official YOLOv8m runs 1, 2, and 3.")
        script_path = Path(__file__).resolve()

        for run_number in (1, 2, 3):
            record(f"Starting official run {run_number}.")
            started = time.perf_counter()
            result = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--run",
                    str(run_number),
                ],
                check=False,
            )
            elapsed_s = time.perf_counter() - started
            marker_path = LOGS_DIR / f"yolorun{run_number}" / "RUN_VALID.json"

            marker_valid = False
            if result.returncode == 0 and marker_path.exists():
                try:
                    marker_data = json.loads(
                        marker_path.read_text(encoding="utf-8")
                    )
                    marker_valid = (
                        marker_data.get("status") == "VALID"
                        and marker_data.get("run") == run_number
                        and marker_data.get("mode") == "official_regular"
                    )
                except Exception:
                    marker_valid = False

            run_statuses.append(
                {
                    "run": run_number,
                    "return_code": result.returncode,
                    "elapsed_s": elapsed_s,
                    "valid": marker_valid,
                    "results_directory": str(
                        LOGS_DIR / f"yolorun{run_number}"
                    ),
                }
            )

            overall_json.write_text(
                json.dumps(
                    {
                        "status": "IN_PROGRESS_OR_FAILED",
                        "runs": run_statuses,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

            if not marker_valid:
                record(
                    f"Run {run_number} failed or did not produce a valid "
                    "marker. Later runs were not started."
                )
                raise SystemExit(
                    result.returncode if result.returncode != 0 else 1
                )

            record(
                f"Official run {run_number} completed successfully in "
                f"{elapsed_s / 3600.0:.2f} hours."
            )

        overall_json.write_text(
            json.dumps(
                {
                    "status": "COMPLETE",
                    "completed_at": datetime.now().astimezone().isoformat(),
                    "runs": run_statuses,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        record(
            "All three official runs completed successfully. Results are in "
            "yolorun1, yolorun2, and yolorun3."
        )
        return

    smoke = bool(args.smoke)
    one_epoch_test = bool(args.one_epoch_test)

    if smoke:
        run_mode = "smoke"
        run_label = "smoke test"
        run_dir = Path(tempfile.gettempdir()) / "yolov8m_amd_smoke"
    elif one_epoch_test:
        run_mode = "full_dataset_one_epoch_test"
        run_label = "full-dataset one-epoch test"
        run_dir = LOGS_DIR / "yolotest1epoch"
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
        set_seed(SEED)

        counts = verify_dataset()
        if not WEIGHTS_PATH.exists():
            raise RuntimeError(f"YOLOv8m weights do not exist: {WEIGHTS_PATH}")
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

        epochs_to_run = 1 if (smoke or one_epoch_test) else EPOCHS
        if smoke:
            selected_yaml, training_images, validation_images = create_smoke_yaml(
                run_dir
            )
        else:
            selected_yaml = DATA_YAML
            training_images = counts["train_images"]
            validation_images = counts["val_images"]

        print("=" * 84)
        print(f"YOLOv8m COCO MiniTrain 10K AMD {run_label}")
        print("=" * 84)
        print(f"Timestamp: {datetime.now().astimezone().isoformat()}")
        print(f"Output directory: {run_dir}")
        print(f"Dataset root: {DATASET_ROOT}")
        print(f"Dataset YAML: {selected_yaml}")
        print(f"Weights: {WEIGHTS_PATH}")
        print(f"Python: {platform.python_version()}")
        print(f"Platform: {platform.platform()}")
        print(f"PyTorch: {torch.__version__}")
        print(f"HIP runtime: {torch.version.hip}")
        print(f"System ROCm: {read_rocm_version()}")
        print(f"Ultralytics: {ultralytics.__version__}")
        print(f"AMD SMI Python: {importlib.metadata.version('amdsmi')}")
        print(f"GPU: {gpu_name}")
        print(f"GPU architecture: {gpu_arch}")
        print()

        config = {
            "mode": run_mode,
            "run": args.run if run_mode == "official_regular" else None,
            "seed": SEED,
            "model": "YOLOv8m",
            "weights": str(WEIGHTS_PATH),
            "dataset": "COCO 2017 MiniTrain 10K",
            "dataset_source": "Kaggle banuprasadb/coco-minitrain-10k",
            "dataset_root": str(DATASET_ROOT),
            "dataset_yaml": str(selected_yaml),
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
            "torch_version": torch.__version__,
            "hip_runtime": torch.version.hip,
            "system_rocm": read_rocm_version(),
            "ultralytics_version": ultralytics.__version__,
            "amdsmi_version": importlib.metadata.version("amdsmi"),
            "gpu": gpu_name,
            "gpu_architecture": gpu_arch,
            "validation_metric": "mAP50-95",
            "script_revision": "2026-06-15 remove-unsupported-violation-status",
        }
        with (run_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

        monitor = TelemetryMonitor()
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

        with (run_dir / "amdsmi_api_errors.json").open(
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
        total_samples = training_images * epochs_to_run

        if not smoke:
            expected_batches = (
                math.ceil(training_images / BATCH_SIZE) * epochs_to_run
            )
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
        if not last_weights.exists():
            raise RuntimeError(
                f"Expected trained checkpoint was not created: {last_weights}"
            )

        print()
        print("Starting final validation...")
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
        average_epoch_time = statistics.mean(
            row["epoch_time_s"] for row in state.epoch_rows
        )

        average_power_w = series_stat(telemetry_df, "power_w", "mean")
        average_gpu_util = series_stat(
            telemetry_df, "gpu_util_percent", "mean"
        )
        average_vram_mb = series_stat(telemetry_df, "vram_used_mb", "mean")
        peak_vram_mb = series_stat(telemetry_df, "vram_used_mb", "max")
        average_edge_temp_c = series_stat(
            telemetry_df, "gpu_edge_temperature_c", "mean"
        )
        average_hotspot_temp_c = series_stat(
            telemetry_df, "gpu_hotspot_temperature_c", "mean"
        )
        average_memory_temp_c = series_stat(
            telemetry_df, "gpu_memory_temperature_c", "mean"
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

        peak_edge_temp_c = series_stat(
            telemetry_df, "gpu_edge_temperature_c", "max"
        )
        peak_hotspot_temp_c = series_stat(
            telemetry_df, "gpu_hotspot_temperature_c", "max"
        )
        peak_memory_temp_c = series_stat(
            telemetry_df, "gpu_memory_temperature_c", "max"
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
            "AMD SMI, or GPU fatal errors detected."
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
            error_count = sum(
                item["count"] for item in monitor.api_errors.values()
            )
            stability_notes += (
                f" AMD SMI had {error_count} recoverable telemetry API errors."
            )
        else:
            stability_notes += " All AMD SMI telemetry queries succeeded."

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
            "total_images_processed": total_samples,
            "total_batches_processed": total_batches,
            "telemetry_samples": len(telemetry_df),
            "telemetry_interval_s": SAMPLE_INTERVAL_S,
            "workers": NUM_WORKERS,
            "amp_enabled": bool(state.amp_enabled),
            "model_parameter_dtype": state.model_dtype,
            "stability_error_notes": stability_notes,
        }
        pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)

        with (run_dir / "summary.txt").open("w", encoding="utf-8") as file:
            file.write(f"YOLOv8m AMD {run_label.title()}\n")
            file.write("=" * 64 + "\n")
            file.write(f"Throughput: {format_value(throughput)} images/s\n")
            file.write(
                f"Batch Latency: {format_value(batch_latency_ms)} ms/batch\n"
            )
            file.write(
                f"Total Training Time: {format_value(total_training_time)} s\n"
            )
            file.write(
                f"Average Epoch Time: {format_value(average_epoch_time)} s\n"
            )
            file.write(
                f"Average GPU Power Draw: {format_value(average_power_w)} W\n"
            )
            file.write(
                "Performance per Watt: "
                f"{format_value(performance_per_watt)} images/s/W\n"
            )
            file.write(
                f"Average GPU Utilization: {format_value(average_gpu_util)} %\n"
            )
            file.write(
                f"Average VRAM Usage: {format_value(average_vram_mb)} MB\n"
            )
            file.write(
                f"Peak VRAM Usage: {format_value(peak_vram_mb)} MB\n"
            )
            file.write(
                "Average GPU Edge Temperature: "
                f"{format_value(average_edge_temp_c)} C\n"
            )
            file.write(
                "Average GPU Hotspot Temperature: "
                f"{format_value(average_hotspot_temp_c)} C\n"
            )
            file.write(
                "Average GPU Memory Temperature: "
                f"{format_value(average_memory_temp_c)} C\n"
            )
            file.write(
                f"Final Validation mAP50-95: {validation['map50_95']:.4f}\n"
            )
            file.write(f"Final Validation mAP50: {validation['map50']:.4f}\n")
            file.write(
                f"Final Validation Precision: "
                f"{validation['mean_precision']:.4f}\n"
            )
            file.write(
                f"Final Validation Recall: {validation['mean_recall']:.4f}\n"
            )
            file.write(
                f"Peak System RAM: {format_value(peak_system_ram_mb)} MB\n"
            )
            file.write(
                f"Average System RAM: "
                f"{format_value(average_system_ram_mb)} MB\n"
            )
            file.write(
                f"Average CPU Utilization: "
                f"{format_value(average_cpu_util)} %\n"
            )
            file.write(
                f"Average CPU Package Temperature: "
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
            file.write(
                f"Precision: FP32; AMP enabled: {bool(state.amp_enabled)}\n"
            )
            file.write(f"Workers: {NUM_WORKERS}\n")
            file.write(f"Stability / Error Notes: {stability_notes}\n")

        valid_marker = {
            "status": "VALID",
            "timestamp": datetime.now().astimezone().isoformat(),
            "mode": run_mode,
            "run": args.run if run_mode == "official_regular" else None,
            "workers": NUM_WORKERS,
            "amp_enabled": bool(state.amp_enabled),
        }
        with (run_dir / "RUN_VALID.json").open("w", encoding="utf-8") as file:
            json.dump(valid_marker, file, indent=2)

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
        print(f"Average GPU Edge Temperature: {average_edge_temp_c:.2f} C")
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
                with (run_dir / "amdsmi_api_errors.json").open(
                    "w", encoding="utf-8"
                ) as file:
                    json.dump(monitor.api_errors, file, indent=2)
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
                append_gpu_error_history(run_label, traceback_text)
                print(f"GPU-related failure appended to: {GPU_ERROR_HISTORY}")
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

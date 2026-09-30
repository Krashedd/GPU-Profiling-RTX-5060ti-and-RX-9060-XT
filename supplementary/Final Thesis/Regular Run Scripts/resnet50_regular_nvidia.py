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
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
import pynvml
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet50


SEED = 55
BATCH_SIZE = 256
EPOCHS = 10
LR = 0.01
MOMENTUM = 0.9
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
SAMPLE_INTERVAL_S = 0.2
WAIT_BETWEEN_RUNS_S = 10

PROJECT_DIR = Path.home() / "Thesis-Testing"
DATA_DIR = PROJECT_DIR / "Dataset" / "CIFAR10"
LOGS_DIR = PROJECT_DIR / "Logs"
TEMP_ROOT = Path.home() / "Downloads" / "Thesis-Testing-Temporary" / "ResNet50"
SMOKE_DIR = TEMP_ROOT / "smoke"
ONE_EPOCH_DIR = TEMP_ROOT / "one_epoch"
GPU_ERROR_HISTORY = LOGS_DIR / "resnet_gpu_error_history.log"
ALL_RUNS_STATUS_JSON = LOGS_DIR / "resnet_all_runs_status.json"
ALL_RUNS_STATUS_LOG = LOGS_DIR / "resnet_all_runs_status.log"
EXPECTED_GPU_NAME_FRAGMENT = "RTX 5060 Ti"
SCRIPT_REVISION = "2026-06-18 nvidia-regular-v1 clean-rebuild"

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

# Suppress only the known NumPy 2.4/Torchvision CIFAR pickle warning. This does
# not alter loading, transforms, or training.
warnings.filterwarnings(
    "ignore",
    message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
)


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


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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


def script_sha256() -> str:
    path = Path(__file__).resolve()
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_state_sha256(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, tensor in model.state_dict().items():
            cpu_tensor = tensor.detach().cpu().contiguous()
            digest.update(name.encode("utf-8"))
            digest.update(str(cpu_tensor.dtype).encode("utf-8"))
            digest.update(str(tuple(cpu_tensor.shape)).encode("utf-8"))
            digest.update(cpu_tensor.numpy().tobytes())
    return digest.hexdigest()


def training_order_sha256(
    dataset_size: int,
    epochs: int,
    max_batches_per_epoch: int | None,
) -> str:
    generator = torch.Generator()
    generator.manual_seed(SEED)
    digest = hashlib.sha256()

    for epoch_index in range(epochs):
        indices = torch.randperm(dataset_size, generator=generator)
        if max_batches_per_epoch is not None:
            indices = indices[: max_batches_per_epoch * BATCH_SIZE]
        digest.update(epoch_index.to_bytes(4, "little", signed=False))
        digest.update(indices.numpy().astype(np.int64, copy=False).tobytes())

    return digest.hexdigest()


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

    required_numeric = (
        "gpu_util_percent",
        "vram_used_mb",
        "vram_total_mb",
        "power_w",
        "gpu_temperature_c",
    )
    missing = []
    for column in required_numeric:
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
        df["gpu_temperature_c"],
        errors="coerce",
    ).dropna()

    problems = []
    if ((gpu_util < 0) | (gpu_util > 100)).any():
        problems.append("GPU utilization outside 0-100%")
    if vram_total.median() < 1000 or vram_total.median() > 200000:
        problems.append(
            f"implausible total VRAM ({vram_total.median():.2f} MB)"
        )
    if vram_used.max() < 100:
        problems.append(
            f"implausible used VRAM ({vram_used.max():.2f} MB maximum)"
        )
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
        else TEMP_ROOT / "resnet_gpu_error_history.log"
    )
    history_path.parent.mkdir(parents=True, exist_ok=True)
    with history_path.open("a", encoding="utf-8") as file:
        file.write("\n" + "=" * 80 + "\n")
        file.write(f"Timestamp: {datetime.now().astimezone().isoformat()}\n")
        file.write(f"Workload: ResNet-50 CIFAR-10 NVIDIA {run_label}\n")
        file.write(traceback_text.rstrip() + "\n")
    return history_path


def build_loaders():
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=True,
        download=False,
        transform=transform,
    )
    testset = torchvision.datasets.CIFAR10(
        root=DATA_DIR,
        train=False,
        download=False,
        transform=transform,
    )

    if len(trainset) != 50000 or len(testset) != 10000:
        raise RuntimeError(
            f"Unexpected CIFAR-10 sizes: train={len(trainset)}, "
            f"test={len(testset)}"
        )

    generator = torch.Generator()
    generator.manual_seed(SEED)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=NUM_WORKERS > 0,
    )
    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        worker_init_fn=seed_worker,
        persistent_workers=NUM_WORKERS > 0,
    )

    return trainset, testset, trainloader, testloader


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


def run_all_official() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    current_sha256 = script_sha256()
    verify_prerequisite_marker(
        SMOKE_DIR / "RUN_VALID.json",
        "smoke",
        current_sha256,
    )
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
    record("Starting sequential official ResNet-50 runs 1, 2, and 3.")
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
        run_dir = LOGS_DIR / f"resnetrun{run_number}"
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
                {
                    "status": "IN_PROGRESS_OR_FAILED",
                    "script_sha256": current_sha256,
                    "runs": statuses,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        if not marker_valid:
            record(
                f"Run {run_number} failed validation. Later runs were not "
                f"started. Details: {marker_error}"
            )
            raise SystemExit(result.returncode if result.returncode != 0 else 1)

        record(
            f"Official run {run_number} completed successfully in "
            f"{elapsed_s / 60.0:.2f} minutes."
        )

        if run_number < 3:
            record(f"Waiting {WAIT_BETWEEN_RUNS_S} seconds before the next run.")
            time.sleep(WAIT_BETWEEN_RUNS_S)

    ALL_RUNS_STATUS_JSON.write_text(
        json.dumps(
            {
                "status": "COMPLETE",
                "completed_at": datetime.now().astimezone().isoformat(),
                "script_sha256": current_sha256,
                "runs": statuses,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    record("All three official ResNet-50 runs completed successfully.")


def run_single(mode: str, run_number: int | None = None) -> None:
    smoke = mode == "smoke"
    one_epoch_test = mode == "full_dataset_one_epoch_test"
    official = mode == "official_regular"

    if smoke:
        run_label = "smoke test"
        run_dir = SMOKE_DIR
        epochs_to_run = 1
        max_train_batches = 5
        max_validation_batches = 2
    elif one_epoch_test:
        run_label = "full-dataset one-epoch test"
        run_dir = ONE_EPOCH_DIR
        epochs_to_run = 1
        max_train_batches = None
        max_validation_batches = None
    elif official:
        if run_number not in (1, 2, 3):
            raise ValueError("Official run number must be 1, 2, or 3.")
        run_label = f"regular run {run_number}"
        run_dir = LOGS_DIR / f"resnetrun{run_number}"
        epochs_to_run = EPOCHS
        max_train_batches = None
        max_validation_batches = None
    else:
        raise ValueError(f"Unknown run mode: {mode}")

    reset_directory(run_dir)
    log_file = (run_dir / "training_output.log").open(
        "w",
        encoding="utf-8",
        buffering=1,
    )
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    monitor = None
    telemetry_df = pd.DataFrame()

    try:
        set_seed(SEED)
        current_script_sha256 = script_sha256()

        if not DATA_DIR.exists():
            raise RuntimeError(f"Dataset directory does not exist: {DATA_DIR}")
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot detect the NVIDIA GPU.")
        if torch.version.cuda is None:
            raise RuntimeError("This PyTorch build does not report a CUDA runtime.")

        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        compute_capability = torch.cuda.get_device_capability(0)
        if EXPECTED_GPU_NAME_FRAGMENT not in gpu_name:
            raise RuntimeError(
                f"Unexpected GPU selected: {gpu_name}. Expected "
                f"{EXPECTED_GPU_NAME_FRAGMENT}."
            )

        trainset, testset, trainloader, testloader = build_loaders()
        order_sha256 = training_order_sha256(
            len(trainset),
            epochs_to_run,
            max_train_batches,
        )

        print("=" * 80)
        print(f"ResNet-50 CIFAR-10 NVIDIA {run_label}")
        print("=" * 80)
        print(f"Timestamp: {datetime.now().astimezone().isoformat()}")
        print(f"Output directory: {run_dir}")
        print(f"Dataset directory: {DATA_DIR}")
        print(f"Python: {platform.python_version()}")
        print(f"Platform: {platform.platform()}")
        print(f"PyTorch: {torch.__version__}")
        print(f"Torchvision: {torchvision.__version__}")
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"NVML Python: {package_version('nvidia-ml-py')}")
        print(f"GPU: {gpu_name}")
        print(f"Compute capability: {compute_capability[0]}.{compute_capability[1]}")
        print(f"Script SHA-256: {current_script_sha256}")
        print(f"Training-order SHA-256: {order_sha256}")
        print()

        # Create the model only after all dataset and environment checks. The
        # fingerprint is computed before the model is trained or timed.
        model = resnet50(num_classes=10)
        initial_model_sha256 = model_state_sha256(model)
        model = model.to(device)

        monitor = TelemetryMonitor(interval_s=SAMPLE_INTERVAL_S)
        if monitor.gpu_name != gpu_name:
            raise RuntimeError(
                "PyTorch and NVML selected different GPU names: "
                f"PyTorch={gpu_name!r}, NVML={monitor.gpu_name!r}"
            )

        print(f"NVIDIA driver (NVML): {monitor.driver_version}")
        print(f"Initial-model SHA-256: {initial_model_sha256}")
        print()

        config = {
            "mode": mode,
            "run": run_number if official else None,
            "seed": SEED,
            "model": "torchvision.models.resnet50",
            "model_initialization": "from scratch",
            "initial_model_sha256": initial_model_sha256,
            "training_order_sha256": order_sha256,
            "dataset": "CIFAR-10",
            "dataset_path": str(DATA_DIR),
            "training_images": len(trainset),
            "test_images": len(testset),
            "batch_size": BATCH_SIZE,
            "epochs": epochs_to_run,
            "official_epochs": EPOCHS,
            "optimizer": "SGD",
            "learning_rate": LR,
            "momentum": MOMENTUM,
            "weight_decay": WEIGHT_DECAY,
            "loss_function": "CrossEntropyLoss",
            "precision": "AMP FP16",
            "transform": "ToTensor only",
            "num_workers": NUM_WORKERS,
            "pin_memory": True,
            "persistent_workers": NUM_WORKERS > 0,
            "training_shuffle": True,
            "validation_shuffle": False,
            "telemetry_interval_s": SAMPLE_INTERVAL_S,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "cuda_runtime": torch.version.cuda,
            "nvidia_driver": monitor.driver_version,
            "nvml_python_version": package_version("nvidia-ml-py"),
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "psutil_version": psutil.__version__,
            "gpu": gpu_name,
            "compute_capability": (
                f"{compute_capability[0]}.{compute_capability[1]}"
            ),
            "max_train_batches": max_train_batches,
            "max_validation_batches": max_validation_batches,
            "script_revision": SCRIPT_REVISION,
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": current_script_sha256,
            "validation_metric": "top-1 accuracy",
        }
        with (run_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=LR,
            momentum=MOMENTUM,
            weight_decay=WEIGHT_DECAY,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        if not scaler.is_enabled():
            raise RuntimeError("AMP GradScaler is not enabled.")

        epoch_rows = []
        total_samples_processed = 0
        total_batches_processed = 0
        first_batch_output_dtype = "not_recorded"

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        print("Starting monitored AMP training...")
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
                        f"First-batch output dtype: {first_batch_output_dtype}"
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
        total_training_time = time.perf_counter() - total_start
        monitor.stop()

        print(f"Total Training Time: {total_training_time:.2f}s")
        print("Training complete.")
        print()

        pd.DataFrame(epoch_rows).to_csv(
            run_dir / "epoch_metrics.csv",
            index=False,
        )
        telemetry_df = monitor.dataframe()
        telemetry_df.to_csv(run_dir / "telemetry.csv", index=False)
        with (run_dir / "nvml_api_errors.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(monitor.api_errors, file, indent=2, sort_keys=True)
        write_nvml_error_log(run_dir, monitor)
        validate_telemetry(telemetry_df)

        expected_batches_per_epoch = math.ceil(len(trainset) / BATCH_SIZE)
        if smoke:
            expected_samples = min(
                len(trainset),
                max_train_batches * BATCH_SIZE,
            )
            expected_batches = max_train_batches
        else:
            expected_samples = len(trainset) * epochs_to_run
            expected_batches = expected_batches_per_epoch * epochs_to_run

        if len(epoch_rows) != epochs_to_run:
            raise RuntimeError(
                f"Incomplete run: expected {epochs_to_run} epochs, "
                f"got {len(epoch_rows)}."
            )
        if total_samples_processed != expected_samples:
            raise RuntimeError(
                f"Incomplete run: expected {expected_samples} training "
                f"samples, got {total_samples_processed}."
            )
        if total_batches_processed != expected_batches:
            raise RuntimeError(
                f"Incomplete run: expected {expected_batches} batches, "
                f"got {total_batches_processed}."
            )
        if not all(math.isfinite(row["loss"]) for row in epoch_rows):
            raise RuntimeError("One or more epoch losses are non-finite.")
        if monitor.thermal_throttle_seen:
            raise RuntimeError(
                "NVML detected a thermal throttling condition during training."
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
            min(len(testset), max_validation_batches * BATCH_SIZE)
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
        mean_epoch_loop_time = sum(
            row["epoch_time_s"] for row in epoch_rows
        ) / len(epoch_rows)

        average_power_w = series_stat(telemetry_df, "power_w", "mean")
        average_gpu_util = series_stat(
            telemetry_df,
            "gpu_util_percent",
            "mean",
        )
        average_vram_mb = series_stat(telemetry_df, "vram_used_mb", "mean")
        peak_vram_mb = series_stat(telemetry_df, "vram_used_mb", "max")
        average_gpu_temp_c = series_stat(
            telemetry_df,
            "gpu_temperature_c",
            "mean",
        )
        peak_gpu_temp_c = series_stat(
            telemetry_df,
            "gpu_temperature_c",
            "max",
        )
        average_gfx_clock_mhz = series_stat(
            telemetry_df,
            "graphics_clock_mhz",
            "mean",
        )
        average_mem_clock_mhz = series_stat(
            telemetry_df,
            "memory_clock_mhz",
            "mean",
        )
        average_cpu_util = series_stat(
            telemetry_df,
            "cpu_util_percent",
            "mean",
        )
        average_cpu_temp_c = series_stat(
            telemetry_df,
            "cpu_package_temperature_c",
            "mean",
        )
        average_system_ram_mb = series_stat(
            telemetry_df,
            "system_ram_used_mb",
            "mean",
        )
        peak_system_ram_mb = series_stat(
            telemetry_df,
            "system_ram_used_mb",
            "max",
        )
        average_disk_read_mb_s = series_stat(
            telemetry_df,
            "disk_read_mb_s",
            "mean",
        )
        average_disk_write_mb_s = series_stat(
            telemetry_df,
            "disk_write_mb_s",
            "mean",
        )
        peak_disk_read_mb_s = series_stat(
            telemetry_df,
            "disk_read_mb_s",
            "max",
        )
        peak_disk_write_mb_s = series_stat(
            telemetry_df,
            "disk_write_mb_s",
            "max",
        )
        torch_peak_allocated_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
        )

        performance_per_watt = (
            throughput / average_power_w
            if not math.isnan(average_power_w) and average_power_w > 0
            else float("nan")
        )

        stability_notes = (
            "Completed successfully; no Python, CUDA, NVIDIA driver, NVML, "
            "OOM, or thermal fatal errors detected. No NVML thermal "
            "throttling condition was observed."
        )
        if monitor.api_error_count:
            stability_notes += (
                f" NVML recorded {monitor.api_error_count} recoverable API "
                "query errors; required telemetry fields remained valid."
            )
        else:
            stability_notes += " All required NVML telemetry queries succeeded."

        summary = {
            "mode": mode,
            "run": run_number if official else None,
            "throughput_samples_s": throughput,
            "batch_latency_ms_batch": batch_latency_ms,
            "total_training_time_s": total_training_time,
            "average_epoch_time_s": average_epoch_time,
            "mean_measured_epoch_loop_time_s": mean_epoch_loop_time,
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
            "telemetry_interval_s": SAMPLE_INTERVAL_S,
            "nvml_api_error_count": monitor.api_error_count,
            "nvml_api_errors_json": json.dumps(
                monitor.api_errors,
                sort_keys=True,
            ),
            "amp_grad_scaler_enabled": scaler.is_enabled(),
            "amp_first_batch_output_dtype": first_batch_output_dtype,
            "initial_model_sha256": initial_model_sha256,
            "training_order_sha256": order_sha256,
            "script_sha256": current_script_sha256,
            "stability_error_notes": stability_notes,
        }
        pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)

        with (run_dir / "summary.txt").open("w", encoding="utf-8") as file:
            file.write(f"ResNet-50 NVIDIA {run_label.title()}\n")
            file.write("=" * 64 + "\n")
            file.write(f"Throughput: {format_value(throughput)} samples/s\n")
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
                f"{format_value(performance_per_watt)} samples/s/W\n"
            )
            file.write(
                f"Average GPU Utilization: {format_value(average_gpu_util)} %\n"
            )
            file.write(
                f"Average VRAM Usage: {format_value(average_vram_mb)} MB\n"
            )
            file.write(f"Peak VRAM Usage: {format_value(peak_vram_mb)} MB\n")
            file.write(
                f"Average GPU Temperature: {format_value(average_gpu_temp_c)} C\n"
            )
            file.write(
                f"Final Validation Top-1 Accuracy: {validation_accuracy:.4f}\n"
            )
            file.write(
                f"Peak System RAM: {format_value(peak_system_ram_mb)} MB\n"
            )
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
            file.write(
                "AMP: GradScaler enabled; first-batch output dtype "
                f"{first_batch_output_dtype}\n"
            )
            file.write(f"Initial Model SHA-256: {initial_model_sha256}\n")
            file.write(f"Training Order SHA-256: {order_sha256}\n")
            file.write(f"Script SHA-256: {current_script_sha256}\n")
            file.write(f"Stability / Error Notes: {stability_notes}\n")

        # Write the validity marker last. It is only created after all workload,
        # telemetry, validation, and output checks have succeeded.
        valid_marker = {
            "status": "VALID",
            "timestamp": datetime.now().astimezone().isoformat(),
            "mode": mode,
            "run": run_number if official else None,
            "epochs_completed": len(epoch_rows),
            "total_samples_processed": total_samples_processed,
            "total_batches_processed": total_batches_processed,
            "validation_samples": validation_total,
            "validation_score": validation_accuracy,
            "telemetry_samples": len(telemetry_df),
            "thermal_throttling_observed": monitor.thermal_throttle_seen,
            "initial_model_sha256": initial_model_sha256,
            "training_order_sha256": order_sha256,
            "script_sha256": current_script_sha256,
        }
        with (run_dir / "RUN_VALID.json").open("w", encoding="utf-8") as file:
            json.dump(valid_marker, file, indent=2)

        outputs_valid, missing_or_empty = output_files_are_nonempty(run_dir)
        if not outputs_valid:
            (run_dir / "RUN_VALID.json").unlink(missing_ok=True)
            raise RuntimeError(
                "One or more required output files are missing or empty: "
                + ", ".join(missing_or_empty)
            )

        print()
        print("=" * 80)
        if smoke:
            print("SMOKE TEST PASSED")
        elif one_epoch_test:
            print("ONE-EPOCH TEST PASSED")
        else:
            print("OFFICIAL RUN VALID")
        print("=" * 80)
        print(f"Throughput: {throughput:.2f} samples/s")
        print(f"Batch Latency: {batch_latency_ms:.2f} ms/batch")
        print(f"Total Training Time: {total_training_time:.2f} s")
        print(f"Average GPU Power Draw: {average_power_w:.2f} W")
        print(f"Average GPU Utilization: {average_gpu_util:.2f} %")
        print(f"Average VRAM Usage: {average_vram_mb:.2f} MB")
        print(f"Peak VRAM Usage: {peak_vram_mb:.2f} MB")
        print(f"Average GPU Temperature: {average_gpu_temp_c:.2f} C")
        print(f"Final Validation Top-1 Accuracy: {validation_accuracy:.4f}")
        print(f"Stability / Error Notes: {stability_notes}")
        print(f"Results saved to: {run_dir}")

    except Exception:
        if monitor is not None:
            try:
                monitor.stop()
            except Exception:
                pass
            try:
                partial_df = monitor.dataframe()
                if not partial_df.empty:
                    partial_df.to_csv(
                        run_dir / "telemetry_partial.csv",
                        index=False,
                    )
                with (run_dir / "nvml_api_errors.json").open(
                    "w",
                    encoding="utf-8",
                ) as file:
                    json.dump(monitor.api_errors, file, indent=2, sort_keys=True)
                write_nvml_error_log(run_dir, monitor)
            except Exception:
                pass

        traceback_text = traceback.format_exc()
        print()
        print("RUN FAILED")
        print(traceback_text)

        try:
            (run_dir / "RUN_FAILED.txt").write_text(
                traceback_text,
                encoding="utf-8",
            )
            (run_dir / "RUN_VALID.json").unlink(missing_ok=True)
        except Exception:
            pass

        if is_gpu_related_error(traceback_text):
            try:
                history_path = append_gpu_error_history(
                    run_label,
                    traceback_text,
                    official=official,
                )
                print(f"GPU-related failure appended to: {history_path}")
            except Exception as history_error:
                print(f"Could not append GPU error history: {history_error}")
        else:
            print("Failure was not automatically classified as GPU-related.")

        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ResNet-50 CIFAR-10 NVIDIA regular benchmark"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", type=int, choices=(1, 2, 3))
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Run five training batches and two validation batches.",
    )
    mode.add_argument(
        "--one-epoch-test",
        action="store_true",
        help="Run one complete training epoch and full validation.",
    )
    mode.add_argument(
        "--all-runs",
        action="store_true",
        help=(
            "Run official Runs 1-3 sequentially after verifying matching "
            "smoke and one-epoch validity markers."
        ),
    )
    args = parser.parse_args()

    if args.all_runs:
        run_all_official()
    elif args.smoke:
        run_single("smoke")
    elif args.one_epoch_test:
        run_single("full_dataset_one_epoch_test")
    else:
        run_single("official_regular", run_number=args.run)


if __name__ == "__main__":
    main()

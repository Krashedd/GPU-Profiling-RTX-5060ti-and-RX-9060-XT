#!/usr/bin/env python3

import argparse
import importlib.metadata
import json
import math
import platform
import random
import shutil
import statistics
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

PROJECT_DIR = Path.home() / "Thesis-Testing"
DATA_DIR = PROJECT_DIR / "Dataset" / "CIFAR10"
LOGS_DIR = PROJECT_DIR / "Logs"
GPU_ERROR_HISTORY = LOGS_DIR / "resnet_gpu_error_history.log"


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
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
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
        if entries:
            values = [numeric(entry.current) for entry in entries]
            values = [value for value in values if not math.isnan(value)]
            if values:
                return max(values)

    return float("nan")


class TelemetryMonitor:
    def __init__(self, interval_s: float = 0.2):
        self.interval_s = interval_s
        self.rows = []
        self.stop_event = threading.Event()
        self.thread = None
        self.start_time = None
        self.running = False
        self.initialized = False
        self.api_error_counts = {}
        self.thermal_limit_seen = False
        self.temperature_limits_c = {}

        amdsmi.amdsmi_init()
        self.initialized = True
        devices = amdsmi.amdsmi_get_processor_handles()
        if not devices:
            self.close()
            raise RuntimeError("AMD SMI returned no GPU handles.")
        self.gpu = devices[0]

        for label, sensor_type in (
            ("edge", amdsmi.AmdSmiTemperatureType.EDGE),
            ("hotspot", amdsmi.AmdSmiTemperatureType.HOTSPOT),
            ("memory", amdsmi.AmdSmiTemperatureType.VRAM),
        ):
            self.temperature_limits_c[label] = self._temperature_c(
                sensor_type,
                amdsmi.AmdSmiTemperatureMetric.CRITICAL,
                f"temperature_{label}_critical",
            )

    def _safe_api(self, name, function, default):
        try:
            return function()
        except Exception:
            self.api_error_counts[name] = self.api_error_counts.get(name, 0) + 1
            return default

    @property
    def api_error_count(self) -> int:
        return sum(self.api_error_counts.values())

    def _temperature_c(self, sensor_type, metric, query_name) -> float:
        raw = self._safe_api(
            query_name,
            lambda: amdsmi.amdsmi_get_temp_metric(
                self.gpu,
                sensor_type,
                metric,
            ),
            float("nan"),
        )
        raw_value = numeric(raw)
        if math.isnan(raw_value):
            return raw_value

        # AMD SMI's C interface documents millidegrees Celsius, while some
        # Python builds return already-normalized degrees Celsius. Handle both.
        return raw_value / 1000.0 if abs(raw_value) >= 1000.0 else raw_value

    def start(self) -> None:
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
                "gpu_vram_usage",
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
                "memory_clock",
                lambda: amdsmi.amdsmi_get_clock_info(
                    self.gpu, amdsmi.AmdSmiClkType.MEM
                ),
                {},
            )

            edge_temp_c = self._temperature_c(
                amdsmi.AmdSmiTemperatureType.EDGE,
                amdsmi.AmdSmiTemperatureMetric.CURRENT,
                "temperature_edge_current",
            )
            hotspot_temp_c = self._temperature_c(
                amdsmi.AmdSmiTemperatureType.HOTSPOT,
                amdsmi.AmdSmiTemperatureMetric.CURRENT,
                "temperature_hotspot_current",
            )
            memory_temp_c = self._temperature_c(
                amdsmi.AmdSmiTemperatureType.VRAM,
                amdsmi.AmdSmiTemperatureMetric.CURRENT,
                "temperature_memory_current",
            )

            thermal_limit_active = int(
                any(
                    not math.isnan(current)
                    and not math.isnan(limit)
                    and limit > 0
                    and current >= limit
                    for current, limit in (
                        (edge_temp_c, self.temperature_limits_c.get("edge", float("nan"))),
                        (hotspot_temp_c, self.temperature_limits_c.get("hotspot", float("nan"))),
                        (memory_temp_c, self.temperature_limits_c.get("memory", float("nan"))),
                    )
                )
            )
            self.thermal_limit_seen |= bool(thermal_limit_active)

            # amdsmi_get_gpu_vram_usage returns both values directly in MB.
            vram_used_mb = numeric(vram.get("vram_used"))
            vram_total_mb = numeric(vram.get("vram_total"))
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
                    "vram_used_mb": vram_used_mb,
                    "vram_total_mb": vram_total_mb,
                    "power_w": first_numeric(
                        power,
                        (
                            "socket_power",
                            "average_socket_power",
                            "current_socket_power",
                        ),
                    ),
                    "gpu_edge_temperature_c": edge_temp_c,
                    "gpu_hotspot_temperature_c": hotspot_temp_c,
                    "gpu_memory_temperature_c": memory_temp_c,
                    "gpu_edge_critical_temperature_c": self.temperature_limits_c.get(
                        "edge", float("nan")
                    ),
                    "gpu_hotspot_critical_temperature_c": self.temperature_limits_c.get(
                        "hotspot", float("nan")
                    ),
                    "gpu_memory_critical_temperature_c": self.temperature_limits_c.get(
                        "memory", float("nan")
                    ),
                    "graphics_clock_mhz": numeric(gfx_clock.get("clk")),
                    "memory_clock_mhz": numeric(mem_clock.get("clk")),
                    "cpu_util_percent": psutil.cpu_percent(interval=None),
                    "cpu_package_temperature_c": cpu_package_temperature_c(),
                    "system_ram_used_mb": ram.used / (1024**2),
                    "system_ram_percent": ram.percent,
                    "disk_read_mb_s": read_mb_s,
                    "disk_write_mb_s": write_mb_s,
                    "thermal_limit_active": thermal_limit_active,
                    "amdsmi_api_error_count_cumulative": self.api_error_count,
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
    if operation == "min":
        return float(values.min())
    raise ValueError(f"Unsupported operation: {operation}")


def format_value(value, decimals: int = 2) -> str:
    value = numeric(value)
    if math.isnan(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def is_gpu_related_error(text: str) -> bool:
    lowered = text.lower()
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
    )
    return any(term in lowered for term in terms)


def append_gpu_error_history(run_label: str, traceback_text: str) -> None:
    GPU_ERROR_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with GPU_ERROR_HISTORY.open("a", encoding="utf-8") as file:
        file.write("\n" + "=" * 80 + "\n")
        file.write(f"Timestamp: {datetime.now().astimezone().isoformat()}\n")
        file.write(f"Workload: ResNet-50 CIFAR-10 AMD {run_label}\n")
        file.write(traceback_text.rstrip() + "\n")


def read_rocm_version() -> str:
    version_file = Path("/opt/rocm/.info/version")
    try:
        return version_file.read_text(encoding="utf-8").strip()
    except OSError:
        return "Unavailable"


def validate_telemetry(df: pd.DataFrame) -> None:
    if df.empty:
        raise RuntimeError("No telemetry samples were collected.")

    required_numeric = (
        "gpu_util_percent",
        "vram_used_mb",
        "vram_total_mb",
        "power_w",
        "gpu_edge_temperature_c",
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
    edge_temp = pd.to_numeric(
        df["gpu_edge_temperature_c"], errors="coerce"
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
    if edge_temp.mean() < 10 or edge_temp.mean() > 120:
        problems.append(
            f"implausible GPU edge temperature ({edge_temp.mean():.2f} C mean)"
        )

    if problems:
        raise RuntimeError(
            "Telemetry validation failed: " + "; ".join(problems)
        )


def build_loaders(smoke: bool):
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ResNet-50 CIFAR-10 AMD regular benchmark"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--run", type=int, choices=(1, 2, 3))
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Run one short disposable validation pass in /tmp.",
    )
    args = parser.parse_args()

    smoke = bool(args.smoke)
    run_label = "smoke test" if smoke else f"regular run {args.run}"
    run_dir = (
        Path(tempfile.gettempdir()) / "resnet50_amd_smoke"
        if smoke
        else LOGS_DIR / f"resnetrun{args.run}"
    )

    reset_directory(run_dir)

    log_file = (run_dir / "training_output.log").open(
        "w", encoding="utf-8", buffering=1
    )
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    monitor = None
    telemetry_df = pd.DataFrame()

    try:
        set_seed(SEED)

        if not DATA_DIR.exists():
            raise RuntimeError(f"Dataset directory does not exist: {DATA_DIR}")
        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot detect the AMD GPU.")
        if torch.version.hip is None:
            raise RuntimeError("This PyTorch build does not report a HIP runtime.")

        device = torch.device("cuda:0")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_arch = torch.cuda.get_device_properties(0).gcnArchName
        epochs_to_run = 1 if smoke else EPOCHS
        max_train_batches = 5 if smoke else None
        max_validation_batches = 2 if smoke else None

        print("=" * 76)
        print(f"ResNet-50 CIFAR-10 AMD {run_label}")
        print("=" * 76)
        print(f"Timestamp: {datetime.now().astimezone().isoformat()}")
        print(f"Output directory: {run_dir}")
        print(f"Dataset directory: {DATA_DIR}")
        print(f"Python: {platform.python_version()}")
        print(f"Platform: {platform.platform()}")
        print(f"PyTorch: {torch.__version__}")
        print(f"Torchvision: {torchvision.__version__}")
        print(f"HIP runtime: {torch.version.hip}")
        print(f"System ROCm: {read_rocm_version()}")
        print(f"AMD SMI Python: {importlib.metadata.version('amdsmi')}")
        print(f"GPU: {gpu_name}")
        print(f"GPU architecture: {gpu_arch}")
        print()

        if "9060 XT" not in gpu_name:
            raise RuntimeError(
                f"Unexpected GPU selected: {gpu_name}. Expected RX 9060 XT."
            )

        trainset, testset, trainloader, testloader = build_loaders(smoke)

        config = {
            "mode": "smoke" if smoke else "official_regular",
            "run": None if smoke else args.run,
            "seed": SEED,
            "model": "torchvision.models.resnet50",
            "dataset": "CIFAR-10",
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
            "telemetry_interval_s": SAMPLE_INTERVAL_S,
            "torch_version": torch.__version__,
            "torchvision_version": torchvision.__version__,
            "hip_runtime": torch.version.hip,
            "system_rocm": read_rocm_version(),
            "amdsmi_version": importlib.metadata.version("amdsmi"),
            "gpu": gpu_name,
            "gpu_architecture": gpu_arch,
            "max_train_batches": max_train_batches,
            "max_validation_batches": max_validation_batches,
        }
        with (run_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

        model = resnet50(num_classes=10).to(device)
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

        monitor = TelemetryMonitor(interval_s=SAMPLE_INTERVAL_S)
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
                if max_train_batches is not None and batch_index >= max_train_batches:
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
            run_dir / "epoch_metrics.csv", index=False
        )

        telemetry_df = monitor.dataframe()
        telemetry_df.to_csv(run_dir / "telemetry.csv", index=False)
        with (run_dir / "amdsmi_api_errors.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(monitor.api_error_counts, file, indent=2, sort_keys=True)
        validate_telemetry(telemetry_df)

        expected_batches_per_epoch = math.ceil(len(trainset) / BATCH_SIZE)
        if not smoke:
            expected_samples = len(trainset) * EPOCHS
            expected_batches = expected_batches_per_epoch * EPOCHS
            if len(epoch_rows) != EPOCHS:
                raise RuntimeError(
                    f"Incomplete run: expected {EPOCHS} epochs, got {len(epoch_rows)}."
                )
            if total_samples_processed != expected_samples:
                raise RuntimeError(
                    "Incomplete run: expected "
                    f"{expected_samples} training samples, got "
                    f"{total_samples_processed}."
                )
            if total_batches_processed != expected_batches:
                raise RuntimeError(
                    "Incomplete run: expected "
                    f"{expected_batches} batches, got "
                    f"{total_batches_processed}."
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
                "Incomplete validation: expected "
                f"{len(testset)} samples, got {validation_total}."
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
        peak_hotspot_temp_c = series_stat(
            telemetry_df, "gpu_hotspot_temperature_c", "max"
        )
        average_memory_temp_c = series_stat(
            telemetry_df, "gpu_memory_temperature_c", "mean"
        )
        average_gfx_clock_mhz = series_stat(
            telemetry_df, "graphics_clock_mhz", "mean"
        )
        average_mem_clock_mhz = series_stat(
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
        torch_peak_allocated_mb = (
            torch.cuda.max_memory_allocated(device) / (1024**2)
        )

        performance_per_watt = (
            throughput / average_power_w
            if not math.isnan(average_power_w) and average_power_w > 0
            else float("nan")
        )

        stability_notes = (
            "Completed successfully; no Python, HIP, ROCm, or AMD SMI "
            "fatal errors detected."
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
                f" AMD SMI had {monitor.api_error_count} recoverable API "
                "query errors."
            )
        else:
            stability_notes += " All AMD SMI telemetry queries succeeded."

        summary = {
            "mode": "smoke" if smoke else "official_regular",
            "run": None if smoke else args.run,
            "throughput_samples_s": throughput,
            "batch_latency_ms_batch": batch_latency_ms,
            "total_training_time_s": total_training_time,
            "average_epoch_time_s": average_epoch_time,
            "average_gpu_power_w": average_power_w,
            "performance_per_watt_samples_s_w": performance_per_watt,
            "average_gpu_util_percent": average_gpu_util,
            "average_vram_usage_mb": average_vram_mb,
            "peak_vram_usage_mb": peak_vram_mb,
            "torch_peak_allocated_memory_mb": torch_peak_allocated_mb,
            "average_gpu_edge_temperature_c": average_edge_temp_c,
            "average_gpu_hotspot_temperature_c": average_hotspot_temp_c,
            "peak_gpu_hotspot_temperature_c": peak_hotspot_temp_c,
            "average_gpu_memory_temperature_c": average_memory_temp_c,
            "average_graphics_clock_mhz": average_gfx_clock_mhz,
            "average_memory_clock_mhz": average_mem_clock_mhz,
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
            "telemetry_interval_s": SAMPLE_INTERVAL_S,
            "amdsmi_api_error_count": monitor.api_error_count,
            "amdsmi_api_error_counts_json": json.dumps(
                monitor.api_error_counts, sort_keys=True
            ),
            "amp_grad_scaler_enabled": scaler.is_enabled(),
            "amp_first_batch_output_dtype": first_batch_output_dtype,
            "stability_error_notes": stability_notes,
        }

        pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)

        with (run_dir / "summary.txt").open("w", encoding="utf-8") as file:
            file.write(f"ResNet-50 AMD {run_label.title()}\n")
            file.write("=" * 60 + "\n")
            file.write(f"Throughput: {format_value(throughput)} samples/s\n")
            file.write(
                f"Batch Latency: {format_value(batch_latency_ms)} ms/batch\n"
            )
            file.write(
                "Total Training Time: "
                f"{format_value(total_training_time)} s\n"
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
                "Average GPU Utilization: "
                f"{format_value(average_gpu_util)} %\n"
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
                f"Final Validation Score: {validation_accuracy:.4f}\n"
            )
            file.write(
                f"Peak System RAM: {format_value(peak_system_ram_mb)} MB\n"
            )
            file.write(
                "Average System RAM: "
                f"{format_value(average_system_ram_mb)} MB\n"
            )
            file.write(
                "Average CPU Utilization: "
                f"{format_value(average_cpu_util)} %\n"
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
            file.write(f"Stability / Error Notes: {stability_notes}\n")

        valid_marker = {
            "status": "VALID",
            "timestamp": datetime.now().astimezone().isoformat(),
            "mode": "smoke" if smoke else "official_regular",
            "run": None if smoke else args.run,
        }
        with (run_dir / "RUN_VALID.json").open("w", encoding="utf-8") as file:
            json.dump(valid_marker, file, indent=2)

        print()
        print("=" * 76)
        print("SMOKE TEST PASSED" if smoke else "OFFICIAL RUN VALID")
        print("=" * 76)
        print(f"Throughput: {throughput:.2f} samples/s")
        print(f"Batch Latency: {batch_latency_ms:.2f} ms/batch")
        print(f"Total Training Time: {total_training_time:.2f} s")
        print(f"Average GPU Power Draw: {average_power_w:.2f} W")
        print(f"Average GPU Utilization: {average_gpu_util:.2f} %")
        print(f"Average VRAM Usage: {average_vram_mb:.2f} MB")
        print(f"Peak VRAM Usage: {peak_vram_mb:.2f} MB")
        print(f"Average GPU Edge Temperature: {average_edge_temp_c:.2f} C")
        print(f"Final Validation Score: {validation_accuracy:.4f}")
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
                        run_dir / "telemetry_partial.csv", index=False
                    )
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

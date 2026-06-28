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
import threading
import time
import traceback
from datetime import datetime
from pathlib import Path

import pynvml
import numpy as np
import pandas as pd
import psutil
import torch
from datasets import DatasetDict, load_dataset, load_from_disk
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
)


# -----------------------------------------------------------------------------
# Fixed thesis configuration
# -----------------------------------------------------------------------------
SEED = 55
MODEL_SOURCE = "distilbert-base-uncased"
DATASET_SOURCE = "nyu-mll/glue"
DATASET_CONFIG = "sst2"
BATCH_SIZE = 64
EPOCHS = 10
MAX_LENGTH = 128
LEARNING_RATE = 5e-5
NUM_WORKERS = 4
SAMPLE_INTERVAL_S = 0.2
AMP_ENABLED = True

EXPECTED_TRAIN_SAMPLES = 67_349
EXPECTED_VALIDATION_SAMPLES = 872

PROJECT_DIR = Path.home() / "Thesis-Testing"
DATASET_DIR = PROJECT_DIR / "Dataset" / "SST2"
RAW_DATASET_DIR = DATASET_DIR / "hf_sst2_raw"
TOKENIZED_DATASET_DIR = DATASET_DIR / "tokenized_distilbert_dynamic_maxlen128"
MODEL_DIR = PROJECT_DIR / "scripts" / "distilbert-base-uncased-sst2-init-seed55"
LOGS_DIR = PROJECT_DIR / "Logs"
GPU_ERROR_HISTORY = LOGS_DIR / "distilbert_nvidia_gpu_error_history.log"
ALL_RUNS_STATUS_JSON = LOGS_DIR / "distilbert_all_runs_status.json"
ALL_RUNS_STATUS_LOG = LOGS_DIR / "distilbert_all_runs_status.log"

SCRIPT_REVISION = "2026-06-22 nvidia-regular-v1 dynamic-padding-token-metrics"
EXPECTED_DRIVER_VERSION = "580.167.08"
TEMP_ROOT = Path.home() / "Downloads" / "Thesis-Testing-Temporary" / "DistilBERT"


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
        self.thermal_throttle_seen = False

        pynvml.nvmlInit()
        self.initialized = True
        count = pynvml.nvmlDeviceGetCount()
        if count < 1:
            self.close()
            raise RuntimeError("NVML returned no GPU handles.")
        self.gpu = pynvml.nvmlDeviceGetHandleByIndex(0)
        self.gpu_name = text_value(pynvml.nvmlDeviceGetName(self.gpu))
        self.driver_version = text_value(pynvml.nvmlSystemGetDriverVersion())

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
                        self.gpu, pynvml.NVML_TEMPERATURE_GPU
                    ),
                    float("nan"),
                )
            )
            graphics_clock_mhz = numeric(
                self._safe_api(
                    "graphics_clock",
                    lambda: pynvml.nvmlDeviceGetClockInfo(
                        self.gpu, pynvml.NVML_CLOCK_GRAPHICS
                    ),
                    float("nan"),
                )
            )
            memory_clock_mhz = numeric(
                self._safe_api(
                    "memory_clock",
                    lambda: pynvml.nvmlDeviceGetClockInfo(
                        self.gpu, pynvml.NVML_CLOCK_MEM
                    ),
                    float("nan"),
                )
            )
            pstate = self._safe_api(
                "performance_state",
                lambda: pynvml.nvmlDeviceGetPerformanceState(self.gpu),
                None,
            )

            reasons_function = getattr(
                pynvml, "nvmlDeviceGetCurrentClocksEventReasons", None
            ) or getattr(
                pynvml, "nvmlDeviceGetCurrentClocksThrottleReasons", None
            )
            if reasons_function is None:
                reasons = 0
            else:
                reasons = int(
                    self._safe_api(
                        "clock_event_reasons",
                        lambda: reasons_function(self.gpu),
                        0,
                    ) or 0
                )

            thermal_mask = 0
            for name in (
                "nvmlClocksEventReasonSwThermalSlowdown",
                "nvmlClocksThrottleReasonSwThermalSlowdown",
                "nvmlClocksEventReasonHwThermalSlowdown",
                "nvmlClocksThrottleReasonHwThermalSlowdown",
            ):
                thermal_mask |= int(getattr(pynvml, name, 0) or 0)
            thermal_throttle_active = int(bool(thermal_mask and reasons & thermal_mask))
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
                    "graphics_clock_mhz": graphics_clock_mhz,
                    "memory_clock_mhz": memory_clock_mhz,
                    "performance_state": (
                        f"P{int(pstate)}" if pstate is not None else "N/A"
                    ),
                    "clock_event_reasons_bitmask": reasons,
                    "clock_event_reasons_hex": hex(reasons),
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
    raise ValueError(f"Unsupported operation: {operation}")


def format_value(value, decimals: int = 2) -> str:
    value = numeric(value)
    if math.isnan(value):
        return "N/A"
    return f"{value:.{decimals}f}"


def text_value(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def read_nvidia_driver_version() -> str:
    initialized = False
    try:
        pynvml.nvmlInit()
        initialized = True
        return text_value(pynvml.nvmlSystemGetDriverVersion())
    except Exception:
        return "Unavailable"
    finally:
        if initialized:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "Unavailable"


def validate_telemetry(df: pd.DataFrame) -> None:
    if df.empty:
        raise RuntimeError("No telemetry samples were collected.")

    required = (
        "gpu_util_percent",
        "vram_used_mb",
        "power_w",
        "gpu_temperature_c",
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


def is_gpu_related_error(text: str) -> bool:
    lowered = text.lower()
    terms = (
        "cuda", "cudnn", "gpu", "nvml", "nvidia", "out of memory",
        "xid", "illegal memory access", "cublas", "nccl",
    )
    return any(term in lowered for term in terms)


def append_gpu_error_history(run_label: str, traceback_text: str) -> None:
    GPU_ERROR_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with GPU_ERROR_HISTORY.open("a", encoding="utf-8") as file:
        file.write("\n" + "=" * 80 + "\n")
        file.write(f"Timestamp: {datetime.now().astimezone().isoformat()}\n")
        file.write(f"Workload: DistilBERT SST-2 NVIDIA {run_label}\n")
        file.write(traceback_text.rstrip() + "\n")


def assets_are_prepared() -> bool:
    model_ok = (
        MODEL_DIR.exists()
        and (MODEL_DIR / "config.json").exists()
        and any(
            (MODEL_DIR / filename).exists()
            for filename in ("model.safetensors", "pytorch_model.bin")
        )
        and (MODEL_DIR / "tokenizer_config.json").exists()
    )
    raw_ok = RAW_DATASET_DIR.exists() and (
        RAW_DATASET_DIR / "dataset_dict.json"
    ).exists()
    tokenized_ok = TOKENIZED_DATASET_DIR.exists() and (
        TOKENIZED_DATASET_DIR / "dataset_dict.json"
    ).exists()
    return model_ok and raw_ok and tokenized_ok


def prepare_assets(force: bool = False) -> None:
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 84)
    print("PREPARING DISTILBERT SST-2 ASSETS")
    print("=" * 84)
    print(f"Model source: {MODEL_SOURCE}")
    print(f"Dataset source: {DATASET_SOURCE}, configuration {DATASET_CONFIG}")
    print(f"Model destination: {MODEL_DIR}")
    print(f"Raw dataset destination: {RAW_DATASET_DIR}")
    print(f"Tokenized dataset destination: {TOKENIZED_DATASET_DIR}")
    print()

    if force and MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    if force and RAW_DATASET_DIR.exists():
        shutil.rmtree(RAW_DATASET_DIR)
    if force and TOKENIZED_DATASET_DIR.exists():
        shutil.rmtree(TOKENIZED_DATASET_DIR)

    if not MODEL_DIR.exists() or not (MODEL_DIR / "config.json").exists():
        print("Downloading tokenizer and deterministic initial model...")
        set_seed(SEED)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_SOURCE, use_fast=True)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_SOURCE,
            num_labels=2,
        )
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        tokenizer.save_pretrained(MODEL_DIR)
        model.save_pretrained(MODEL_DIR, safe_serialization=True)
        del model
    else:
        print("Local model already exists; keeping it unchanged.")

    if not RAW_DATASET_DIR.exists() or not (
        RAW_DATASET_DIR / "dataset_dict.json"
    ).exists():
        print("Downloading GLUE/SST-2 dataset...")
        raw = load_dataset(DATASET_SOURCE, DATASET_CONFIG)
        selected = DatasetDict(
            {
                "train": raw["train"],
                "validation": raw["validation"],
            }
        )
        selected.save_to_disk(RAW_DATASET_DIR)
    else:
        print("Local raw SST-2 dataset already exists; keeping it unchanged.")

    if not TOKENIZED_DATASET_DIR.exists() or not (
        TOKENIZED_DATASET_DIR / "dataset_dict.json"
    ).exists():
        print("Tokenizing SST-2 with truncation at 128 tokens and no fixed padding...")
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_DIR,
            use_fast=True,
            local_files_only=True,
        )
        raw = load_from_disk(RAW_DATASET_DIR)

        def tokenize_batch(batch):
            return tokenizer(
                batch["sentence"],
                padding=False,
                truncation=True,
                max_length=MAX_LENGTH,
                return_length=True,
            )

        tokenized = raw.map(
            tokenize_batch,
            batched=True,
            desc="Tokenizing SST-2",
        )
        removable = [
            column
            for column in ("sentence", "idx")
            if column in tokenized["train"].column_names
        ]
        if removable:
            tokenized = tokenized.remove_columns(removable)
        tokenized.save_to_disk(TOKENIZED_DATASET_DIR)
    else:
        print("Local tokenized SST-2 dataset already exists; keeping it unchanged.")

    verify_assets()
    manifest = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "model_source": MODEL_SOURCE,
        "dataset_source": DATASET_SOURCE,
        "dataset_config": DATASET_CONFIG,
        "seed_used_to_initialize_classification_head": SEED,
        "max_length": MAX_LENGTH,
        "padding": "dynamic_per_batch",
        "train_samples": EXPECTED_TRAIN_SAMPLES,
        "validation_samples": EXPECTED_VALIDATION_SAMPLES,
        "model_dir": str(MODEL_DIR),
        "raw_dataset_dir": str(RAW_DATASET_DIR),
        "tokenized_dataset_dir": str(TOKENIZED_DATASET_DIR),
        "transformers_version": package_version("transformers"),
        "datasets_version": package_version("datasets"),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "nvidia_driver": read_nvidia_driver_version(),
        "script_revision": SCRIPT_REVISION,
    }
    with (DATASET_DIR / "distilbert_asset_manifest.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(manifest, file, indent=2)

    print()
    print("ASSET PREPARATION VALID")


def verify_assets() -> DatasetDict:
    if not assets_are_prepared():
        raise RuntimeError(
            "DistilBERT/SST-2 assets are not fully prepared. Run this script "
            "once with --prepare before testing."
        )

    dataset = load_from_disk(TOKENIZED_DATASET_DIR)
    if not isinstance(dataset, DatasetDict):
        raise RuntimeError("Tokenized SST-2 path did not load as a DatasetDict.")
    if "train" not in dataset or "validation" not in dataset:
        raise RuntimeError("Tokenized dataset lacks train or validation split.")

    train_count = len(dataset["train"])
    validation_count = len(dataset["validation"])
    if train_count != EXPECTED_TRAIN_SAMPLES:
        raise RuntimeError(
            f"Unexpected SST-2 training count: {train_count}; "
            f"expected {EXPECTED_TRAIN_SAMPLES}."
        )
    if validation_count != EXPECTED_VALIDATION_SAMPLES:
        raise RuntimeError(
            f"Unexpected SST-2 validation count: {validation_count}; "
            f"expected {EXPECTED_VALIDATION_SAMPLES}."
        )

    required_columns = {"input_ids", "attention_mask", "label", "length"}
    for split_name in ("train", "validation"):
        missing = required_columns.difference(dataset[split_name].column_names)
        if missing:
            raise RuntimeError(
                f"{split_name} split lacks required columns: {sorted(missing)}"
            )

    sample_count = min(1000, len(dataset["train"]))
    stored_lengths = {
        len(dataset["train"][index]["input_ids"])
        for index in range(sample_count)
    }
    if len(stored_lengths) < 2:
        raise RuntimeError(
            "Tokenized dataset appears fixed-length. Expected variable-length "
            "sequences for dynamic per-batch padding."
        )
    if max(stored_lengths) > MAX_LENGTH:
        raise RuntimeError(
            f"Tokenized sequence exceeds max length {MAX_LENGTH}."
        )

    return dataset


class DynamicPaddingCollator:
    def __init__(self, tokenizer):
        self.base_collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            padding=True,
            max_length=MAX_LENGTH,
            pad_to_multiple_of=None,
            return_tensors="pt",
        )

    def __call__(self, features):
        cleaned = []
        valid_tokens = 0
        for feature in features:
            item = dict(feature)
            length = item.pop("length", None)
            if length is None:
                length = len(item["input_ids"])
            valid_tokens += int(length)
            cleaned.append(item)

        batch = self.base_collator(cleaned)
        batch["_valid_tokens"] = valid_tokens
        batch["_padded_tokens"] = int(batch["input_ids"].numel())
        return batch


def build_dataloaders(dataset: DatasetDict, smoke: bool):
    train_dataset = dataset["train"]
    validation_dataset = dataset["validation"]

    if smoke:
        train_dataset = train_dataset.select(range(min(256, len(train_dataset))))
        validation_dataset = validation_dataset.select(
            range(min(256, len(validation_dataset)))
        )

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR,
        use_fast=True,
        local_files_only=True,
    )
    dynamic_collator = DynamicPaddingCollator(tokenizer)

    generator = torch.Generator()
    generator.manual_seed(SEED)

    common = {
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "pin_memory": True,
        "persistent_workers": NUM_WORKERS > 0,
        "worker_init_fn": seed_worker,
        "collate_fn": dynamic_collator,
    }
    if NUM_WORKERS > 0:
        common["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        drop_last=False,
        **common,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **common,
    )
    return train_loader, validation_loader


def move_batch(batch, device):
    return {
        key: value.to(device, non_blocking=True)
        for key, value in batch.items()
    }


def validate_model(model, validation_loader, device):
    model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_correct = torch.zeros((), device=device, dtype=torch.int64)
    total_samples = 0

    with torch.no_grad():
        for batch in tqdm(
            validation_loader,
            desc="Final validation",
            leave=False,
            mininterval=1.0,
        ):
            batch.pop("_valid_tokens", None)
            batch.pop("_padded_tokens", None)
            batch = move_batch(batch, device)
            batch_size = int(batch["labels"].shape[0])
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=AMP_ENABLED,
            ):
                outputs = model(**batch)
                loss = outputs.loss
            predictions = outputs.logits.argmax(dim=-1)
            total_loss += loss.detach().double() * batch_size
            total_correct += (predictions == batch["labels"]).sum()
            total_samples += batch_size

    torch.cuda.synchronize()
    return {
        "loss": float((total_loss / total_samples).item()),
        "accuracy": float((total_correct.double() / total_samples).item()),
        "samples": total_samples,
    }


def run_single(mode: str, run_number=None) -> None:
    smoke = mode == "smoke"
    one_epoch_test = mode == "one_epoch_test"

    if smoke:
        run_label = "smoke test"
        run_dir = TEMP_ROOT / "smoke"
        epochs_to_run = 1
    elif one_epoch_test:
        run_label = "full-dataset one-epoch test"
        run_dir = TEMP_ROOT / "one_epoch"
        epochs_to_run = 1
    else:
        run_label = f"regular run {run_number}"
        run_dir = LOGS_DIR / f"distilbertrun{run_number}"
        epochs_to_run = EPOCHS

    reset_directory(run_dir)
    log_file = (run_dir / "training_output.log").open(
        "w", encoding="utf-8", buffering=1
    )
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_file)
    sys.stderr = Tee(original_stderr, log_file)

    monitor = None

    try:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("WANDB_DISABLED", "true")
        set_seed(SEED)

        if not torch.cuda.is_available():
            raise RuntimeError("PyTorch cannot detect the NVIDIA GPU.")
        if torch.version.cuda is None:
            raise RuntimeError("This PyTorch build does not report a CUDA runtime.")

        gpu_name = torch.cuda.get_device_name(0)
        gpu_arch = ".".join(str(part) for part in torch.cuda.get_device_capability(0))
        driver_version = read_nvidia_driver_version()
        if driver_version != EXPECTED_DRIVER_VERSION:
            raise RuntimeError(
                f"Unexpected NVIDIA driver: {driver_version}. "
                f"Expected frozen driver {EXPECTED_DRIVER_VERSION}."
            )
        if "5060 Ti" not in gpu_name:
            raise RuntimeError(
                f"Unexpected GPU selected: {gpu_name}. Expected RTX 5060 Ti."
            )

        dataset = verify_assets()
        train_loader, validation_loader = build_dataloaders(dataset, smoke=smoke)
        train_samples_per_epoch = len(train_loader.dataset)
        validation_samples = len(validation_loader.dataset)

        print("=" * 84)
        print(f"DistilBERT SST-2 NVIDIA {run_label}")
        print("=" * 84)
        print(f"Timestamp: {datetime.now().astimezone().isoformat()}")
        print(f"Output directory: {run_dir}")
        print(f"Model directory: {MODEL_DIR}")
        print(f"Dataset directory: {TOKENIZED_DATASET_DIR}")
        print(f"Python: {platform.python_version()}")
        print(f"Platform: {platform.platform()}")
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA runtime: {torch.version.cuda}")
        print(f"NVIDIA driver: {driver_version}")
        print(f"Transformers: {package_version('transformers')}")
        print(f"Datasets: {package_version('datasets')}")
        print(f"NVML Python: {package_version('nvidia-ml-py')}")
        print(f"GPU: {gpu_name}")
        print(f"GPU compute capability: {gpu_arch}")
        print()

        config = {
            "mode": mode,
            "run": run_number,
            "seed": SEED,
            "model": "DistilBERT base uncased for sequence classification",
            "model_source": MODEL_SOURCE,
            "model_path": str(MODEL_DIR),
            "dataset": "GLUE SST-2",
            "dataset_source": f"{DATASET_SOURCE}/{DATASET_CONFIG}",
            "dataset_path": str(TOKENIZED_DATASET_DIR),
            "training_samples": train_samples_per_epoch,
            "validation_samples": validation_samples,
            "batch_size": BATCH_SIZE,
            "max_length": MAX_LENGTH,
            "padding": "dynamic_per_batch",
            "pad_to_multiple_of": None,
            "epochs": epochs_to_run,
            "official_epochs": EPOCHS,
            "optimizer": "torch.optim.Adam",
            "learning_rate": LEARNING_RATE,
            "weight_decay": 0.0,
            "precision": "AMP ON / FP16 autocast",
            "amp_enabled": AMP_ENABLED,
            "workers": NUM_WORKERS,
            "device": 0,
            "telemetry_interval_s": SAMPLE_INTERVAL_S,
            "torch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "nvidia_driver": driver_version,
            "transformers_version": package_version("transformers"),
            "datasets_version": package_version("datasets"),
            "nvml_python_version": package_version("nvidia-ml-py"),
            "gpu": gpu_name,
            "gpu_compute_capability": gpu_arch,
            "validation_metric": "accuracy",
            "script_revision": SCRIPT_REVISION,
        }
        with (run_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

        device = torch.device("cuda:0")
        set_seed(SEED)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR,
            local_files_only=True,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=LEARNING_RATE,
        )
        scaler = torch.amp.GradScaler("cuda", enabled=AMP_ENABLED)
        if not scaler.is_enabled():
            raise RuntimeError("AMP was requested but GradScaler is disabled.")

        print("Training configuration verification")
        print(f"Batch size: {BATCH_SIZE}")
        print(f"Workers: {NUM_WORKERS}")
        print(f"Epochs: {epochs_to_run}")
        print(f"Optimizer: Adam")
        print(f"Learning rate: {LEARNING_RATE}")
        print(f"Maximum sequence length: {MAX_LENGTH}")
        print("Padding: dynamic per batch (no pad-to-multiple)")
        print(f"AMP enabled: {scaler.is_enabled()}")
        print(f"Model parameter dtype: {next(model.parameters()).dtype}")
        print()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        monitor = TelemetryMonitor()
        monitor.start()
        total_start = time.perf_counter()
        epoch_rows = []
        total_batches = 0
        total_samples_processed = 0
        total_valid_tokens = 0
        total_padded_tokens = 0

        for epoch_index in range(epochs_to_run):
            model.train()
            epoch_start = time.perf_counter()
            epoch_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
            epoch_samples = 0
            epoch_batches = 0
            epoch_valid_tokens = 0
            epoch_padded_tokens = 0

            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch_index + 1}/{epochs_to_run}",
                leave=True,
                mininterval=1.0,
            )
            for batch in progress:
                valid_tokens = int(batch.pop("_valid_tokens"))
                padded_tokens = int(batch.pop("_padded_tokens"))
                batch_size = int(batch["labels"].shape[0])
                batch = move_batch(batch, device)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=AMP_ENABLED,
                ):
                    outputs = model(**batch)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                epoch_loss_sum += loss.detach().double() * batch_size
                epoch_samples += batch_size
                epoch_batches += 1
                total_batches += 1
                total_samples_processed += batch_size
                epoch_valid_tokens += valid_tokens
                epoch_padded_tokens += padded_tokens
                total_valid_tokens += valid_tokens
                total_padded_tokens += padded_tokens

            torch.cuda.synchronize()
            epoch_end = time.perf_counter()
            epoch_time = epoch_end - epoch_start
            average_loss = float((epoch_loss_sum / epoch_samples).item())
            epoch_throughput = epoch_samples / epoch_time
            epoch_valid_tokens_s = epoch_valid_tokens / epoch_time
            epoch_padded_tokens_s = epoch_padded_tokens / epoch_time
            epoch_avg_valid_length = epoch_valid_tokens / epoch_samples
            epoch_avg_padded_length = epoch_padded_tokens / epoch_samples
            epoch_padding_efficiency = (
                epoch_valid_tokens / epoch_padded_tokens * 100.0
            )

            row = {
                "epoch": epoch_index + 1,
                "epoch_time_s": epoch_time,
                "throughput_samples_s": epoch_throughput,
                "samples": epoch_samples,
                "batches": epoch_batches,
                "average_training_loss": average_loss,
                "valid_tokens": epoch_valid_tokens,
                "padded_tokens": epoch_padded_tokens,
                "valid_tokens_s": epoch_valid_tokens_s,
                "padded_tokens_s": epoch_padded_tokens_s,
                "average_valid_tokens_per_sample": epoch_avg_valid_length,
                "average_padded_sequence_length": epoch_avg_padded_length,
                "padding_efficiency_percent": epoch_padding_efficiency,
            }
            epoch_rows.append(row)
            print(
                f"Epoch {epoch_index + 1}/{epochs_to_run} | "
                f"Time: {epoch_time:.2f}s | "
                f"Throughput: {epoch_throughput:.2f} samples/s | "
                f"Batches: {epoch_batches} | "
                f"Avg padded length: {epoch_avg_padded_length:.2f} | "
                f"Loss: {average_loss:.6f}"
            )

        torch.cuda.synchronize()
        total_training_time = time.perf_counter() - total_start
        monitor.stop()

        telemetry_df = monitor.dataframe()
        telemetry_df.to_csv(run_dir / "telemetry.csv", index=False)
        with (run_dir / "nvml_api_errors.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(monitor.api_errors, file, indent=2)
        validate_telemetry(telemetry_df)

        epoch_df = pd.DataFrame(epoch_rows)
        epoch_df.to_csv(run_dir / "epoch_metrics.csv", index=False)

        validation = validate_model(model, validation_loader, device)

        throughput = total_samples_processed / total_training_time
        valid_tokens_per_second = total_valid_tokens / total_training_time
        padded_tokens_per_second = total_padded_tokens / total_training_time
        average_valid_tokens_per_sample = (
            total_valid_tokens / total_samples_processed
        )
        average_padded_sequence_length = (
            total_padded_tokens / total_samples_processed
        )
        padding_efficiency_percent = (
            total_valid_tokens / total_padded_tokens * 100.0
        )
        batch_latency_ms = total_training_time / total_batches * 1000.0
        average_epoch_time = statistics.mean(
            row["epoch_time_s"] for row in epoch_rows
        )

        average_power_w = series_stat(telemetry_df, "power_w", "mean")
        performance_per_watt = (
            throughput / average_power_w
            if not math.isnan(average_power_w) and average_power_w > 0
            else float("nan")
        )
        average_gpu_util = series_stat(
            telemetry_df, "gpu_util_percent", "mean"
        )
        average_vram_mb = series_stat(telemetry_df, "vram_used_mb", "mean")
        peak_vram_mb = series_stat(telemetry_df, "vram_used_mb", "max")
        torch_peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
        average_gpu_temp_c = series_stat(
            telemetry_df, "gpu_temperature_c", "mean"
        )
        peak_gpu_temp_c = series_stat(
            telemetry_df, "gpu_temperature_c", "max"
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

        temperature_limit_seen = bool(
            monitor.thermal_throttle_seen
            or (not math.isnan(peak_gpu_temp_c) and peak_gpu_temp_c >= 90.0)
        )

        expected_samples = train_samples_per_epoch * epochs_to_run
        expected_batches = len(train_loader) * epochs_to_run
        if total_samples_processed != expected_samples:
            raise RuntimeError(
                f"Processed {total_samples_processed} training samples; "
                f"expected {expected_samples}."
            )
        if total_batches != expected_batches:
            raise RuntimeError(
                f"Processed {total_batches} batches; expected {expected_batches}."
            )
        if len(epoch_rows) != epochs_to_run:
            raise RuntimeError(
                f"Completed {len(epoch_rows)} epochs; expected {epochs_to_run}."
            )
        if not all(
            math.isfinite(row["average_training_loss"])
            for row in epoch_rows
        ):
            raise RuntimeError("One or more epoch training losses are non-finite.")
        if not 0.0 <= validation["accuracy"] <= 1.0:
            raise RuntimeError("Validation accuracy is outside the valid range.")
        if validation["samples"] != validation_samples:
            raise RuntimeError(
                f"Validated {validation['samples']} samples; "
                f"expected {validation_samples}."
            )

        stability_notes = (
            "Completed successfully; no Python, CUDA, cuDNN, Transformers, "
            "Datasets, NVML, GPU, OOM, or thermal fatal errors detected."
        )
        if temperature_limit_seen:
            stability_notes += (
                " At least one monitored GPU temperature reached the configured "
                "critical threshold."
            )
        else:
            stability_notes += (
                " No monitored GPU temperature reached the configured critical "
                "thresholds."
            )
        if monitor.api_errors:
            error_count = sum(
                item["count"] for item in monitor.api_errors.values()
            )
            stability_notes += (
                f" NVML had {error_count} recoverable telemetry API errors."
            )
        else:
            stability_notes += " All required NVML telemetry queries succeeded."

        summary = {
            "mode": mode,
            "run": run_number,
            "throughput_samples_s": throughput,
            "valid_tokens_s": valid_tokens_per_second,
            "padded_tokens_s": padded_tokens_per_second,
            "average_valid_tokens_per_sample": average_valid_tokens_per_sample,
            "average_padded_sequence_length": average_padded_sequence_length,
            "padding_efficiency_percent": padding_efficiency_percent,
            "batch_latency_ms_batch": batch_latency_ms,
            "total_training_time_s": total_training_time,
            "average_epoch_time_s": average_epoch_time,
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
            "telemetry_samples": len(telemetry_df),
            "telemetry_interval_s": SAMPLE_INTERVAL_S,
            "workers": NUM_WORKERS,
            "amp_enabled": scaler.is_enabled(),
            "padding": "dynamic_per_batch",
            "model_parameter_dtype": str(next(model.parameters()).dtype),
            "stability_error_notes": stability_notes,
        }
        pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)

        with (run_dir / "summary.txt").open("w", encoding="utf-8") as file:
            file.write(f"DistilBERT SST-2 NVIDIA {run_label.title()}\n")
            file.write("=" * 72 + "\n")
            file.write(f"Throughput: {format_value(throughput)} samples/s\n")
            file.write(
                f"Valid Token Throughput: {format_value(valid_tokens_per_second)} tokens/s\n"
            )
            file.write(
                f"Padded Token Throughput: {format_value(padded_tokens_per_second)} tokens/s\n"
            )
            file.write(
                "Average Valid Tokens per Sample: "
                f"{format_value(average_valid_tokens_per_sample)}\n"
            )
            file.write(
                "Average Padded Sequence Length: "
                f"{format_value(average_padded_sequence_length)}\n"
            )
            file.write(
                "Padding Efficiency: "
                f"{format_value(padding_efficiency_percent)} %\n"
            )
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
                f"{format_value(performance_per_watt, 4)} samples/s/W\n"
            )
            file.write(
                f"Average GPU Utilization: {format_value(average_gpu_util)} %\n"
            )
            file.write(
                f"Average VRAM Usage: {format_value(average_vram_mb)} MB\n"
            )
            file.write(f"Peak VRAM Usage: {format_value(peak_vram_mb)} MB\n")
            file.write(
                "Average GPU Temperature: "
                f"{format_value(average_gpu_temp_c)} C\n"
            )
            file.write(
                "Peak GPU Temperature: "
                f"{format_value(peak_gpu_temp_c)} C\n"
            )
            file.write(
                f"Final Validation Accuracy: {validation['accuracy']:.4f}\n"
            )
            file.write(
                f"Final Validation Loss: {validation['loss']:.6f}\n"
            )
            file.write(
                "Final Training Loss: "
                f"{epoch_rows[-1]['average_training_loss']:.6f}\n"
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
            file.write(f"Precision: AMP ON; AMP enabled: {scaler.is_enabled()}\n")
            file.write(f"Workers: {NUM_WORKERS}\n")
            file.write(f"Stability / Error Notes: {stability_notes}\n")

        valid_marker = {
            "status": "VALID",
            "timestamp": datetime.now().astimezone().isoformat(),
            "mode": mode,
            "run": run_number,
            "epochs_completed": len(epoch_rows),
            "total_samples_processed": total_samples_processed,
            "total_batches_processed": total_batches,
            "total_valid_tokens": total_valid_tokens,
            "total_padded_tokens": total_padded_tokens,
            "workers": NUM_WORKERS,
            "amp_enabled": scaler.is_enabled(),
            "padding": "dynamic_per_batch",
            "validation_accuracy": validation["accuracy"],
            "driver_version": driver_version,
            "cuda_runtime": torch.version.cuda,
            "script_revision": SCRIPT_REVISION,
            "thermal_throttle_seen": monitor.thermal_throttle_seen,
        }
        with (run_dir / "RUN_VALID.json").open("w", encoding="utf-8") as file:
            json.dump(valid_marker, file, indent=2)

        print()
        print("=" * 84)
        if smoke:
            print("SMOKE TEST PASSED")
        elif one_epoch_test:
            print("FULL-DATASET ONE-EPOCH TEST VALID")
        else:
            print("OFFICIAL RUN VALID")
        print("=" * 84)
        print(f"Throughput: {throughput:.2f} samples/s")
        print(f"Valid Token Throughput: {valid_tokens_per_second:.2f} tokens/s")
        print(f"Padded Token Throughput: {padded_tokens_per_second:.2f} tokens/s")
        print(f"Average Padded Sequence Length: {average_padded_sequence_length:.2f}")
        print(f"Padding Efficiency: {padding_efficiency_percent:.2f} %")
        print(f"Batch Latency: {batch_latency_ms:.2f} ms/batch")
        print(f"Total Training Time: {total_training_time:.2f} s")
        print(f"Average GPU Power Draw: {average_power_w:.2f} W")
        print(f"Average GPU Utilization: {average_gpu_util:.2f} %")
        print(f"Average VRAM Usage: {average_vram_mb:.2f} MB")
        print(f"Peak VRAM Usage: {peak_vram_mb:.2f} MB")
        print(f"Average GPU Temperature: {average_gpu_temp_c:.2f} C")
        print(f"Final Validation Accuracy: {validation['accuracy']:.4f}")
        print(f"Workers: {NUM_WORKERS}")
        print(f"AMP enabled: {scaler.is_enabled()}")
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
                with (run_dir / "nvml_api_errors.json").open(
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


def run_all_runs() -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    status = {
        "status": "RUNNING",
        "timestamp_started": datetime.now().astimezone().isoformat(),
        "script": str(Path(__file__).resolve()),
        "runs": [],
    }
    ALL_RUNS_STATUS_JSON.write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    ALL_RUNS_STATUS_LOG.write_text("", encoding="utf-8")

    def status_line(text: str) -> None:
        print(text)
        with ALL_RUNS_STATUS_LOG.open("a", encoding="utf-8") as file:
            file.write(text + "\n")

    status_line("=" * 84)
    status_line("DISTILBERT NVIDIA ALL THREE REGULAR RUNS")
    status_line("=" * 84)

    for run_number in (1, 2, 3):
        started = datetime.now().astimezone().isoformat()
        status_line(f"Starting official regular run {run_number}...")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--run",
            str(run_number),
        ]
        result = subprocess.run(command, check=False)
        run_dir = LOGS_DIR / f"distilbertrun{run_number}"
        valid_path = run_dir / "RUN_VALID.json"

        valid = False
        marker = None
        if result.returncode == 0 and valid_path.exists():
            try:
                marker = json.loads(valid_path.read_text(encoding="utf-8"))
                valid = (
                    marker.get("status") == "VALID"
                    and marker.get("run") == run_number
                    and marker.get("epochs_completed") == EPOCHS
                    and marker.get("amp_enabled") is True
                    and marker.get("workers") == NUM_WORKERS
                    and marker.get("padding") == "dynamic_per_batch"
                )
            except Exception:
                valid = False

        status["runs"].append(
            {
                "run": run_number,
                "started": started,
                "finished": datetime.now().astimezone().isoformat(),
                "return_code": result.returncode,
                "valid": valid,
                "run_directory": str(run_dir),
                "marker": marker,
            }
        )
        ALL_RUNS_STATUS_JSON.write_text(
            json.dumps(status, indent=2), encoding="utf-8"
        )

        if not valid:
            status["status"] = "FAILED"
            status["failed_run"] = run_number
            status["timestamp_finished"] = datetime.now().astimezone().isoformat()
            ALL_RUNS_STATUS_JSON.write_text(
                json.dumps(status, indent=2), encoding="utf-8"
            )
            status_line(
                f"Run {run_number} failed validation. Stopping all-runs workflow."
            )
            raise SystemExit(1)

        status_line(f"Run {run_number} completed and validated.")
        if run_number != 3:
            status_line("Waiting 10 seconds before the next run...")
            time.sleep(10)

    status["status"] = "VALID"
    status["timestamp_finished"] = datetime.now().astimezone().isoformat()
    ALL_RUNS_STATUS_JSON.write_text(
        json.dumps(status, indent=2), encoding="utf-8"
    )
    status_line("ALL THREE DISTILBERT REGULAR RUNS VALID")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="DistilBERT GLUE/SST-2 NVIDIA regular benchmark"
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--force-prepare", action="store_true")
    mode.add_argument("--smoke", action="store_true")
    mode.add_argument("--one-epoch-test", action="store_true")
    mode.add_argument("--run", type=int, choices=(1, 2, 3))
    mode.add_argument("--all-runs", action="store_true")
    args = parser.parse_args()

    if args.prepare or args.force_prepare:
        prepare_assets(force=bool(args.force_prepare))
        return
    if args.all_runs:
        run_all_runs()
        return
    if args.smoke:
        run_single("smoke")
        return
    if args.one_epoch_test:
        run_single("one_epoch_test")
        return
    run_single("official_regular", run_number=args.run)


if __name__ == "__main__":
    main()

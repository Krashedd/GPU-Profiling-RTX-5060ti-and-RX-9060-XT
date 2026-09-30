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
from torch.profiler import ProfilerActivity, profile, schedule
from transformers import AutoModelForSequenceClassification


PROJECT_DIR = Path.home() / "Thesis-Testing"
SCRIPT_DIR = PROJECT_DIR / "scripts"
REGULAR_SCRIPT = SCRIPT_DIR / "distilbert_regular_amd.py"
PROFILE_SCRIPT = SCRIPT_DIR / "distilbert_profiled_amd.py"
PROFILE_DIR = PROJECT_DIR / "Logs" / "distilbertprofiledrun"
OPERATOR_TEST_DIR = PROJECT_DIR / "Logs" / "distilbertprofiletest1epoch"
GPU_ERROR_HISTORY = PROJECT_DIR / "Logs" / "distilbert_gpu_error_history.log"
ROCPROFV3 = Path("/opt/rocm/bin/rocprofv3")
TRAINING_RANGE_NAME = "DISTILBERT_SST2_TRAINING"
# The operator profiler records one deterministic representative window.
# The full training workload still runs so telemetry, final accuracy, and the
# timed profiled-pass metrics cover the complete run. A separate four-batch
# rocprofv3 pass provides the Perfetto timeline.
PYTORCH_PROFILE_WAIT_STEPS_OFFICIAL = 10
PYTORCH_PROFILE_WARMUP_STEPS_OFFICIAL = 10
PYTORCH_PROFILE_ACTIVE_STEPS_OFFICIAL = 100
PYTORCH_PROFILE_WAIT_STEPS_SMOKE = 0
PYTORCH_PROFILE_WARMUP_STEPS_SMOKE = 0
PYTORCH_PROFILE_ACTIVE_STEPS_SMOKE = 4
ROCPROF_CSV_CHUNK_ROWS = 250_000
TIMELINE_BATCHES = 4

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


def load_regular_module():
    if not REGULAR_SCRIPT.is_file():
        raise RuntimeError(f"Required regular script is missing: {REGULAR_SCRIPT}")

    spec = importlib.util.spec_from_file_location(
        "distilbert_regular_amd_shared", REGULAR_SCRIPT
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
            "The profiling script and regular script no longer use the same "
            "fixed workload configuration:\n" + "\n".join(mismatches)
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
        raise RuntimeError(
            "Required DistilBERT assets are missing:\n" + "\n".join(missing)
        )

    return module


def append_gpu_error_history(label: str, traceback_text: str) -> None:
    GPU_ERROR_HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with GPU_ERROR_HISTORY.open("a", encoding="utf-8") as file:
        file.write("\n" + "=" * 88 + "\n")
        file.write(f"Timestamp: {datetime.now().astimezone().isoformat()}\n")
        file.write(f"Workload: DistilBERT SST-2 AMD {label}\n")
        file.write(traceback_text.rstrip() + "\n")


def is_gpu_related_error(text: str) -> bool:
    terms = (
        "amd-smi", "amdsmi", "amdgpu", "cuda", "device-side", "gfx1200",
        "gpu", "hip", "hsa", "memory access fault", "miopen",
        "out of memory", "rocblas", "rocm", "rocprof", "roctx",
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
    """Minimal ROCTx range wrapper used to mark the timed training window."""

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
                failures.append(f"{library_path}: missing {', '.join(missing)}")
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
    # The thesis operator table contains executable ATen operators only.
    # Optimizer and autograd entries are higher-level profiler scopes.
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
        "Optimizer.step#Adam.step": "Adam optimizer update",
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


class RepresentativeOperatorAccumulator:
    """Aggregate ATen statistics from one representative profiler window."""

    def __init__(
        self,
        output_dir: Path,
        wait_steps: int,
        warmup_steps: int,
        active_steps: int,
    ):
        self.output_dir = output_dir
        self.wait_steps = int(wait_steps)
        self.warmup_steps = int(warmup_steps)
        self.active_steps = int(active_steps)
        self.aggregate = {}
        self.progress_rows = []
        self.windows_completed = 0
        self.total_profiled_batches = 0
        self.total_training_batches_seen = 0

    def consume(self, profiler_object, training_batches_seen: int) -> None:
        if self.windows_completed >= 1:
            raise RuntimeError("More than one representative profiler window completed.")

        operator_rows = 0
        for event in profiler_object.key_averages():
            name = str(event.key)
            if not is_pytorch_operator(name):
                continue
            self_us, total_us = event_device_times(event)
            if self_us <= 0 and total_us <= 0:
                continue

            row = self.aggregate.setdefault(
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
            operator_rows += 1

        self.windows_completed += 1
        self.total_profiled_batches += self.active_steps
        self.total_training_batches_seen = int(training_batches_seen)
        self.progress_rows.append(
            {
                "window": self.windows_completed,
                "wait_batches": self.wait_steps,
                "warmup_batches": self.warmup_steps,
                "profiled_batches": self.active_steps,
                "training_batches_seen_at_window_end": self.total_training_batches_seen,
                "operators_in_window": operator_rows,
                "unique_operators": len(self.aggregate),
            }
        )
        pd.DataFrame(self.progress_rows).to_csv(
            self.output_dir / "pytorch_operator_window_progress.csv", index=False
        )

        print(
            "torch.profiler representative window processed; "
            f"wait={self.wait_steps}, warmup={self.warmup_steps}, "
            f"active={self.active_steps} batches."
        )

    def finalize(self, total_training_batches: int) -> pd.DataFrame:
        rows = list(self.aggregate.values())
        rows.sort(key=lambda row: row["self_gpu_time_ms"], reverse=True)
        total_self_ms = sum(row["self_gpu_time_ms"] for row in rows)

        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
            row["self_gpu_percent"] = (
                row["self_gpu_time_ms"] / total_self_ms * 100.0
                if total_self_ms > 0 else float("nan")
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
        frame.to_csv(self.output_dir / "pytorch_all_operators.csv", index=False)
        frame.head(10).to_csv(
            self.output_dir / "pytorch_top10_operators.csv", index=False
        )

        with (self.output_dir / "pytorch_profiler_full_table.txt").open(
            "w", encoding="utf-8"
        ) as file:
            file.write(
                "Representative-window PyTorch ATen operator table\n"
                f"Wait batches: {self.wait_steps}\n"
                f"Warmup batches: {self.warmup_steps}\n"
                f"Active profiled batches: {self.active_steps}\n"
                f"Full training batches executed: {total_training_batches}\n\n"
            )
            file.write(frame.to_string(index=False))
            file.write("\n")

        metadata = {
            "collection_mode": "representative_window_during_full_training",
            "wait_batches": self.wait_steps,
            "warmup_batches": self.warmup_steps,
            "active_profiled_batches": self.active_steps,
            "windows_completed": self.windows_completed,
            "total_profiled_batches": self.total_profiled_batches,
            "total_training_batches_executed": int(total_training_batches),
            "operator_scope": "ATen operators only",
            "chrome_trace_exported": False,
            "timeline_source": "separate four-batch rocprofv3 Perfetto pass",
            "profiler_object_lifecycle": "one long-lived profiler object",
        }
        with (self.output_dir / "pytorch_operator_window_metadata.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(metadata, file, indent=2)
        return frame


class RepresentativeWindowProfiler:
    """Profile one deterministic window while the complete training run executes."""

    def __init__(
        self,
        output_dir: Path,
        wait_steps: int,
        warmup_steps: int,
        active_steps: int,
    ):
        self.output_dir = output_dir
        self.wait_steps = int(wait_steps)
        self.warmup_steps = int(warmup_steps)
        self.active_steps = int(active_steps)
        self.accumulator = RepresentativeOperatorAccumulator(
            output_dir,
            self.wait_steps,
            self.warmup_steps,
            self.active_steps,
        )
        self.total_batches_seen = 0
        self.started = False
        self.profiler = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(
                wait=self.wait_steps,
                warmup=self.warmup_steps,
                active=self.active_steps,
                repeat=1,
            ),
            on_trace_ready=self._trace_ready,
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
            with_flops=False,
            acc_events=False,
        )

    def _trace_ready(self, profiler_object) -> None:
        self.accumulator.consume(profiler_object, self.total_batches_seen)

    def before_batch(self) -> None:
        if not self.started:
            self.profiler.start()
            self.started = True

    def after_batch(self) -> None:
        if not self.started:
            raise RuntimeError("Profiler was not started before the batch.")
        self.total_batches_seen += 1
        self.profiler.step()

    def flush(self) -> None:
        if not self.started:
            return
        self.profiler.stop()
        self.started = False
        if self.accumulator.windows_completed != 1:
            raise RuntimeError(
                "Representative profiler window did not complete exactly once; "
                f"completed={self.accumulator.windows_completed}."
            )

    def finalize_outputs(self) -> pd.DataFrame:
        return self.accumulator.finalize(self.total_batches_seen)


def run_worker(
    profile_kind: str,
    output_dir: Path,
    smoke: bool,
    epochs_override: int | None = None,
) -> None:
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
    chunk_profiler = None

    try:
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("WANDB_DISABLED", "true")
        regular.set_seed(regular.SEED)

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

        dataset = regular.verify_assets()
        train_loader, validation_loader = regular.build_dataloaders(
            dataset, smoke=smoke
        )
        epochs_to_run = (
            1 if smoke else (int(epochs_override) if epochs_override else regular.EPOCHS)
        )
        train_samples_per_epoch = len(train_loader.dataset)
        validation_samples = len(validation_loader.dataset)

        print("=" * 88)
        print(
            f"DistilBERT SST-2 AMD {profile_kind} profiling "
            f"{'smoke test' if smoke else 'official pass'}"
        )
        print("=" * 88)
        print(f"Timestamp: {datetime.now().astimezone().isoformat()}")
        print(f"Output directory: {output_dir}")
        print(f"Model directory: {regular.MODEL_DIR}")
        print(f"Dataset directory: {regular.TOKENIZED_DATASET_DIR}")
        print(f"Python: {platform.python_version()}")
        print(f"Platform: {platform.platform()}")
        print(f"PyTorch: {torch.__version__}")
        print(f"HIP runtime: {torch.version.hip}")
        print(f"System ROCm: {regular.read_rocm_version()}")
        print(f"Transformers: {regular.package_version('transformers')}")
        print(f"Datasets: {regular.package_version('datasets')}")
        print(f"GPU: {gpu_name}")
        print(f"GPU architecture: {gpu_arch}")
        print()

        if profile_kind == "torch_operator":
            if smoke:
                profile_wait_steps = PYTORCH_PROFILE_WAIT_STEPS_SMOKE
                profile_warmup_steps = PYTORCH_PROFILE_WARMUP_STEPS_SMOKE
                profile_active_steps = PYTORCH_PROFILE_ACTIVE_STEPS_SMOKE
            else:
                profile_wait_steps = PYTORCH_PROFILE_WAIT_STEPS_OFFICIAL
                profile_warmup_steps = PYTORCH_PROFILE_WARMUP_STEPS_OFFICIAL
                profile_active_steps = PYTORCH_PROFILE_ACTIVE_STEPS_OFFICIAL
            chunk_profiler = RepresentativeWindowProfiler(
                output_dir,
                profile_wait_steps,
                profile_warmup_steps,
                profile_active_steps,
            )
            print(
                "torch.profiler will record one representative operator window: "
                f"wait={profile_wait_steps}, warmup={profile_warmup_steps}, "
                f"active={profile_active_steps} batches. The complete training "
                "workload still executes."
            )
        elif profile_kind in ("rocprofv3", "timeline"):
            roctx = RoctxController()
            print(f"ROCTx library: {roctx.library_path}")
        else:
            raise RuntimeError(f"Unknown profile kind: {profile_kind}")

        config = {
            "mode": "smoke" if smoke else "official_profiled",
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
            "pad_to_multiple_of": None,
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
            "hip_runtime": torch.version.hip,
            "system_rocm": regular.read_rocm_version(),
            "transformers_version": regular.package_version("transformers"),
            "datasets_version": regular.package_version("datasets"),
            "gpu": gpu_name,
            "gpu_architecture": gpu_arch,
            "training_range_name": TRAINING_RANGE_NAME,
            "pytorch_profile_collection": "representative_window_during_full_training",
            "pytorch_profile_wait_steps": (
                PYTORCH_PROFILE_WAIT_STEPS_SMOKE
                if smoke else PYTORCH_PROFILE_WAIT_STEPS_OFFICIAL
            ),
            "pytorch_profile_warmup_steps": (
                PYTORCH_PROFILE_WARMUP_STEPS_SMOKE
                if smoke else PYTORCH_PROFILE_WARMUP_STEPS_OFFICIAL
            ),
            "pytorch_profile_active_steps": (
                PYTORCH_PROFILE_ACTIVE_STEPS_SMOKE
                if smoke else PYTORCH_PROFILE_ACTIVE_STEPS_OFFICIAL
            ),
            "timeline_batches": TIMELINE_BATCHES if profile_kind == "timeline" else None,
            "script_revision": "2026-06-17 distilbert-profile-v3 representative-window",
        }
        with (output_dir / "config.json").open("w", encoding="utf-8") as file:
            json.dump(config, file, indent=2)

        device = torch.device("cuda:0")
        regular.set_seed(regular.SEED)
        model = AutoModelForSequenceClassification.from_pretrained(
            regular.MODEL_DIR,
            local_files_only=True,
        ).to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=regular.LEARNING_RATE,
        )
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
        if roctx is not None:
            roctx.push(TRAINING_RANGE_NAME)
            print(f"ROCTx training range started: {TRAINING_RANGE_NAME}")
        total_start = time.perf_counter()

        epoch_rows = []
        total_batches = 0
        total_samples_processed = 0
        total_valid_tokens = 0
        total_padded_tokens = 0
        stop_after_batches = TIMELINE_BATCHES if profile_kind == "timeline" else None
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
                if chunk_profiler is not None:
                    chunk_profiler.before_batch()

                valid_tokens = int(batch.pop("_valid_tokens"))
                padded_tokens = int(batch.pop("_padded_tokens"))
                batch_size = int(batch["labels"].shape[0])
                batch = regular.move_batch(batch, device)

                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=regular.AMP_ENABLED,
                ):
                    outputs = model(**batch)
                    loss = outputs.loss

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                if chunk_profiler is not None:
                    chunk_profiler.after_batch()

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
            epoch_rows.append(
                {
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
                }
            )
            print(
                f"Epoch {epoch_index + 1}/{epochs_to_run} | "
                f"Time: {epoch_time:.2f}s | "
                f"Throughput: {epoch_samples / epoch_time:.2f} samples/s | "
                f"Batches: {epoch_batches} | "
                f"Loss: {average_loss:.6f}"
            )

            if stop_training:
                break

        torch.cuda.synchronize()
        if chunk_profiler is not None:
            # Profiler shutdown and operator aggregation are part of the
            # instrumented pass wall-clock time. Final CSV formatting is outside it.
            chunk_profiler.flush()

        total_training_time = time.perf_counter() - total_start
        if roctx is not None:
            roctx.pop()
            print(f"ROCTx training range ended: {TRAINING_RANGE_NAME}")
        monitor.stop()

        if chunk_profiler is not None:
            operator_frame = chunk_profiler.finalize_outputs()
            if operator_frame.empty:
                raise RuntimeError("PyTorch profiler produced no GPU operator rows.")

        telemetry_df = monitor.dataframe()
        telemetry_df.to_csv(output_dir / "telemetry.csv", index=False)
        with (output_dir / "amdsmi_api_errors.json").open(
            "w", encoding="utf-8"
        ) as file:
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
        average_epoch_time = statistics.mean(row["epoch_time_s"] for row in epoch_rows)

        average_power_w = regular.series_stat(telemetry_df, "power_w", "mean")
        performance_per_watt = (
            throughput / average_power_w
            if not math.isnan(average_power_w) and average_power_w > 0
            else float("nan")
        )
        average_gpu_util = regular.series_stat(
            telemetry_df, "gpu_util_percent", "mean"
        )
        average_vram_mb = regular.series_stat(telemetry_df, "vram_used_mb", "mean")
        peak_vram_mb = regular.series_stat(telemetry_df, "vram_used_mb", "max")
        torch_peak_allocated_mb = torch.cuda.max_memory_allocated() / (1024**2)
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

        expected_samples = (
            total_samples_processed
            if profile_kind == "timeline"
            else train_samples_per_epoch * epochs_to_run
        )
        expected_batches = (
            TIMELINE_BATCHES
            if profile_kind == "timeline"
            else len(train_loader) * epochs_to_run
        )
        if total_samples_processed != expected_samples:
            raise RuntimeError(
                f"Processed {total_samples_processed} samples; expected {expected_samples}."
            )
        if total_batches != expected_batches:
            raise RuntimeError(
                f"Processed {total_batches} batches; expected {expected_batches}."
            )
        if not all(math.isfinite(row["average_training_loss"]) for row in epoch_rows):
            raise RuntimeError("One or more epoch losses are non-finite.")
        if not 0.0 <= validation["accuracy"] <= 1.0:
            raise RuntimeError("Validation accuracy is outside the valid range.")
        if validation["samples"] != validation_samples:
            raise RuntimeError(
                f"Validated {validation['samples']} samples; expected {validation_samples}."
            )

        temperature_limit_seen = any(
            not math.isnan(value) and value >= threshold
            for value, threshold in (
                (peak_edge_temp_c, 100.0),
                (peak_hotspot_temp_c, 110.0),
                (peak_memory_temp_c, 105.0),
            )
        )
        stability_notes = (
            "Completed successfully; no Python, HIP, ROCm, Transformers, "
            "Datasets, AMD SMI, GPU, OOM, profiler, or thermal fatal errors detected."
        )
        if temperature_limit_seen:
            stability_notes += " A monitored GPU temperature reached a critical threshold."
        else:
            stability_notes += " No monitored GPU temperature reached the configured critical thresholds."
        if monitor.api_errors:
            error_count = sum(item["count"] for item in monitor.api_errors.values())
            stability_notes += f" AMD SMI had {error_count} recoverable API errors."
        else:
            stability_notes += " All AMD SMI telemetry queries succeeded."

        summary = {
            "mode": "smoke" if smoke else "official_profiled",
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
            "average_gpu_power_w": average_power_w,
            "performance_per_watt_samples_s_w": performance_per_watt,
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
            "telemetry_interval_s": regular.SAMPLE_INTERVAL_S,
            "workers": regular.NUM_WORKERS,
            "amp_enabled": scaler.is_enabled(),
            "padding": "dynamic_per_batch",
            "model_parameter_dtype": str(next(model.parameters()).dtype),
            "stability_error_notes": stability_notes,
        }
        pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False)

        with (output_dir / "summary.txt").open("w", encoding="utf-8") as file:
            file.write(f"DistilBERT SST-2 AMD {profile_kind} profile\n")
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
            file.write(
                "Average GPU Edge Temperature: "
                f"{regular.format_value(average_edge_temp_c)} C\n"
            )
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
            "mode": "smoke" if smoke else "official_profiled",
            "profile_kind": profile_kind,
            "epochs_completed": len(epoch_rows),
            "total_samples_processed": total_samples_processed,
            "total_batches_processed": total_batches,
            "workers": regular.NUM_WORKERS,
            "amp_enabled": scaler.is_enabled(),
            "padding": "dynamic_per_batch",
            "validation_accuracy": validation["accuracy"],
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
        if chunk_profiler is not None:
            try:
                chunk_profiler.flush()
            except Exception:
                pass
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
                partial_df = monitor.dataframe()
                if not partial_df.empty:
                    partial_df.to_csv(output_dir / "telemetry_partial.csv", index=False)
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
        marker_matches["End_Timestamp"] - marker_matches["Start_Timestamp"]
    )
    marker_row = marker_matches.sort_values("duration_ns", ascending=False).iloc[0]
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

    unsorted_intervals = destination / ".kernel_intervals_unsorted.tsv"
    sorted_intervals = destination / ".kernel_intervals_sorted.tsv"
    for temporary_path in (unsorted_intervals, sorted_intervals):
        temporary_path.unlink(missing_ok=True)

    required_kernel_columns = {"Kernel_Name", "Start_Timestamp", "End_Timestamp"}

    try:
        with unsorted_intervals.open(
            "w", encoding="utf-8", buffering=1024 * 1024
        ) as interval_file:
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
                work["duration_ns"] = work["End_Timestamp"] - work["Start_Timestamp"]
                work = work[work["duration_ns"] >= 0]
                if work.empty:
                    continue

                training_kernel_rows += int(len(work))
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
                sq_grouped = work.groupby("Kernel_Name", dropna=False)[
                    "duration_sq"
                ].sum()

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
            if total_kernel_duration_ns > 0 else float("nan")
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
    grouped_frame.to_csv(destination / "rocprofv3_parsed_summary.csv", index=False)
    grouped_frame.head(5).to_csv(destination / "amd_top5_kernels.csv", index=False)

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
        work["duration_ns"] = work["End_Timestamp"] - work["Start_Timestamp"]
        work = work[work["duration_ns"] >= 0]
        memory_copy_calls += int(len(work))
        memory_copy_total_ns += float(work["duration_ns"].sum())

    memory_copy_total_ms = (
        memory_copy_total_ns / 1e6 if memory_copy_calls else float("nan")
    )
    memory_copy_average_us = (
        memory_copy_total_ns / memory_copy_calls / 1e3
        if memory_copy_calls else float("nan")
    )
    pd.DataFrame(
        [{
            "calls": memory_copy_calls,
            "total_duration_ms": memory_copy_total_ms,
            "average_duration_us": memory_copy_average_us,
        }]
    ).to_csv(destination / "rocprofv3_memory_copy_summary.csv", index=False)

    gpu_idle_ns = max(0.0, training_duration_ns - busy_union_ns)
    gpu_idle_percent = (
        gpu_idle_ns / training_duration_ns * 100.0
        if training_duration_ns > 0 else float("nan")
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
        ("Throughput", float(primary["throughput_samples_s"]), "samples/s", "Full training run with representative 100-batch operator window"),
        ("Valid Token Throughput", float(primary["valid_tokens_s"]), "tokens/s", "Full training run with representative 100-batch operator window"),
        ("Padded Token Throughput", float(primary["padded_tokens_s"]), "tokens/s", "Full training run with representative 100-batch operator window"),
        ("Average Padded Sequence Length", float(primary["average_padded_sequence_length"]), "tokens", "Full training run with representative 100-batch operator window"),
        ("Padding Efficiency", float(primary["padding_efficiency_percent"]), "%", "Full training run with representative 100-batch operator window"),
        ("Batch Latency", float(primary["batch_latency_ms_batch"]), "ms/batch", "Full training run with representative 100-batch operator window"),
        ("Total Training Time", float(primary["total_training_time_s"]), "s", "Full training run with representative 100-batch operator window"),
        ("Average Epoch Time", float(primary["average_epoch_time_s"]), "s", "Full training run with representative 100-batch operator window"),
        ("Average GPU Power Draw", float(primary["average_gpu_power_w"]), "W", "Full training run with representative 100-batch operator window"),
        ("Performance per Watt", float(primary["performance_per_watt_samples_s_w"]), "samples/s/W", "Full training run with representative 100-batch operator window"),
        ("Average GPU Utilization", util, "%", "Full training run with representative 100-batch operator window"),
        ("Average VRAM Usage", float(primary["average_vram_usage_mb"]), "MB", "Full training run with representative 100-batch operator window"),
        ("Peak VRAM Usage", float(primary["peak_vram_usage_mb"]), "MB", "Full training run with representative 100-batch operator window"),
        ("Average GPU Temperature", float(primary["average_gpu_edge_temperature_c"]), "C", "Full training run with representative 100-batch operator window"),
        ("Top PyTorch Operator", top_operator, "", "Representative 100-batch ATen operator window"),
        ("Top AMD Kernel", top_kernel, "", "rocprofv3 full run"),
        ("Top Kernel Duration", top_kernel_duration_ms, "ms cumulative", "rocprofv3 full run"),
        ("Kernel Launch Count", int(parse_metadata["training_kernel_rows"]), "dispatches", "rocprofv3 full run"),
        ("GPU Idle Time", float(parse_metadata["gpu_idle_time_ms"]) / 1000.0, "s", "Exact union of full-run kernel intervals"),
        ("GPU Idle Percentage", idle_percent, "%", "Exact union of full-run kernel intervals"),
        ("Bottleneck Type", bottleneck, "", "Heuristic; interpret with operators/kernels"),
        ("Final Validation Score", float(primary["final_validation_accuracy"]), "accuracy", "Full training run with representative 100-batch operator window"),
        ("Average GPU Memory Copy Time", float(parse_metadata["memory_copy_average_us"]), "us/copy", "rocprofv3 full run"),
        ("Peak System RAM", float(primary["peak_system_ram_mb"]), "MB", "Full training run with representative 100-batch operator window"),
        ("Average System RAM", float(primary["average_system_ram_mb"]), "MB", "Full training run with representative 100-batch operator window"),
        ("CPU Utilization", float(primary["average_cpu_util_percent"]), "%", "Full training run with representative 100-batch operator window"),
        ("Average Disk Read", float(primary["average_disk_read_mb_s"]), "MB/s", "Full training run with representative 100-batch operator window"),
        ("Average Disk Write", float(primary["average_disk_write_mb_s"]), "MB/s", "Full training run with representative 100-batch operator window"),
        ("Peak Disk Read", float(primary["peak_disk_read_mb_s"]), "MB/s", "Full training run with representative 100-batch operator window"),
        ("Peak Disk Write", float(primary["peak_disk_write_mb_s"]), "MB/s", "Full training run with representative 100-batch operator window"),
    ]
    frame = pd.DataFrame(rows, columns=["metric", "value", "unit", "source"])
    frame.to_csv(output_dir / "profile_metrics_summary.csv", index=False)

    with (output_dir / "profile_metrics_summary.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write("DistilBERT AMD Consolidated Profile Metrics\n")
        file.write("=" * 72 + "\n")
        for metric, value, unit, source in rows:
            suffix = f" {unit}" if unit else ""
            file.write(f"{metric}: {value}{suffix} [{source}]\n")


def validate_worker(
    worker_dir: Path,
    kind: str,
    smoke: bool,
    timeline: bool = False,
    expected_samples_override: int | None = None,
    expected_batches_override: int | None = None,
) -> None:
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

    expected_samples = 256 if smoke else 673_490
    expected_batches = 4 if (smoke or timeline) else 10_530
    if timeline:
        expected_samples = 256
    if expected_samples_override is not None:
        expected_samples = int(expected_samples_override)
    if expected_batches_override is not None:
        expected_batches = int(expected_batches_override)

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
    if not bool(summary["amp_enabled"]):
        raise RuntimeError(f"{kind} worker did not enable AMP.")
    if int(summary["workers"]) != 4:
        raise RuntimeError(f"{kind} worker did not use four workers.")
    if str(summary["padding"]) != "dynamic_per_batch":
        raise RuntimeError(f"{kind} worker did not use dynamic padding.")
    if not math.isfinite(float(summary["final_validation_accuracy"])):
        raise RuntimeError(f"{kind} worker produced an invalid accuracy.")


def run_operator_test() -> None:
    """Run one full epoch with one representative PyTorch operator window."""
    output_dir = OPERATOR_TEST_DIR
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
        worker_dir = output_dir / "pytorch_operator_pass"
        expected_training_batches = 1_053
        expected_samples = 67_349
        wait_steps = PYTORCH_PROFILE_WAIT_STEPS_OFFICIAL
        warmup_steps = PYTORCH_PROFILE_WARMUP_STEPS_OFFICIAL
        active_steps = PYTORCH_PROFILE_ACTIVE_STEPS_OFFICIAL

        print("=" * 92)
        print("DISTILBERT AMD ONE-EPOCH REPRESENTATIVE OPERATOR TEST")
        print("=" * 92)
        print(f"Output directory: {output_dir}")
        print(f"Wait batches: {wait_steps}")
        print(f"Warmup batches: {warmup_steps}")
        print(f"Active profiled batches: {active_steps}")
        print(f"Full training batches executed: {expected_training_batches}")
        print(
            "Purpose: verify useful operator coverage with realistic profiler "
            "overhead before the official three-pass workflow."
        )
        print()

        command = [
            sys.executable,
            str(PROFILE_SCRIPT),
            "--internal-worker",
            "torch_operator",
            "--internal-output",
            str(worker_dir),
            "--internal-epochs",
            "1",
        ]
        return_code = stream_subprocess(command, env=os.environ.copy())
        if return_code != 0:
            raise RuntimeError(
                f"One-epoch PyTorch operator worker exited with code {return_code}."
            )

        validate_worker(
            worker_dir,
            "one-epoch PyTorch operator",
            smoke=False,
            expected_samples_override=expected_samples,
            expected_batches_override=expected_training_batches,
        )

        files = (
            "config.json",
            "epoch_metrics.csv",
            "telemetry.csv",
            "amdsmi_api_errors.json",
            "summary.csv",
            "summary.txt",
            "pytorch_all_operators.csv",
            "pytorch_top10_operators.csv",
            "pytorch_profiler_full_table.txt",
            "pytorch_operator_window_progress.csv",
            "pytorch_operator_window_metadata.json",
        )
        for name in files:
            link_or_copy(worker_dir / name, output_dir / name)
        link_or_copy(
            worker_dir / "training_output.log",
            output_dir / "pytorch_operator_training_output.log",
        )

        metadata = json.loads(
            (output_dir / "pytorch_operator_window_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        if int(metadata.get("windows_completed", 0)) != 1:
            raise RuntimeError(
                f"Expected one profiler window, found {metadata.get('windows_completed')}."
            )
        if int(metadata.get("total_profiled_batches", 0)) != active_steps:
            raise RuntimeError(
                f"Expected {active_steps} profiled batches, found "
                f"{metadata.get('total_profiled_batches')}."
            )
        if int(metadata.get("total_training_batches_executed", 0)) != expected_training_batches:
            raise RuntimeError(
                f"Expected {expected_training_batches} executed batches, found "
                f"{metadata.get('total_training_batches_executed')}."
            )

        summary = pd.read_csv(output_dir / "summary.csv").iloc[0]
        test_valid = {
            "status": "VALID",
            "timestamp": datetime.now().astimezone().isoformat(),
            "mode": "one_epoch_representative_operator_test",
            "epochs": 1,
            "training_batches_executed": expected_training_batches,
            "profiled_batches": active_steps,
            "profile_windows": 1,
            "wait_batches": wait_steps,
            "warmup_batches": warmup_steps,
            "throughput_samples_s": float(summary["throughput_samples_s"]),
            "epoch_time_s": float(summary["average_epoch_time_s"]),
            "peak_system_ram_mb": float(summary["peak_system_ram_mb"]),
            "top_operator": str(
                pd.read_csv(output_dir / "pytorch_top10_operators.csv").iloc[0]["operator"]
            ),
        }
        with (output_dir / "RUN_VALID.json").open("w", encoding="utf-8") as file:
            json.dump(test_valid, file, indent=2)

        print("=" * 92)
        print("ONE-EPOCH REPRESENTATIVE OPERATOR TEST VALID")
        print("=" * 92)
        print(f"Throughput: {test_valid['throughput_samples_s']:.2f} samples/s")
        print(f"Epoch time: {test_valid['epoch_time_s']:.2f} s")
        print(f"Profiled batches: {test_valid['profiled_batches']}")
        print(f"Peak system RAM: {test_valid['peak_system_ram_mb']:.2f} MB")
        print(f"Results saved to: {output_dir}")

    except Exception:
        traceback_text = traceback.format_exc()
        print("\nONE-EPOCH REPRESENTATIVE OPERATOR TEST FAILED")
        print(traceback_text)
        if is_gpu_related_error(traceback_text):
            append_gpu_error_history("one-epoch representative operator test", traceback_text)
        raise
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


def run_parent(smoke: bool) -> None:
    output_dir = (
        Path("/tmp/distilbert_amd_profile_smoke") if smoke else PROFILE_DIR
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
        regular = load_regular_module()
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

        expected_training_batches = 4 if smoke else 10_530
        profile_wait_steps = (
            PYTORCH_PROFILE_WAIT_STEPS_SMOKE
            if smoke else PYTORCH_PROFILE_WAIT_STEPS_OFFICIAL
        )
        profile_warmup_steps = (
            PYTORCH_PROFILE_WARMUP_STEPS_SMOKE
            if smoke else PYTORCH_PROFILE_WARMUP_STEPS_OFFICIAL
        )
        expected_profiled_batches = (
            PYTORCH_PROFILE_ACTIVE_STEPS_SMOKE
            if smoke else PYTORCH_PROFILE_ACTIVE_STEPS_OFFICIAL
        )
        expected_profile_windows = 1

        parent_config = {
            "mode": "smoke" if smoke else "official_profiled",
            "workflow": "three_pass_single_command",
            "primary_profile_metrics_source": "pytorch_operator_pass",
            "operator_profiler": (
                "torch.profiler one representative operator window during a "
                "complete training run"
            ),
            "kernel_profiler": (
                "rocprofv3 kernel/memory-copy/marker traces with ROCTx "
                "training-window filtering"
            ),
            "representative_timeline": (
                "fixed four-batch rocprofv3 Perfetto trace for visualization only"
            ),
            "reason_for_separate_passes": (
                "PyTorch operator profiling and rocprofv3 tracing are separated "
                "to avoid profiler interference. The short third pass retains a "
                "readable Perfetto timeline without using it for quantitative metrics."
            ),
            "both_quantitative_passes_execute_identical_fixed_workload": True,
            "operator_window_used_for_full_run_cumulative_counts": False,
            "timeline_used_for_quantitative_metrics": False,
            "training_range_name": TRAINING_RANGE_NAME,
            "pytorch_wait_steps": profile_wait_steps,
            "pytorch_warmup_steps": profile_warmup_steps,
            "pytorch_active_profiled_steps": expected_profiled_batches,
            "expected_pytorch_profile_windows": expected_profile_windows,
            "full_training_batches_executed": expected_training_batches,
            "batch_size": regular.BATCH_SIZE,
            "epochs": 1 if smoke else regular.EPOCHS,
            "max_length": regular.MAX_LENGTH,
            "padding": "dynamic_per_batch",
            "amp_enabled": regular.AMP_ENABLED,
            "workers": regular.NUM_WORKERS,
            "timestamp": datetime.now().astimezone().isoformat(),
        }
        with (output_dir / "profile_workflow_config.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(parent_config, file, indent=2)

        worker_smoke_arg = ["--internal-smoke"] if smoke else []

        print("=" * 92)
        print(
            "DistilBERT AMD profiled workflow "
            f"({'smoke test' if smoke else 'official'})"
        )
        print("=" * 92)
        print(f"Output directory: {output_dir}")
        print(
            "This command performs three controlled passes: representative-window "
            "PyTorch operator profiling during a full training run, full rocprofv3 "
            "aggregate tracing, then a "
            "short representative Perfetto timeline."
        )
        print()

        print("PHASE 1/3: PYTORCH OPERATOR WINDOW DURING FULL TRAINING")
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
            "distilbert_profile",
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
            raise RuntimeError(
                f"rocprofv3 aggregate pass exited with code {return_code}."
            )
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
            "distilbert_timeline",
            "--perfetto-buffer-size",
            "2097152",
            "--perfetto-buffer-fill-policy",
            "ring_buffer",
            "--",
            sys.executable,
            str(PROFILE_SCRIPT),
            "--internal-worker",
            "timeline",
            "--internal-output",
            str(timeline_worker_dir),
            "--internal-smoke",
        ]
        return_code = stream_subprocess(timeline_command, env=env)
        if return_code != 0:
            raise RuntimeError(
                f"rocprofv3 timeline pass exited with code {return_code}."
            )
        validate_worker(
            timeline_worker_dir, "rocprofv3 timeline", True, timeline=True
        )
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

        window_metadata = json.loads(
            (output_dir / "pytorch_operator_window_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        if int(window_metadata.get("windows_completed", 0)) != expected_profile_windows:
            raise RuntimeError(
                "Unexpected PyTorch profiler window count: "
                f"{window_metadata.get('windows_completed')} instead of "
                f"{expected_profile_windows}."
            )
        if int(window_metadata.get("total_profiled_batches", 0)) != expected_profiled_batches:
            raise RuntimeError(
                "Unexpected profiled batch count: "
                f"{window_metadata.get('total_profiled_batches')} instead of "
                f"{expected_profiled_batches}."
            )
        if int(window_metadata.get("total_training_batches_executed", 0)) != expected_training_batches:
            raise RuntimeError(
                "Unexpected executed training batch count: "
                f"{window_metadata.get('total_training_batches_executed')} instead of "
                f"{expected_training_batches}."
            )

        run_valid = {
            "status": "VALID",
            "timestamp": datetime.now().astimezone().isoformat(),
            "mode": "smoke" if smoke else "official_profiled",
            "primary_metrics_source": "pytorch_operator_pass",
            "pytorch_operator_pass_valid": True,
            "pytorch_operator_windows_completed": int(
                window_metadata["windows_completed"]
            ),
            "pytorch_profiled_batches": int(
                window_metadata["total_profiled_batches"]
            ),
            "pytorch_total_training_batches_executed": int(
                window_metadata["total_training_batches_executed"]
            ),
            "rocprofv3_aggregate_pass_valid": True,
            "rocprofv3_timeline_pass_valid": True,
            "timeline_used_for_quantitative_metrics": False,
            "representative_timeline": str(timeline_files[0]),
            "training_window_filter_applied": parse_metadata[
                "training_window_filter_applied"
            ],
            "training_kernel_dispatches": parse_metadata["training_kernel_rows"],
            "unique_training_kernels": parse_metadata["unique_training_kernels"],
            "gpu_idle_time_ms": parse_metadata["gpu_idle_time_ms"],
            "gpu_idle_percent": parse_metadata["gpu_idle_percent"],
            "top_operator": str(top_operators.iloc[0]["operator"]),
            "top_kernel": str(top_kernels.iloc[0]["kernel_name"]),
        }
        with (output_dir / "RUN_VALID.json").open("w", encoding="utf-8") as file:
            json.dump(run_valid, file, indent=2)

        print("=" * 92)
        print("PROFILED WORKFLOW VALID")
        print("=" * 92)
        print("Primary profiled-pass metrics: full training run with a representative operator window")
        print(f"PyTorch operator windows completed: {run_valid['pytorch_operator_windows_completed']}")
        print(f"Profiled operator batches: {run_valid['pytorch_profiled_batches']}")
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
        description="DistilBERT GLUE/SST-2 AMD profiling workflow"
    )
    public = parser.add_mutually_exclusive_group(required=False)
    public.add_argument(
        "--smoke",
        action="store_true",
        help="Run the complete three-pass profiling smoke test.",
    )
    public.add_argument(
        "--run",
        action="store_true",
        help="Run the complete official three-pass profiled workflow.",
    )
    public.add_argument(
        "--operator-test",
        action="store_true",
        help="Run a full-dataset one-epoch PyTorch operator profiling test.",
    )
    parser.add_argument(
        "--internal-worker",
        choices=("torch_operator", "rocprofv3", "timeline"),
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--internal-output", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--internal-smoke", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--internal-epochs", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.internal_worker:
        if args.internal_output is None:
            parser.error("--internal-output is required for an internal worker.")
        run_worker(
            args.internal_worker,
            args.internal_output,
            args.internal_smoke,
            epochs_override=args.internal_epochs,
        )
        return

    if args.smoke:
        run_parent(smoke=True)
        return
    if args.operator_test:
        run_operator_test()
        return
    if args.run:
        run_parent(smoke=False)
        return
    parser.error("Choose --smoke, --operator-test, or --run.")


if __name__ == "__main__":
    main()

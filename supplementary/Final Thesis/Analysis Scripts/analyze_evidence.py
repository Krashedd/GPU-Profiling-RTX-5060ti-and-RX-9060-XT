"""Reproduce descriptive profiling findings. Python 3.10+, standard library only.

Never runs training or modifies experiment inputs. See Codex Folder/README.md for scope and usage.
Created after collection to consolidate the read-only exploratory analyses.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import statistics
from datetime import datetime, timezone

MODELS = {"resnet": 1960, "yolo": 6250, "distilbert": 10530}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_trace_events(path):
    """Decode the traceEvents array incrementally, without loading the GB file.

    Unlike line-based matching, this supports reordered keys, nested arguments,
    compact JSON and arbitrary line breaks. Missing/truncated arrays raise.
    """
    decoder = json.JSONDecoder()
    pattern = re.compile(r'"traceEvents"\s*:\s*\[')
    with path.open(encoding="utf-8-sig") as stream:
        buffer = ""
        while True:
            block = stream.read(1024 * 1024)
            if not block:
                raise ValueError("traceEvents array not found")
            buffer += block
            match = pattern.search(buffer)
            if match:
                buffer = buffer[match.end():]
                break
            buffer = buffer[-128:]
        pos = 0
        expect_value = True
        first = True
        while True:
            while pos < len(buffer) and buffer[pos].isspace():
                pos += 1
            if pos == len(buffer):
                buffer = stream.read(1024 * 1024)
                pos = 0
                if not buffer:
                    raise ValueError("Truncated traceEvents array")
                continue
            char = buffer[pos]
            if char == "]":
                if expect_value and not first:
                    raise ValueError("Trailing comma in traceEvents")
                return
            if not expect_value:
                if char != ",":
                    raise ValueError("Expected comma between trace events")
                pos += 1
                expect_value = True
                continue
            try:
                event, end = decoder.raw_decode(buffer, pos)
            except json.JSONDecodeError:
                block = stream.read(1024 * 1024)
                if not block:
                    raise ValueError("Invalid or truncated trace event") from None
                buffer = buffer[pos:] + block
                pos = 0
                continue
            if not isinstance(event, dict):
                raise ValueError("Trace event is not an object")
            pos = end
            expect_value = False
            first = False
            yield event


def trace_coverage(path):
    groups = {}
    for event in iter_trace_events(path):
        if event.get("ph") != "X" or "ts" not in event or "dur" not in event:
            continue
        cat = event.get("cat", "uncategorized")
        start, duration = float(event["ts"]), float(event["dur"])
        if duration < 0 or not all(map(math.isfinite, (start, duration))):
            raise ValueError("Invalid event timing")
        g = groups.setdefault(cat, {"events": 0, "first_us": start,
                                    "last_us": start, "summed_duration_us": 0.0})
        g["events"] += 1
        g["first_us"] = min(g["first_us"], start)
        g["last_us"] = max(g["last_us"], start + duration)
        g["summed_duration_us"] += duration
    for g in groups.values():
        g["span_s"] = (g["last_us"] - g["first_us"]) / 1e6
        g["summed_duration_s"] = g.pop("summed_duration_us") / 1e6
    return groups


def kernel_group(name):
    """Transparent name-based families; not exact high-level operator mapping."""
    patterns = [
        (r"^Im2d2Col", "im2col"), (r"^Cijk_", "AMD GEMM"),
        (r"miopenSp3AsmConv", "MIOpen assembly convolution"),
        (r"naive_conv", "naive-named convolution"),
        (r"fmha|attention", "attention"),
        (r"cutlass|gemm", "other GEMM"),
        (r"float16tofloat32_copy", "FP16-to-FP32 conversion"),
        (r"float16_copy", "conversion to FP16"),
        (r"copyBuffer", "runtime copyBuffer"),
        (r"fillBuffer", "runtime fillBuffer"),
        (r"direct_copy", "direct copy"),
        (r"multi_tensor_apply", "multi-tensor kernels"),
        (r"batchnorm|bn_|BatchNorm", "batch normalization"),
        (r"nchwToNhwc|nhwcToNchw", "layout conversion"),
        (r"cudnn.*(wgrad|dgrad|conv)", "cuDNN convolution/gradient"),
    ]
    return next((label for pattern, label in patterns
                 if re.search(pattern, name, re.I)), "other")


def summarize_kernels(rows, batches):
    groups = {}
    calls = sum(int(r["calls"]) for r in rows)
    duration = sum(float(r["total_duration_ms"]) for r in rows)
    for r in rows:
        g = groups.setdefault(kernel_group(r["kernel_name"]),
                              {"calls": 0, "duration_ms": 0.0})
        g["calls"] += int(r["calls"])
        g["duration_ms"] += float(r["total_duration_ms"])
    for g in groups.values():
        g["launch_share_pct"] = 100 * g["calls"] / calls
        g["summed_time_share_pct"] = 100 * g["duration_ms"] / duration
    return {"launches": calls, "launches_per_batch": calls / batches,
            "summed_kernel_s": duration / 1000, "groups": groups}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path,
                        default=Path(os.environ.get("THESIS_DATA_ROOT",
                            str(Path.home() / "Documents/ChatGPT/Thesis"))))
    parser.add_argument("--deep-trace", action="store_true",
                        help="Stream the complete AMD ResNet PyTorch JSON capture")
    parser.add_argument("--output-root", type=Path,
                        help="Directory for timestamped audit reports; defaults to synchronized Codex Folder/QA/profiling_evidence")
    args = parser.parse_args()
    root = args.data_root.resolve(strict=True)
    sync = Path(__file__).resolve().parents[2]
    if root == sync or sync in root.parents:
        raise ValueError("Raw AMD/NVIDIA experiment archives must remain local")
    output_root = (args.output_root or sync / "Codex Folder/QA/profiling_evidence").resolve()
    manifest = {}

    def register(path):
        path = path.resolve(strict=True)
        key = path.relative_to(root).as_posix()
        if key not in manifest:
            manifest[key] = {"size_bytes": path.stat().st_size,
                             "sha256": sha256(path)}
        return path

    def read_csv(path):
        with register(path).open(encoding="utf-8-sig", newline="") as stream:
            return list(csv.DictReader(stream))

    def read_json(path):
        return json.loads(register(path).read_text(encoding="utf-8-sig"))

    result = {"regular": {}, "profile": {}, "limitations": [
        "Regular repeats share seed 55; not a multi-seed accuracy study.",
        "Kernel families are explicit name-based groups, not exact operator attribution.",
        "Summed kernel duration is not wall time or union busy time.",
        "Operator timers differ across vendors; avoid exact causal speedup attribution.",
        "AMD copy/fill kernels and NVIDIA memcpy/memset tables classify work differently.",
        "Trace coverage is event extent, not proof of continuous sampling or root cause.",
        "Native summary totals reconcile to parser metadata, not a fresh scan of every native event.",
    ]}
    for vendor in ("AMD", "NVIDIA"):
        for model, batches in MODELS.items():
            print(f"Checking {vendor}/{model}", flush=True)
            logs = root / vendor / "Thesis-Testing/Logs"
            key = f"{vendor}/{model}"
            summaries = []
            for run in (1, 2, 3):
                d = logs / f"{model}run{run}"
                summary, = read_csv(d / "summary.csv")
                epochs = read_csv(d / "epoch_metrics.csv")
                if len(epochs) != 10 or sum(int(e["batches"]) for e in epochs) != batches:
                    raise ValueError(f"Incomplete epoch totals: {d}")
                if int(summary["total_batches_processed"]) != batches:
                    raise ValueError(f"Summary batch mismatch: {d}")
                read_json(d / "config.json")
                summaries.append(summary)
            rate_key = "throughput_images_s" if model == "yolo" else "throughput_samples_s"
            rates = [float(r[rate_key]) for r in summaries]
            result["regular"][key] = {
                "runs": summaries, "throughput_mean": statistics.mean(rates),
                "throughput_sample_sd": statistics.stdev(rates),
                "ten_epochs_verified_each_run": True}
            d = logs / f"{model}profiledrun"
            read_json(d / "config.json")
            read_json(d / "profile_workflow_config.json")
            meta = read_json(d / ("rocprofv3_parse_metadata.json" if vendor == "AMD"
                                  else "nsys_parse_metadata.json"))
            rows = read_csv(d / ("rocprofv3_parsed_summary.csv" if vendor == "AMD"
                                else "nsys_parsed_kernel_summary.csv"))
            ops = read_csv(d / "pytorch_all_operators.csv")
            stats = summarize_kernels(rows, batches)
            expected = meta.get("training_kernel_rows", meta.get("kernel_rows_streamed"))
            if expected is None and vendor == "NVIDIA":
                expected = read_json(d / "nsys_training_metrics.json")["total_kernel_launches"]
            if stats["launches"] != int(expected):
                raise ValueError(f"Native summary/metadata mismatch: {key}")
            stats.update({"metadata": meta, "kernels": rows, "operators": ops,
                          "all_entries_self_ms": sum(float(r["self_gpu_time_ms"]) for r in ops),
                          "aten_only_self_ms": sum(float(r["self_gpu_time_ms"]) for r in ops
                                                   if r["operator"].startswith("aten::"))})
            if vendor == "NVIDIA":
                db = next((d / "nsight_systems_raw").glob("*.sqlite"))
                # Read-only SQLite; hash large database only once as evidence input.
                register(db)
                with sqlite3.connect(db.as_uri() + "?mode=ro", uri=True) as conn:
                    start, end = meta["training_start_timestamp_ns"], meta["training_end_timestamp_ns"]
                    stats["memset"] = conn.execute(
                        "SELECT count(*),sum(end-start)/1e9,sum(bytes) FROM CUPTI_ACTIVITY_KIND_MEMSET "
                        "WHERE start>=? AND end<=?", (start, end)).fetchone()
                    stats["memcpy_by_kind"] = conn.execute(
                        "SELECT copyKind,count(*),sum(end-start)/1e9,sum(bytes) "
                        "FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>=? AND end<=? GROUP BY copyKind",
                        (start, end)).fetchall()
            elif model != "resnet":
                path = next((d / "rocprofv3_raw").glob("*memory_copy_trace.csv"))
                copies = read_csv(path)
                start, end = meta["training_start_timestamp_ns"], meta["training_end_timestamp_ns"]
                groups = {}
                for r in copies:
                    a, b = int(r["Start_Timestamp"]), int(r["End_Timestamp"])
                    if a >= start and b <= end:
                        g = groups.setdefault(r["Direction"], {"calls": 0, "seconds": 0.0})
                        g["calls"] += 1
                        g["seconds"] += (b-a) / 1e9
                stats["memory_copy_directions"] = groups
            result["profile"][key] = stats

    amd = result["profile"]["AMD/resnet"]
    im2col = [r for r in amd["kernels"] if r["kernel_name"] == "Im2d2Col_v2"]
    six = [r for r in amd["kernels"] if r["kernel_name"].startswith("Cijk_") and int(r["calls"]) == 500000]
    if len(im2col) != 1 or len(six) != 6:
        raise ValueError("The reviewed ResNet pattern changed; inspect inputs")
    combined = sum(int(r["calls"]) for r in im2col + six)
    result["resnet_pattern"] = {"im2col": im2col, "six_gemm_rows": six,
                                "combined_launches": combined,
                                "combined_launch_share_pct": 100 * combined / amd["launches"]}
    trace = root / "AMD/Thesis-Testing/Logs/resnetprofiledrun/pytorch_trace.json"
    if args.deep_trace:
        print("Streaming AMD ResNet PyTorch trace; this may take several minutes...", flush=True)
        register(trace)
        result["amd_resnet_pytorch_coverage"] = trace_coverage(trace)
    else:
        result["amd_resnet_pytorch_coverage"] = {"status": "NOT CHECKED; use --deep-trace"}
    result["source_manifest"] = manifest
    result["analysis_script_sha256"] = sha256(Path(__file__))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = output_root / stamp
    output.mkdir(parents=True, exist_ok=False)
    (output / "evidence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Profiling evidence audit", "", "See evidence.json for exact rows, provenance and hashes.", "",
             "| Dataset | Launches | Launches/batch | Summed kernel seconds |",
             "|---|---:|---:|---:|"]
    for key, p in result["profile"].items():
        lines.append(f"| {key} | {p['launches']:,} | {p['launches_per_batch']:.3f} | {p['summed_kernel_s']:.3f} |")
    lines += ["", f"ResNet im2col + six specified GEMMs: {combined:,} launches "
              f"({result['resnet_pattern']['combined_launch_share_pct']:.4f}%).", "",
              "## AMD ResNet PyTorch coverage", "", "```json",
              json.dumps(result["amd_resnet_pytorch_coverage"], indent=2), "```", "",
              "## Interpretation limits", ""] + [f"- {x}" for x in result["limitations"]]
    (output / "audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Reports: {output}")


if __name__ == "__main__":
    main()

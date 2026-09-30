"""Read-only check of NVIDIA data needed for a matched ResNet operator window.

Run with python -B check_resnet_window_match.py. Prints JSON; no file writes.
Use analyze_evidence.py --deep-trace for full AMD coverage evidence.
"""
import csv
import json
import os
from collections import Counter
from pathlib import Path
from analyze_evidence import iter_trace_events, sha256


def main():
    root = Path(os.environ.get("THESIS_DATA_ROOT",
                str(Path.home() / "Documents/ChatGPT/Thesis")))
    d = root / "NVIDIA/Thesis-Testing/Logs/resnetprofiledrun"
    categories, names = Counter(), Counter()
    argument_keys = set()
    for event in iter_trace_events(d / "pytorch_trace_representative_batch.json"):
        argument_keys.update(event.get("args", {}))
        if event.get("ph") == "X":
            categories[event.get("cat", "")] += 1
            names[event.get("name", "")] += 1
    files = ("pytorch_trace_representative_batch.json",
             "pytorch_all_profile_events.csv", "pytorch_all_operators.csv")
    duplicates = {name: sha256(d / name) == sha256(d / "pytorch_operator_pass" / name)
                  for name in files}
    with (d / "pytorch_all_profile_events.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    print(json.dumps({
        "nvidia_complete_event_categories": dict(categories),
        "nvidia_trace_argument_keys": sorted(argument_keys),
        "representative_batch_convolution_backward_calls": names["aten::convolution_backward"],
        "root_and_worker_files_identical": duplicates,
        "all_profile_events_table_rows": len(rows),
        "all_profile_events_table_columns": list(rows[0]),
        "interpretation": [
            "Saved NVIDIA batch JSON contains CPU events, not batch-level GPU timings.",
            "The all_profile_events CSV is an aggregate by name, not a timestamped event log.",
            "Identical worker copies do not recover batch-level GPU timing detail.",
            "A matched GPU-operator timing comparison cannot be reconstructed from these files.",
            "This does not invalidate the separately collected full native kernel captures."
        ]
    }, indent=2))


if __name__ == "__main__":
    main()

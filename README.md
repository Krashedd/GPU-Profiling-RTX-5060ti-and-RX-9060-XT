# GPU Profiling: RTX 5060 Ti and RX 9060 XT

Companion code and retained results for the thesis comparing ASUS PRIME OC 16 GB NVIDIA RTX 5060 Ti and AMD RX 9060 XT GPUs on ResNet-50, YOLOv8m and DistilBERT training.

## Repository map

| Directory | Contents |
|---|---|
| `AMD/Thesis-Testing/scripts/` | Six original AMD regular/profiling collectors |
| `NVIDIA/Thesis-Testing/scripts/` | Six original NVIDIA regular/profiling collectors |
| Each vendor's `Logs/` | Retained trial summaries, configurations, telemetry and profiling exports |
| Each vendor's `requirements-*.txt` | Recorded vendor environment requirements |
| `supplementary/Final Thesis/` | Exact thesis source snapshot: regular, profiling and analysis scripts |
| `supplementary/Codex Folder/Working Scripts/test_analysis.py` | Four synthetic analysis tests |
| `supplementary/SHA256_manifest.json` | Hashes of all 15 snapshot Python files |
| `supplementary/Appendix_C_excerpt_manifest.json` | Appendix C source ranges and exact excerpts |

The existing vendor folders are preserved to keep paths in saved records meaningful. The supplementary snapshot preserves the paper's directory structure and source line numbers; the 12 collector texts match the vendor copies apart from line endings. The analyzer and diagnostic checker were created after collection. No experiment is rerun by reading this repository.

## Safe analysis and verification

Python 3.10+ is sufficient for the standard-library synthetic tests:

```sh
python -B "supplementary/Codex Folder/Working Scripts/test_analysis.py"
```

From the repository root, analyze the retained summaries with:

```sh
python -B "supplementary/Final Thesis/Analysis Scripts/analyze_evidence.py" --data-root . --output-root ./analysis-output
```

The public archive excludes large raw traces, databases, model weights and some other artifacts. The full audit and `check_resnet_window_match.py` require the complete local archive and may report missing files on a GitHub-only checkout. Set `THESIS_DATA_ROOT` or use `--data-root` to point to that archive. Passing four synthetic tests validates selected parsing and arithmetic conditions, not complete capture coverage or GPU timing accuracy.

## Running collectors

Collectors require the original vendor-specific Linux GPU environment, compatible PyTorch/ROCm or CUDA libraries, vendor profiling tools, datasets and model assets. Inspect each script's path constants and command-line help before execution. Profiling scripts import their paired regular collector; keep the files together or follow their existing import-path handling. Official execution can clear its run-output directory: use a separate experiment directory and inspect the collector first. These are retained research scripts, not a portable one-command benchmark installer.

## Interpreting results

Use regular trials for headline throughput, training time and resource measurements. Profiling passes diagnose execution and include instrumentation overhead. PyTorch operator and native profiler captures are separate executions; do not add their durations. AMD ResNet operator GPU timing is incomplete. Its full native kernel capture remains separate evidence. Kernel families use name-pattern rules, not verified operator-to-kernel mappings. Cumulative kernel durations are not elapsed training time.

## Reproducibility

Appendix C prints selected implementation excerpts; the supplementary snapshot supplies the full code. Cite a Git commit when reusing this repository because the default branch may change. The included hash manifest identifies the exact source snapshot. The public repository does not establish that every runtime dependency, dataset or raw trace needed for a complete rerun is included.

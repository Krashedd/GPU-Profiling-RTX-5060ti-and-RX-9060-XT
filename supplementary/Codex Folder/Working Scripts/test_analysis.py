"""Synthetic tests; run with python -B test_analysis.py. No experiment writes."""
import json
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "Final Thesis/Analysis Scripts"))
from analyze_evidence import iter_trace_events, trace_coverage, summarize_kernels, kernel_group


class AnalysisTests(unittest.TestCase):
    def test_coverage_and_nested_events(self):
        events = [{"args": {"text": "braces } [, inside string"}, "cat": "kernel", "ph": "X", "ts": 10, "dur": 4},
                  {"ph": "X", "cat": "kernel", "ts": 12, "dur": 4},
                  {"ph": "X", "cat": "cpu_op", "ts": 0, "dur": 100},
                  {"ph": "i", "cat": "kernel", "ts": 999}]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "trace.json"
            for indent in (None, 2):
                p.write_text(json.dumps({"traceEvents": events}, indent=indent))
                self.assertEqual(list(iter_trace_events(p)), events)
                g = trace_coverage(p)["kernel"]
                self.assertEqual(g["events"], 2)
                self.assertAlmostEqual(g["span_s"], 6e-6)
                self.assertAlmostEqual(g["summed_duration_s"], 8e-6)

    def test_invalid_capture_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "trace.json"
            for text in ('{}', '{"traceEvents":[{"ph":"X"}', '{"traceEvents":[{},]}'):
                p.write_text(text)
                with self.assertRaises(ValueError):
                    list(iter_trace_events(p))

    def test_event_across_read_boundaries(self):
        events = [{"args": {"text": "x" * (2 * 1024 * 1024)},
                   "ph": "X", "cat": "kernel", "ts": 1, "dur": 2},
                  {"ph": "X", "cat": "kernel", "ts": 4, "dur": 3}]
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "trace.json"
            p.write_text(json.dumps({"traceEvents": events}), encoding="utf-8")
            self.assertEqual(list(iter_trace_events(p)), events)

    def test_counts_not_time_share(self):
        rows = [{"kernel_name": "Im2d2Col_v2", "calls": "9", "total_duration_ms": "1"},
                {"kernel_name": "naive_conv_example", "calls": "1", "total_duration_ms": "9"}]
        result = summarize_kernels(rows, 2)
        self.assertEqual(result["launches_per_batch"], 5)
        self.assertEqual(result["groups"]["im2col"]["launch_share_pct"], 90)
        self.assertEqual(result["groups"]["im2col"]["summed_time_share_pct"], 10)
        self.assertEqual(kernel_group("float16tofloat32_copy_kernel_cuda"), "FP16-to-FP32 conversion")


if __name__ == "__main__":
    unittest.main()

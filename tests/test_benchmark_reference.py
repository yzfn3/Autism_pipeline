from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ams.benchmark.metrics import score_main_timeline
from ams.benchmark.timeline import parse_reference_timeline, parse_main_timeline_csv, timecode_to_sec


class BenchmarkReferenceTests(unittest.TestCase):
    def test_timecode_to_sec(self) -> None:
        self.assertAlmostEqual(timecode_to_sec("00:01:05:12", 24.0), 65.5)

    def test_parse_edl_marker_pairs(self) -> None:
        text = (
            '"#","Source In","Source FPS"\n'
            '"001","00:00:01:00","24.000"\n'
            '"002","00:00:03:00","24.000"\n'
            '"003","00:00:05:12","24.000"\n'
            '"004","00:00:06:00","24.000"\n'
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ref.csv"
            path.write_text(text, encoding="utf-8")
            ref = parse_main_timeline_csv(path)
        self.assertEqual(ref.ranges, [(1.0, 3.0), (5.5, 6.0)])
        self.assertEqual(ref.source, "edl_marker_pairs")

    def test_parse_rttm_selects_longest_speaker(self) -> None:
        text = (
            "SPEAKER sample 1 0.400 6.640 <NA> <NA> spk00 <NA> <NA>\n"
            "SPEAKER sample 1 8.680 55.960 <NA> <NA> spk00 <NA> <NA>\n"
            "SPEAKER sample 1 1.000 1.000 <NA> <NA> spk01 <NA> <NA>\n"
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "ref.rttm"
            path.write_text(text, encoding="utf-8")
            ref = parse_reference_timeline(path)
        self.assertEqual(ref.source, "rttm")
        self.assertEqual(ref.target_speaker, "spk00")
        self.assertEqual(ref.ranges, [(0.4, 7.04), (8.68, 64.64)])

    def test_main_timeline_der_and_speaking_time_error(self) -> None:
        metrics = score_main_timeline(ref=[(0.0, 10.0)], hyp=[(2.0, 12.0)])
        self.assertAlmostEqual(metrics["missed_main_sec"], 2.0)
        self.assertAlmostEqual(metrics["false_alarm_main_sec"], 2.0)
        self.assertAlmostEqual(metrics["der"], 0.4)
        self.assertAlmostEqual(metrics["speaking_time_error_sec"], 0.0)


if __name__ == "__main__":
    unittest.main()

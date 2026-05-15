from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ams.stages.stage_e_finalize import _drop_short, _merge_ranges
from run_wav_rttm_eval import compute_metrics


class TimelineMetricsTests(unittest.TestCase):
    def test_stage_e_merge_and_drop_short_ranges(self) -> None:
        merged = _merge_ranges([(2.0, 3.0), (0.0, 1.0), (1.2, 1.5)], gap=0.25)
        cleaned, dropped = _drop_short(merged, min_dur=0.4)
        self.assertEqual(merged, [(0.0, 1.5), (2.0, 3.0)])
        self.assertEqual(cleaned, [(0.0, 1.5), (2.0, 3.0)])
        self.assertEqual(dropped, 0)

    def test_overlap_metrics_are_computed_from_seconds(self) -> None:
        metrics = compute_metrics(ref=[(0.0, 2.0), (4.0, 6.0)], hyp=[(1.0, 5.0)])
        self.assertTrue(math.isclose(metrics["overlap_sec"], 2.0))
        self.assertTrue(math.isclose(metrics["precision"], 0.5))
        self.assertTrue(math.isclose(metrics["recall"], 0.5))
        self.assertTrue(math.isclose(metrics["f1"], 0.5))


if __name__ == "__main__":
    unittest.main()

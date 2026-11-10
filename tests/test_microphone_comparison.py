import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

from compare_microphones import longest, match_timeline


class MicrophoneComparisonTests(unittest.TestCase):
    def test_paired_match_can_select_a_shorter_phone_cluster(self):
        phone = {"partner": [(4.0, 10.0)], "target": [(0.0, 3.0)]}
        self.assertEqual(longest(phone), "partner")
        self.assertEqual(match_timeline([(0.0, 3.0)], phone), ("target", 1.0))

    def test_empty_or_non_overlapping_anchor_does_not_assign_identity(self):
        phone = {"partner": [(4.0, 10.0)]}
        self.assertEqual(match_timeline([], phone), (None, None))
        self.assertEqual(match_timeline([(0.0, 3.0)], phone), (None, 0.0))

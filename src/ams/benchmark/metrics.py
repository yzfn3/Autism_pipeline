from __future__ import annotations

from typing import Dict, List

from ams.benchmark.timeline import Range, merge_ranges, overlap_duration, subtract_collar, total_duration


def score_main_timeline(ref: List[Range], hyp: List[Range], *, audio_duration_sec: float | None = None, collar_sec: float = 0.0) -> Dict[str, float | None]:
    ref_scored = subtract_collar(merge_ranges(ref), collar_sec)
    hyp_scored = merge_ranges(hyp)
    ref_total = total_duration(ref_scored)
    hyp_total = total_duration(hyp_scored)
    overlap = overlap_duration(ref_scored, hyp_scored)
    missed = max(0.0, ref_total - overlap)
    false_alarm = max(0.0, hyp_total - overlap)
    union = ref_total + hyp_total - overlap

    precision = overlap / hyp_total if hyp_total > 0 else None
    recall = overlap / ref_total if ref_total > 0 else None
    f1 = 2.0 * precision * recall / (precision + recall) if precision is not None and recall is not None and precision + recall > 0 else None
    der = (missed + false_alarm) / ref_total if ref_total > 0 else None
    speaking_time_error = hyp_total - ref_total
    abs_speaking_time_error = abs(speaking_time_error)

    out: Dict[str, float | None] = {
        "ref_main_sec": ref_total,
        "hyp_main_sec": hyp_total,
        "overlap_sec": overlap,
        "missed_main_sec": missed,
        "false_alarm_main_sec": false_alarm,
        "confusion_sec": 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": overlap / union if union > 0 else None,
        "der": der,
        "miss_rate": missed / ref_total if ref_total > 0 else None,
        "false_alarm_rate_vs_ref": false_alarm / ref_total if ref_total > 0 else None,
        "speaking_time_error_sec": speaking_time_error,
        "abs_speaking_time_error_sec": abs_speaking_time_error,
        "speaking_time_error_pct_ref": (speaking_time_error / ref_total * 100.0) if ref_total > 0 else None,
        "abs_speaking_time_error_pct_ref": (abs_speaking_time_error / ref_total * 100.0) if ref_total > 0 else None,
        "coverage_pct_audio": (hyp_total / audio_duration_sec * 100.0) if audio_duration_sec and audio_duration_sec > 0 else None,
        "collar_sec": collar_sec,
    }
    return out

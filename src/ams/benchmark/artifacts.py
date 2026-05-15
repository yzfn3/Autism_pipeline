from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, List

from ams.benchmark.segmentation import SegmenterResult
from ams.benchmark.timeline import Range, merge_ranges


def ranges_from_turns(turns: Iterable[dict[str, Any]]) -> List[Range]:
    return merge_ranges(
        (float(turn["start_sec"]), float(turn["end_sec"])) for turn in turns
    )


def write_segmenter_result(path: Path, result: SegmenterResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": result.model,
        "mode": result.mode,
        "note": result.note,
        "speech_ranges": [
            {"start_sec": float(start), "end_sec": float(end)}
            for start, end in result.speech_ranges
        ],
        "speaker_turns": [
            {
                "start_sec": float(turn["start_sec"]),
                "end_sec": float(turn["end_sec"]),
                "speaker": str(turn.get("speaker", "")),
            }
            for turn in result.speaker_turns
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_segmenter_result(path: Path) -> SegmenterResult:
    data = json.loads(path.read_text(encoding="utf-8"))
    ranges = [
        (float(item["start_sec"]), float(item["end_sec"]))
        for item in data.get("speech_ranges", [])
    ]
    turns = data.get("speaker_turns") or []
    if not ranges and turns:
        ranges = ranges_from_turns(turns)
    return SegmenterResult(
        speech_ranges=merge_ranges(ranges),
        speaker_turns=turns,
        model=str(data.get("model") or path.stem),
        mode=str(data.get("mode") or "unknown"),
        note=str(data.get("note") or ""),
    )

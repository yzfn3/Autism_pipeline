from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from ams.benchmark.artifacts import write_segmenter_result
from ams.benchmark.segmentation import segmenter_factories


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one segmentation/diarization model and write a JSON artifact.")
    parser.add_argument("--wav", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--meta", type=Path)
    args = parser.parse_args()

    os.environ.setdefault("WANDB_MODE", "disabled")
    t0 = time.perf_counter()
    status = "not_available"
    error = ""
    try:
        segmenter = segmenter_factories()[args.model]()
        result = segmenter.run(args.wav)
        write_segmenter_result(args.out, result)
        status = "ok"
    except Exception as exc:
        status = "not_available"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if args.meta:
            args.meta.parent.mkdir(parents=True, exist_ok=True)
            args.meta.write_text(
                f"status={status}\nerror={error}\nwall_time_sec={time.perf_counter() - t0:.3f}\n",
                encoding="utf-8",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

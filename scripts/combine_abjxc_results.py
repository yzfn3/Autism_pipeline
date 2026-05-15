from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd


PYANNOTE_FIRST = Path("benchmark_abjxc_final_visible")
NEMO_FIRST = Path("benchmark_abjxc_final_visible2")
PYANNOTE_TITANET_PROBE = Path("benchmark_abjxc_pyannote_titanet_probe")
OUT = Path("benchmark_abjxc_final_combined")


def main() -> None:
    OUT.mkdir(exist_ok=True)
    pyannote_first = pd.read_csv(PYANNOTE_FIRST / "benchmark_results.csv")
    nemo_first = pd.read_csv(NEMO_FIRST / "benchmark_results.csv")
    pyannote_titanet_probe = pd.read_csv(
        PYANNOTE_TITANET_PROBE / "benchmark_results.csv"
    )
    frames = [pyannote_first, nemo_first, pyannote_titanet_probe]

    keys = sorted(
        set().union(
            *[
                set(map(tuple, frame[["segmenter", "embedding"]].to_numpy()))
                for frame in frames
            ]
        )
    )

    rows = []
    for segmenter, embedding in keys:
        candidates = pd.concat(
            [
                frame[
                    (frame.segmenter == segmenter) & (frame.embedding == embedding)
                ]
                for frame in frames
            ],
            ignore_index=True,
        )
        ok = candidates[candidates.status == "ok"]
        if len(ok):
            row = ok.sort_values("der").iloc[0]
        else:
            errors = candidates.error.fillna("").astype(str)
            cleaner = candidates[
                ~errors.str.contains("k2_fsa|Lazy import", case=False, regex=True)
            ]
            row = cleaner.iloc[-1] if len(cleaner) else candidates.iloc[-1]
        rows.append(row.to_dict())

    combined = pd.DataFrame(rows).sort_values(
        ["status", "der", "segmenter", "embedding"], na_position="last"
    )
    combined.to_csv(OUT / "benchmark_results.csv", index=False)

    cols = [
        "segmenter",
        "embedding",
        "der",
        "f1",
        "recall",
        "precision",
        "speaking_time_error_sec",
        "wall_time_sec",
    ]

    summary = [
        "# abjxc benchmark combined result",
        "",
        "Combined from two full passes: pyannote-first and nemo-first. This avoids a Windows/Python import-order conflict between NeMo, SpeechBrain, and k2_fsa while keeping only real model outputs.",
        "",
        "## Status counts",
        combined.status.value_counts().to_string(),
        "",
        "## Completed combinations by segmenter",
        pd.crosstab(combined.segmenter, combined.status).to_string(),
        "",
        "## Top completed rows",
        combined[combined.status == "ok"]
        .sort_values(["der", "wall_time_sec"])[cols]
        .head(25)
        .to_string(index=False),
        "",
        "## Unavailable combinations",
        combined[combined.status != "ok"]
        .groupby(["segmenter", "embedding"])["error"]
        .first()
        .to_string(),
    ]
    (OUT / "RESULTS_SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")

    shutil.copyfile(
        NEMO_FIRST / "benchmark_report.json", OUT / "benchmark_report_nemo_first.json"
    )
    shutil.copyfile(
        PYANNOTE_FIRST / "benchmark_report.json",
        OUT / "benchmark_report_pyannote_first.json",
    )
    shutil.copyfile(
        PYANNOTE_TITANET_PROBE / "benchmark_report.json",
        OUT / "benchmark_report_pyannote_titanet_probe.json",
    )

    print(OUT.resolve())
    print(combined.status.value_counts().to_string())


if __name__ == "__main__":
    main()

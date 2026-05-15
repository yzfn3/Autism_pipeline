from __future__ import annotations

import dataclasses
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (SRC, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ams.config.schema import RuntimeConfig, load_resolved_config
from ams.stages.stage_utils import params_hash, speaker_embedding_runtime_params


class ConfigAndCacheTests(unittest.TestCase):
    def test_noisy_preset_loads_expected_backend(self) -> None:
        cfg = load_resolved_config(preset="noisy", config_path=None, set_overrides=[])
        self.assertEqual(cfg.runtime.speaker_embedding_backend, "simple_band_energy")
        self.assertGreater(cfg.b.on_thresh, cfg.b.off_thresh)

    def test_embedding_cache_params_include_all_model_ids(self) -> None:
        params = speaker_embedding_runtime_params(RuntimeConfig())
        self.assertEqual(
            set(params),
            {
                "speaker_embedding_backend",
                "prefer_gpu",
                "speechbrain_model_id",
                "wavlm_model_id",
                "titanet_model_id",
                "ecapa2_model_id",
                "wespeaker_campp_model_id",
            },
        )

    def test_embedding_cache_hash_changes_when_non_default_model_id_changes(self) -> None:
        base = RuntimeConfig(speaker_embedding_backend="wavlm_xvector")
        changed = dataclasses.replace(base, wavlm_model_id="example/custom-wavlm")
        self.assertNotEqual(
            params_hash({"runtime": speaker_embedding_runtime_params(base)}),
            params_hash({"runtime": speaker_embedding_runtime_params(changed)}),
        )


if __name__ == "__main__":
    unittest.main()

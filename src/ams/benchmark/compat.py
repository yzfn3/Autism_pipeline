from __future__ import annotations

import os


def patch_speechbrain_lazy_import_windows() -> None:
    """Patch SpeechBrain's LazyModule inspect guard for Windows paths.

    SpeechBrain 1.1.0 checks for `/inspect.py` to avoid accidental lazy imports
    triggered by Python's inspection machinery. On Windows, the filename uses
    backslashes, so the guard misses and optional integrations such as
    `speechbrain.integrations.k2_fsa` can raise during unrelated imports.
    """

    try:
        from speechbrain.utils import importutils
    except Exception:
        return

    if getattr(importutils, "_ams_windows_inspect_patch", False):
        return

    original = importutils.LazyModule.ensure_module

    def patched(self, stacklevel: int):
        importer_frame = None
        try:
            importer_frame = importutils.inspect.getframeinfo(importutils.sys._getframe(stacklevel + 1))
        except AttributeError:
            importutils.warnings.warn(
                "Failed to inspect frame to check if we should ignore importing "
                "a module lazily. This relies on a CPython implementation detail."
            )

        if importer_frame is not None:
            normalized = os.path.normpath(importer_frame.filename).lower().replace("\\", "/")
            if normalized.endswith("/inspect.py"):
                raise AttributeError()

        return original(self, stacklevel)

    importutils.LazyModule.ensure_module = patched
    importutils._ams_windows_inspect_patch = True


def patch_nemo_windows_compat() -> None:
    """Apply NeMo/SpeechBrain compatibility patches needed on Windows."""

    patch_speechbrain_lazy_import_windows()

    try:
        import nemo.core.classes as classes
        import nemo.core.classes.dataset as dataset
    except Exception:
        return

    if not hasattr(classes, "dataset"):
        classes.dataset = dataset

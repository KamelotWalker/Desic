"""Laya-style neural student (optional: ``pip install 'desic[neural]'``).

Import of this package is torch-free; the heavy modules (model, train) are
loaded only when a neural checkpoint is trained or used.
"""

from __future__ import annotations


def available() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except Exception as e:  # pragma: no cover - environment dependent
        return False, f"PyTorch is not installed ({type(e).__name__}); pip install 'desic[neural]'"
    return True, ""

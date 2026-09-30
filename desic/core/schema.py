"""Value helpers: missing values, numbers and labels from messy user data."""

from __future__ import annotations

import math
from typing import Any

_MISSING = {"", "na", "n/a", "nan", "null", "none", "?"}


def _is_missing(v: Any) -> bool:
    return v is None or (isinstance(v, str) and v.strip().lower() in _MISSING)


def to_number(v: Any) -> float | None:
    if _is_missing(v) or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        f = float(v)
    else:
        try:
            f = float(str(v).strip().replace(",", "."))
        except ValueError:
            return None
    return f if math.isfinite(f) else None


def to_label(v: Any) -> str | None:
    """Normalise a target value to a string label ("1.0" and 1 both -> "1")."""
    if _is_missing(v):
        return None
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return str(v).strip()

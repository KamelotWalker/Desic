"""Feature schema: typing, coercion and inference from raw rows."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

NUMERIC, CATEGORICAL = "numeric", "categorical"
MAX_TRACKED_VALUES = 50
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


@dataclass
class Feature:
    name: str
    type: str = NUMERIC
    values: list[str] = field(default_factory=list)  # known categories (UI hints)
    description: str = ""

    def coerce(self, v: Any) -> Any:
        if self.type == NUMERIC:
            return to_number(v)
        if _is_missing(v):
            return None
        return to_label(v)


@dataclass
class Schema:
    features: list[Feature]
    target: str
    classes: list[str] = field(default_factory=list)

    @property
    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    def coerce(self, raw: dict) -> dict:
        return {f.name: f.coerce(raw.get(f.name)) for f in self.features}

    def observe_label(self, label: str) -> None:
        if label not in self.classes:
            self.classes.append(label)

    def observe_row(self, x: dict) -> None:
        for f in self.features:
            if f.type == CATEGORICAL:
                v = x.get(f.name)
                if v is not None and v not in f.values and len(f.values) < MAX_TRACKED_VALUES:
                    f.values.append(v)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Schema":
        return cls(
            features=[Feature(**f) for f in d["features"]],
            target=d["target"],
            classes=list(d.get("classes", [])),
        )

    @classmethod
    def infer(cls, rows: list[dict], target: str, features: list[str] | None = None) -> "Schema":
        if not rows:
            raise ValueError("cannot infer a schema from an empty dataset")
        columns = features or [c for c in rows[0] if c != target]
        feats = []
        for col in columns:
            if col == target:
                continue
            present = [r.get(col) for r in rows if not _is_missing(r.get(col))]
            numeric = bool(present) and all(to_number(v) is not None for v in present)
            if numeric:
                feats.append(Feature(col, NUMERIC))
            else:
                values = list(dict.fromkeys(to_label(v) for v in present))[:MAX_TRACKED_VALUES]
                feats.append(Feature(col, CATEGORICAL, values))
        classes = list(dict.fromkeys(l for l in (to_label(r.get(target)) for r in rows) if l is not None))
        return cls(feats, target, classes)

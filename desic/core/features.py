"""Turn a decision *state* (free text or JSON) into features.

Two views of the same state are produced:

* ``sparse`` – named features for the linear and memory experts:
  ``w:refund`` (word), ``p5:iades`` (5-char prefix, a strong light stemmer for
  Turkish and other agglutinative languages), ``b:charged_twice`` (bigram),
  ``k:customer.plan=pro`` (categorical field), ``n:customer.age`` (standardised
  number) and ``bias``.
* ``scalars`` – flat ``path -> number | short string`` values for the tree
  expert, which is good at thresholds such as ``credit_score > 1420``.

Text is case-folded and ASCII-folded (ş→s, ı→i, ö→o …) so "şikayet" and
"sikayet" are the same token.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from .stats import Gaussian

TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_FOLD = str.maketrans({"ı": "i", "ş": "s", "ğ": "g", "ç": "c", "ö": "o", "ü": "u", "â": "a", "î": "i", "û": "u"})
SHORT_TEXT_CHARS = 40  # strings up to this length (and <= 4 words) are treated as categories
MAX_TOKENS = 400       # long texts are truncated for speed


def fold(text: str) -> str:
    text = text.replace("İ", "i").lower().translate(_FOLD)
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text if not unicodedata.combining(c))


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(fold(text))[:MAX_TOKENS]


def state_text(state: Any) -> str:
    """A readable one-line rendering of the state (for LLM prompts and search)."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False, sort_keys=True)


def state_hash(state: Any) -> str:
    raw = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, sort_keys=True)
    return hashlib.blake2b(raw.strip().encode(), digest_size=10).hexdigest()


def flatten(state: Any, prefix: str = "", out: dict | None = None, depth: int = 0) -> dict[str, Any]:
    """{"a": {"b": 1}, "tags": ["x"]} -> {"a.b": 1, "tags[]": ["x"]}; a bare string -> {"$text": s}."""
    if out is None:
        out = {}
    if depth > 8:
        return out
    if isinstance(state, dict):
        for k, v in state.items():
            flatten(v, f"{prefix}.{k}" if prefix else str(k), out, depth + 1)
    elif isinstance(state, list):
        if all(not isinstance(v, (dict, list)) for v in state):
            out[f"{prefix}[]"] = state
        else:
            for i, v in enumerate(state[:20]):
                flatten(v, f"{prefix}[{i}]", out, depth + 1)
    else:
        out[prefix or "$text"] = state
    return out


def _is_short(s: str) -> bool:
    return len(s) <= SHORT_TEXT_CHARS and len(s.split()) <= 4


@dataclass
class Features:
    sparse: dict[str, float]
    scalars: dict[str, Any]
    flat: dict[str, Any]  # for hard rules
    text_features: int = 0

    def norm(self) -> float:
        return math.sqrt(sum(v * v for v in self.sparse.values())) or 1.0


@dataclass
class Featurizer:
    """Stateful only for number scaling (running mean/std per numeric field)."""

    numeric: dict[str, Gaussian] = field(default_factory=dict)

    def extract(self, state: Any) -> Features:
        flat = flatten(state)
        text_feats: dict[str, float] = {}
        cat_feats: dict[str, float] = {}
        num_feats: dict[str, float] = {}
        scalars: dict[str, Any] = {}
        for path, v in flat.items():
            if v is None:
                continue
            if isinstance(v, bool):
                cat_feats[f"k:{path}={str(v).lower()}"] = 1.0
                scalars[path] = str(v).lower()
            elif isinstance(v, (int, float)):
                if not math.isfinite(v):
                    continue
                g = self.numeric.get(path)
                if g is not None and g.n >= 2 and g.std > 1e-9:
                    num_feats[f"n:{path}"] = max(-4.0, min(4.0, (v - g.mean) / g.std))
                scalars[path] = float(v)
            elif isinstance(v, list):
                for item in v[:30]:
                    cat_feats[f"k:{path}={fold(str(item))}"] = 1.0
                scalars[f"{path}#len"] = float(len(v))
            else:
                s = str(v).strip()
                if not s:
                    continue
                if path != "$text" and _is_short(s):
                    cat_feats[f"k:{path}={fold(s)}"] = 1.0
                    scalars[path] = fold(s)
                self._text(s, text_feats)
        # Text features share a unit budget so long texts don't drown structured fields.
        if text_feats:
            scale = 1.0 / math.sqrt(len(text_feats))
            text_feats = {k: v * scale for k, v in text_feats.items()}
        sparse = {"bias": 1.0, **text_feats, **cat_feats, **num_feats}
        return Features(sparse, scalars, flat, len(text_feats))

    @staticmethod
    def _text(s: str, out: dict[str, float]) -> None:
        toks = tokens(s)
        prev = None
        for t in toks:
            out[f"w:{t}"] = 1.0
            if len(t) > 5 and not t.isdigit():
                out[f"p5:{t[:5]}"] = 1.0
            if prev is not None:
                out[f"b:{prev}_{t}"] = 1.0
            prev = t

    def observe(self, feats: Features) -> None:
        """Update number scaling with a *labelled* example (never with raw traffic)."""
        for path, v in feats.flat.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
                self.numeric.setdefault(path, Gaussian()).update(float(v))

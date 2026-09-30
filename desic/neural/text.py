"""Turning a typed question + state into encoder input (torch-free).

Laya-style layout: the question and every allowed answer come first, each
answer introduced by a marker that becomes the tokenizer's ``[MASK]`` token;
the state follows and is the only part that is truncated.

    [CLS] choice: Which team? [MASK] billing: payments [MASK] technical: bugs … [SEP] <state> [SEP]

The encoder reads everything at once (bidirectional), and the decision head
scores each ``[MASK]`` position — so answers can be new at request time.
"""

from __future__ import annotations

import math
import re
import zlib
from typing import Any

from ..core.features import flatten, fold

MARK = "⟦M⟧"  # placeholder, replaced by the backbone's mask token
NUM_RE = re.compile(r"\d+(?:[.,]\d+)?|\w+", re.UNICODE)
NOUL_OPTIONS = {"true": "yes, the statement is true", "false": "no, the statement is false"}


def question_text(qtype: str, instructions: str, options: dict[str, str]) -> str:
    parts = [f"{qtype}: {instructions}".strip()]
    for name, desc in options.items():
        desc = desc or (NOUL_OPTIONS.get(name, "") if qtype == "noul" else "")
        parts.append(f"{MARK} {name}" + (f": {desc}" if desc else ""))
    return " ".join(parts)


def state_to_text(state: Any, limit: int = 4000) -> str:
    """JSON becomes 'path: value | path: value' — easier for a language encoder than raw JSON."""
    if isinstance(state, str):
        return state[:limit]
    parts = []
    for path, v in flatten(state).items():
        if v is None:
            continue
        if isinstance(v, list):
            v = ", ".join(str(x) for x in v)
        parts.append(f"{path}: {v}")
    return " | ".join(parts)[:limit]


class ScratchTokenizer:
    """Hashing tokenizer for the built-in encoder (no downloads).

    Words are case/ASCII-folded and cut to a 6-character stem (a light stemmer
    that suits agglutinative languages such as Turkish), numbers become
    order-of-magnitude buckets, and everything is hashed into a fixed vocabulary.
    """

    PAD, CLS, SEP, MASK = 0, 1, 2, 3
    SPECIALS = 4

    def __init__(self, vocab_size: int = 16384, stem: int = 6) -> None:
        self.vocab_size = vocab_size
        self.stem = stem

    def _ids(self, text: str) -> list[int]:
        out = []
        for w in NUM_RE.findall(fold(text)):
            if w[0].isdigit():
                # numbers become magnitude buckets (quarter decades): 1640 and 1700 share a token
                v = float(w.replace(",", "."))
                w = f"#num{round(math.log10(v) * 4) if v > 0 else 'zero'}"
            else:
                w = w[: self.stem]
            out.append(self.SPECIALS + zlib.crc32(w.encode()) % (self.vocab_size - self.SPECIALS))
        return out

    def encode(self, first: str, second: str, max_len: int) -> list[int] | None:
        ids = [self.CLS]
        for i, chunk in enumerate(first.split(MARK)):
            if i:
                ids.append(self.MASK)
            ids.extend(self._ids(chunk))
        ids.append(self.SEP)
        if len(ids) > max_len - 8:
            return None  # the answers alone don't fit
        ids.extend(self._ids(second)[: max_len - len(ids) - 1])
        ids.append(self.SEP)
        return ids

    def config(self) -> dict:
        return {"vocab_size": self.vocab_size, "stem": self.stem}

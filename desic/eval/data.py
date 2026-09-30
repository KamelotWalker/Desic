"""Benchmark data, downloaded once and cached on disk."""

from __future__ import annotations

import csv
import io
import os
import urllib.request
from pathlib import Path

BANKING77 = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/"


def cache_dir() -> Path:
    return Path(os.environ.get("DESIC_CACHE") or Path.home() / ".cache" / "desic")


def banking77() -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Banking77 (PolyAI): 10,003 train / 3,080 test customer messages, 77 intents.

    The train file is sorted by intent, and is returned in file order.
    """
    root = cache_dir() / "banking77"
    root.mkdir(parents=True, exist_ok=True)
    out = []
    for split in ("train", "test"):
        path = root / f"{split}.csv"
        if not path.exists():
            with urllib.request.urlopen(BANKING77 + f"{split}.csv", timeout=60) as r:
                path.write_bytes(r.read())
        rows = csv.DictReader(io.StringIO(path.read_text(encoding="utf-8")))
        out.append([(row["text"], row["category"]) for row in rows])
    return out[0], out[1]

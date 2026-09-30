"""Parsing user-supplied datasets (CSV / TSV / JSON / JSON Lines)."""

from __future__ import annotations

import csv
import io
import json

MAX_ROWS = 200_000


class DatasetError(ValueError):
    pass


def _clean_row(row: dict) -> dict:
    out = {}
    for k, v in row.items():
        if k is None:
            continue
        key = str(k).strip()
        if not key:
            continue
        out[key] = v.strip() if isinstance(v, str) else v
    return out


def parse_csv(text: str) -> list[dict]:
    sample = text[:20_000]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    except csv.Error:
        dialect = csv.excel
    reader = csv.DictReader(io.StringIO(text), dialect=dialect)
    if not reader.fieldnames:
        raise DatasetError("CSV has no header row")
    rows = []
    for i, row in enumerate(reader):
        if i >= MAX_ROWS:
            break
        cleaned = _clean_row(row)
        if any(v not in (None, "") for v in cleaned.values()):
            rows.append(cleaned)
    return rows


def parse_json(text: str) -> list[dict]:
    text = text.strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # JSON Lines
        data = []
        for n, line in enumerate(text.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise DatasetError(f"invalid JSON on line {n}: {e.msg}") from None
    if isinstance(data, dict):
        for key in ("rows", "data", "records", "items"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list) or not all(isinstance(r, dict) for r in data):
        raise DatasetError("JSON must be a list of objects (or {\"rows\": [...]})")
    return [_clean_row(r) for r in data[:MAX_ROWS]]


def parse_dataset(filename: str, content: bytes) -> tuple[list[str], list[dict]]:
    try:
        text = content.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = content.decode("latin-1")
    name = filename.lower()
    if name.endswith((".json", ".jsonl", ".ndjson")) or text.lstrip()[:1] in ("[", "{"):
        rows = parse_json(text)
    else:
        rows = parse_csv(text)
    if not rows:
        raise DatasetError("dataset is empty")
    columns = list(dict.fromkeys(k for r in rows for k in r))
    return columns, rows

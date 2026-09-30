import asyncio

import pytest

from desic.generation import GenerationError, normalize_design, rows_schema, validate_rows, generate_rows, design_schema

DESIGN = {
    "name": "loan",
    "target": "decision",
    "classes": ["approve", "reject"],
    "decision_logic": "approve when income is high",
    "features": [
        {"name": "income", "type": "numeric", "description": "", "values": [], "min": 0, "max": 100},
        {"name": "city", "type": "categorical", "description": "", "values": ["a", "b"], "min": 0, "max": 0},
    ],
}


class FakeProvider:
    def __init__(self):
        self.calls = 0

    async def complete_json(self, system, prompt, schema):
        self.calls += 1
        if "Design a dataset" in prompt:
            return DESIGN
        size = int(prompt.split("Generate exactly ")[1].split()[0])
        rows = [{"income": i * 3, "city": "a" if i % 2 else "b", "decision": "approve" if i > 5 else "reject"} for i in range(size)]
        rows.append({"income": "oops", "city": "a", "decision": "approve"})  # invalid → dropped
        rows.append({"income": 1, "city": "zzz", "decision": "approve"})     # unknown category → dropped
        return {"rows": rows}


def test_design_and_generate():
    p = FakeProvider()
    design = asyncio.run(design_schema(p, "approve loans"))
    assert design["target"] == "decision" and len(design["features"]) == 2
    seen = []

    async def progress(done, total):
        seen.append((done, total))

    rows = asyncio.run(generate_rows(p, design, "approve loans", 100, progress))
    assert len(rows) == 100
    assert all(set(r) == {"income", "city", "decision"} for r in rows)
    assert p.calls == 1 + 3  # design + ceil(100 / 40) batches
    assert seen[-1][0] == 100


def test_rows_schema_is_strict():
    s = rows_schema(normalize_design(DESIGN))
    row = s["properties"]["rows"]["items"]
    assert row["additionalProperties"] is False
    assert row["properties"]["city"]["enum"] == ["a", "b"]
    assert row["properties"]["decision"]["enum"] == ["approve", "reject"]


def test_validate_rows_filters():
    rows = validate_rows(normalize_design(DESIGN), [{"income": "5", "city": "a", "decision": "approve"}, {"city": "a"}, "junk"])
    assert rows == [{"income": 5.0, "city": "a", "decision": "approve"}]


def test_normalize_design_rejects_bad():
    with pytest.raises(GenerationError):
        normalize_design({**DESIGN, "classes": ["only"]})
    with pytest.raises(GenerationError):
        normalize_design({**DESIGN, "features": [{"name": "c", "type": "categorical", "values": []}]})


def test_all_batches_failing_raises():
    class Broken:
        async def complete_json(self, *a):
            raise GenerationError("boom")

    with pytest.raises(GenerationError):
        asyncio.run(generate_rows(Broken(), normalize_design(DESIGN), "x", 50))

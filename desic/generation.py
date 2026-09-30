"""Synthetic training data for a typed question, written by the user's own LLM.

1. ``design_task`` – a plain-language description ("route support tickets to
   billing / technical / sales") becomes a typed question: type, instructions,
   answers with descriptions, and the shape of a realistic state.
2. ``generate_examples`` – the LLM writes realistic, diverse states with the
   correct answer, in batches, validated against the question.

The result is an ordinary dataset: review it, then train the student on it.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, Awaitable, Callable

from .core.task import CHOICE, NOUL, TYPES, QuestionSpec
from .llm import JevCompatibleProvider, LLMError

BATCH_SIZE = 20
MAX_EXAMPLES = 5000

SYSTEM = (
    "You design and write realistic training data for Desic, a fast typed-decision model. States must look like "
    "real application data (messages, tickets, records), be diverse in wording, length, tone and language mix, and "
    "the labelled answer must be what a careful domain expert would choose."
)

DESIGN_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "short snake_case question name"},
        "type": {"type": "string", "enum": list(TYPES)},
        "instructions": {"type": "string", "description": "the question, as asked at decision time"},
        "answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "description": {"type": "string"}},
                "required": ["name", "description"],
                "additionalProperties": False,
            },
        },
        "state_format": {"type": "string", "enum": ["text", "json"]},
        "state_description": {"type": "string", "description": "what a state contains"},
        "decision_guidelines": {"type": "string", "description": "how an expert decides, incl. edge cases"},
    },
    "required": ["name", "type", "instructions", "answers", "state_format", "state_description", "decision_guidelines"],
    "additionalProperties": False,
}


def _require_json_provider(provider: Any) -> None:
    if isinstance(provider, JevCompatibleProvider):
        raise LLMError("data generation needs a generative LLM (Anthropic or OpenAI-compatible), not a decision API")


async def design_task(provider: Any, description: str) -> dict:
    _require_json_provider(provider)
    prompt = (
        f"Design a typed decision question for this need:\n\n{description}\n\n"
        "Pick the type: 'choice' (one of several answers), 'score' (ordered levels, list them lowest first) or "
        "'noul' (a yes/no proposition; answers must be exactly 'true' and 'false'). Use 2-12 answers with short "
        "snake_case names and a one-line description each. Choose 'text' states for messages/documents and 'json' "
        "for structured records."
    )
    return normalize_design(await provider.complete_json(SYSTEM, prompt, DESIGN_SCHEMA))


def normalize_design(d: dict) -> dict:
    qtype = d.get("type") if d.get("type") in TYPES else CHOICE
    answers = {}
    for a in d.get("answers", []):
        name = str(a.get("name", "")).strip() if isinstance(a, dict) else str(a).strip()
        if name:
            answers[name] = str(a.get("description", "")) if isinstance(a, dict) else ""
    if qtype == NOUL:
        answers = {"true": answers.get("true", ""), "false": answers.get("false", "")}
    name = "".join(c if c.isalnum() or c in "_-." else "_" for c in str(d.get("name") or "decision"))[:64] or "decision"
    spec = QuestionSpec.parse(name, {"type": qtype, "instructions": d.get("instructions", ""), "criteria": answers})
    return {
        **spec.to_dict(),
        "state_format": "json" if d.get("state_format") == "json" else "text",
        "state_description": str(d.get("state_description", "")),
        "decision_guidelines": str(d.get("decision_guidelines", "")),
    }


def examples_schema(spec: QuestionSpec) -> dict:
    return {
        "type": "object",
        "properties": {
            "examples": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "state": {"type": "string"},
                        "answer": {"type": "string", "enum": spec.option_names},
                    },
                    "required": ["state", "answer"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["examples"],
        "additionalProperties": False,
    }


def validate_examples(spec: QuestionSpec, items: Any, state_format: str) -> list[dict]:
    out = []
    for it in items if isinstance(items, list) else []:
        if not isinstance(it, dict) or it.get("answer") not in spec.options:
            continue
        state: Any = str(it.get("state", "")).strip()
        if not state:
            continue
        if state_format == "json":
            try:
                parsed = json.loads(state)
            except json.JSONDecodeError:
                continue
            if not isinstance(parsed, dict):
                continue
            state = parsed
        out.append({"state": state, "answer": it["answer"]})
    return out


async def generate_examples(
    provider: Any,
    design: dict,
    description: str,
    n: int,
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
    concurrency: int = 3,
) -> list[dict]:
    _require_json_provider(provider)
    spec = QuestionSpec.parse(design["name"], {"type": design["type"], "instructions": design.get("instructions", ""),
                                               "criteria": design["options"]})
    fmt = "json" if design.get("state_format") == "json" else "text"
    n = max(1, min(int(n), MAX_EXAMPLES))
    n_batches = (n + BATCH_SIZE - 1) // BATCH_SIZE
    answers = "\n".join(f"- {k}" + (f": {v}" if v else "") for k, v in spec.options.items())
    kind = {"choice": "one answer", "score": "one level (listed lowest first)", "noul": "true or false"}[spec.type]
    fmt_note = ("Each state is plain text (a message, ticket, note …)." if fmt == "text" else
                "Each state is a JSON object serialised as a string (realistic field names and values).")
    schema = examples_schema(spec)
    sem = asyncio.Semaphore(concurrency)
    results: list[list[dict]] = [[] for _ in range(n_batches)]
    done = 0
    lock = asyncio.Lock()

    async def run(i: int) -> None:
        nonlocal done
        size = min(BATCH_SIZE, n - i * BATCH_SIZE)
        focus = spec.option_names[i % len(spec.option_names)]
        prompt = (
            f"Need: {description}\n\nQuestion ({spec.type}): {spec.instructions or spec.name}\n"
            f"Answers ({kind}):\n{answers}\n\n"
            f"State: {design.get('state_description', '')}\n{fmt_note}\n"
            f"Expert guidelines: {design.get('decision_guidelines', '')}\n\n"
            f"Write exactly {size} examples. Batch {i + 1} of {n_batches}: vary wording, length and tone, include "
            f"hard and borderline cases, some noise (typos, mixed Turkish/English where natural), and slightly more "
            f"'{focus}' cases than usual while still covering every answer."
        )
        async with sem:
            data = await provider.complete_json(SYSTEM, prompt, schema)
        rows = validate_examples(spec, data.get("examples", []) if isinstance(data, dict) else [], fmt)
        results[i] = rows
        async with lock:
            done += len(rows)
            if on_progress:
                await on_progress(done, n)

    outcomes = await asyncio.gather(*(run(i) for i in range(n_batches)), return_exceptions=True)
    errors = [e for e in outcomes if isinstance(e, BaseException)]
    rows = [r for batch in results for r in batch]
    if not rows:
        if errors:
            raise errors[0]
        raise LLMError("the model did not return any valid examples")
    return rows[:n]


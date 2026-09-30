"""Synthetic dataset generation with the user's own LLM API key.

Two steps:
1. ``design_schema`` – turn a plain-language description of a decision
   ("approve a small-business loan") into features, classes and the hidden
   decision logic.
2. ``generate_rows`` – ask the LLM for labelled rows in batches, validated
   against that schema.

API keys are passed per request and are never stored or logged by Desic.
Providers: Anthropic (Claude) through the official SDK, and any
OpenAI-compatible Chat Completions endpoint (OpenAI, Gemini, Groq,
OpenRouter, Ollama, LM Studio, ...).
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

import anthropic
import httpx

from .core.schema import CATEGORICAL, NUMERIC, to_label, to_number

DEFAULT_ANTHROPIC_MODEL = "claude-opus-5-5"
# Models that accept output_config.effort and server-side refusal fallbacks.
_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
BATCH_SIZE = 40
MAX_ROWS = 5000


class GenerationError(RuntimeError):
    pass


class JSONProvider(Protocol):
    async def complete_json(self, system: str, prompt: str, schema: dict) -> Any: ...


@dataclass
class ProviderConfig:
    provider: str = "anthropic"  # "anthropic" | "openai"
    api_key: str = ""
    model: str = ""
    base_url: str = ""

    def __repr__(self) -> str:  # never leak the key into logs / tracebacks
        return f"ProviderConfig(provider={self.provider!r}, model={self.model!r}, base_url={self.base_url!r})"


class AnthropicProvider:
    def __init__(self, api_key: str, model: str = "") -> None:
        # An empty key falls back to the server's own credentials (ANTHROPIC_API_KEY etc.).
        self.client = anthropic.AsyncAnthropic(api_key=api_key or None, max_retries=3)
        self.model = model or DEFAULT_ANTHROPIC_MODEL

    async def complete_json(self, system: str, prompt: str, schema: dict) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 16000,
            "system": system,
            "messages": [{"role": "user", "content": prompt}],
        }
        output_config: dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        try:
            if self.model in _FALLBACK_MODELS:
                output_config["effort"] = "medium"
                resp = await self.client.beta.messages.create(
                    **kwargs,
                    output_config=output_config,
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                )
            else:
                resp = await self.client.messages.create(**kwargs, output_config=output_config)
        except anthropic.AuthenticationError:
            raise GenerationError("Anthropic rejected the API key") from None
        except anthropic.PermissionDeniedError:
            raise GenerationError("this API key is not allowed to use that model") from None
        except anthropic.NotFoundError:
            raise GenerationError(f"unknown Anthropic model {self.model!r}") from None
        except anthropic.RateLimitError:
            raise GenerationError("rate limited by Anthropic, try again shortly or request fewer rows") from None
        except anthropic.BadRequestError as e:
            raise GenerationError(f"Anthropic rejected the request: {e.message}") from None
        except anthropic.APIStatusError as e:
            raise GenerationError(f"Anthropic API error ({e.status_code}): {e.message}") from None
        except anthropic.APIConnectionError:
            raise GenerationError("could not reach the Anthropic API") from None

        if resp.stop_reason == "refusal":
            raise GenerationError("the model declined to generate this data; try rephrasing the description")
        if resp.stop_reason == "max_tokens":
            raise GenerationError("the response was cut off; request fewer rows per batch")
        text = next((b.text for b in resp.content if b.type == "text"), "")
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            raise GenerationError("the model returned invalid JSON") from None


class OpenAICompatibleProvider:
    def __init__(self, api_key: str, model: str, base_url: str = "") -> None:
        if not model:
            raise GenerationError("model name is required for OpenAI-compatible providers")
        self.api_key = api_key
        self.model = model
        self.base_url = (base_url or "https://api.openai.com/v1").rstrip("/")

    async def complete_json(self, system: str, prompt: str, schema: dict) -> Any:
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt + "\n\nRespond with a single JSON object that matches this JSON Schema:\n"
                 + json.dumps(schema)},
            ],
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            async with httpx.AsyncClient(timeout=180) as client:
                r = await client.post(f"{self.base_url}/chat/completions", json=body, headers=headers)
        except httpx.HTTPError as e:
            raise GenerationError(f"could not reach {self.base_url}: {type(e).__name__}") from None
        if r.status_code in (401, 403):
            raise GenerationError("the provider rejected the API key")
        if r.status_code >= 400:
            raise GenerationError(f"provider error {r.status_code}: {r.text[:300]}")
        try:
            content = r.json()["choices"][0]["message"]["content"]
            return json.loads(content)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError):
            raise GenerationError("the provider returned an unexpected or non-JSON response") from None


def make_provider(cfg: ProviderConfig) -> JSONProvider:
    if cfg.provider == "anthropic":
        return AnthropicProvider(cfg.api_key, cfg.model)
    if cfg.provider in ("openai", "openai_compatible"):
        return OpenAICompatibleProvider(cfg.api_key, cfg.model, cfg.base_url)
    raise GenerationError(f"unknown provider {cfg.provider!r}")


# ---------------------------------------------------------------- schema design
SYSTEM = (
    "You design and generate realistic tabular training data for Desic, a self-learning decision engine. "
    "Data must be plausible for the domain, internally consistent, and useful for learning a decision."
)

DESIGN_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "description": "short snake_case name for the decision model"},
        "target": {"type": "string", "description": "snake_case name of the decision column"},
        "classes": {"type": "array", "items": {"type": "string"}, "description": "possible decisions"},
        "decision_logic": {"type": "string", "description": "the rules a domain expert would use"},
        "features": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "type": {"type": "string", "enum": [NUMERIC, CATEGORICAL]},
                    "description": {"type": "string"},
                    "values": {"type": "array", "items": {"type": "string"}},
                    "min": {"type": "number"},
                    "max": {"type": "number"},
                },
                "required": ["name", "type", "description", "values", "min", "max"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["name", "target", "classes", "decision_logic", "features"],
    "additionalProperties": False,
}


async def design_schema(provider: JSONProvider, description: str, n_features: int = 6) -> dict:
    prompt = (
        f"Design a dataset for this decision problem:\n\n{description}\n\n"
        f"Use about {n_features} input features that a real system would have at decision time "
        "(no identifiers, no leakage of the answer). For categorical features list every allowed value in "
        "`values` and set min/max to 0. For numeric features leave `values` empty and give a realistic min/max. "
        "`classes` are the possible decisions (2-6). `decision_logic` explains, in a few sentences, how an "
        "expert maps features to the decision, including trade-offs and exceptions."
    )
    return normalize_design(await provider.complete_json(SYSTEM, prompt, DESIGN_SCHEMA))


def normalize_design(d: dict) -> dict:
    feats = []
    seen = set()
    for f in d.get("features", []):
        name = str(f.get("name", "")).strip()
        if not name or name in seen or name == d.get("target"):
            continue
        seen.add(name)
        ftype = CATEGORICAL if f.get("type") == CATEGORICAL else NUMERIC
        feat = {"name": name, "type": ftype, "description": str(f.get("description", ""))}
        if ftype == CATEGORICAL:
            feat["values"] = [str(v) for v in f.get("values", []) if str(v).strip()]
            if not feat["values"]:
                raise GenerationError(f"categorical feature {name!r} needs at least one value")
        else:
            feat["min"] = float(f.get("min", 0) or 0)
            feat["max"] = float(f.get("max", 0) or 0)
        feats.append(feat)
    classes = [str(c) for c in d.get("classes", []) if str(c).strip()]
    if not feats:
        raise GenerationError("the design has no features")
    if len(classes) < 2:
        raise GenerationError("the design needs at least two classes")
    return {
        "name": str(d.get("name") or "generated_model"),
        "target": str(d.get("target") or "decision"),
        "classes": classes,
        "decision_logic": str(d.get("decision_logic", "")),
        "features": feats,
    }


# ------------------------------------------------------------------ generation
def rows_schema(design: dict) -> dict:
    props: dict[str, Any] = {}
    for f in design["features"]:
        if f["type"] == CATEGORICAL:
            props[f["name"]] = {"type": "string", "enum": f["values"]}
        else:
            props[f["name"]] = {"type": "number"}
    props[design["target"]] = {"type": "string", "enum": design["classes"]}
    row = {"type": "object", "properties": props, "required": list(props), "additionalProperties": False}
    return {
        "type": "object",
        "properties": {"rows": {"type": "array", "items": row}},
        "required": ["rows"],
        "additionalProperties": False,
    }


def _describe_columns(design: dict) -> str:
    lines = []
    for f in design["features"]:
        if f["type"] == CATEGORICAL:
            lines.append(f"- {f['name']} (one of: {', '.join(f['values'])}): {f.get('description', '')}")
        else:
            rng = f" range ~{f.get('min')}..{f.get('max')}" if f.get("max", 0) > f.get("min", 0) else ""
            lines.append(f"- {f['name']} (number{rng}): {f.get('description', '')}")
    lines.append(f"- {design['target']} (the decision, one of: {', '.join(design['classes'])})")
    return "\n".join(lines)


def validate_rows(design: dict, rows: list) -> list[dict]:
    out = []
    target = design["target"]
    classes = set(design["classes"])
    for r in rows if isinstance(rows, list) else []:
        if not isinstance(r, dict):
            continue
        label = to_label(r.get(target))
        if label not in classes:
            continue
        row: dict[str, Any] = {}
        ok = True
        for f in design["features"]:
            v = r.get(f["name"])
            if f["type"] == CATEGORICAL:
                v = to_label(v)
                if v not in f["values"]:
                    ok = False
                    break
            else:
                v = to_number(v)
                if v is None:
                    ok = False
                    break
            row[f["name"]] = v
        if ok:
            row[target] = label
            out.append(row)
    return out


async def generate_rows(
    provider: JSONProvider,
    design: dict,
    description: str,
    n_rows: int,
    on_progress: Callable[[int, int], Awaitable[None]] | None = None,
    concurrency: int = 3,
) -> list[dict]:
    n_rows = max(1, min(int(n_rows), MAX_ROWS))
    n_batches = (n_rows + BATCH_SIZE - 1) // BATCH_SIZE
    schema = rows_schema(design)
    columns = _describe_columns(design)
    sem = asyncio.Semaphore(concurrency)
    results: list[list[dict]] = [[] for _ in range(n_batches)]
    done = 0
    lock = asyncio.Lock()

    async def run(i: int) -> None:
        nonlocal done
        size = min(BATCH_SIZE, n_rows - i * BATCH_SIZE)
        focus = design["classes"][i % len(design["classes"])]
        prompt = (
            f"Decision problem: {description}\n\n"
            f"Expert decision logic (follow it, with ~5% realistic exceptions/noise):\n{design['decision_logic']}\n\n"
            f"Columns:\n{columns}\n\n"
            f"Generate exactly {size} rows in `rows`. This is batch {i + 1} of {n_batches}: make these rows different "
            f"from typical examples, spread values across their full ranges, include cases close to the decision "
            f"boundaries, and include slightly more '{focus}' examples than usual while still covering every class."
        )
        async with sem:
            data = await provider.complete_json(SYSTEM, prompt, schema)
        rows = validate_rows(design, data.get("rows", []) if isinstance(data, dict) else [])
        results[i] = rows
        async with lock:
            done += len(rows)
            if on_progress:
                await on_progress(done, n_rows)

    outcomes = await asyncio.gather(*(run(i) for i in range(n_batches)), return_exceptions=True)
    errors = [e for e in outcomes if isinstance(e, BaseException)]
    rows = [r for batch in results for r in batch]
    if not rows:
        if errors:
            raise errors[0]
        raise GenerationError("the model did not return any valid rows")
    return rows[:n_rows]

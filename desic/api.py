"""HTTP + WebSocket API and dashboard hosting."""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from . import __version__
from .core.schema import CATEGORICAL, NUMERIC, Feature, Schema
from .datasets import DatasetError, parse_dataset
from .generation import GenerationError, ProviderConfig, design_schema, generate_rows, make_provider
from .service import Conflict, Desic, NotFound

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


# ------------------------------------------------------------------ requests
class FeatureIn(BaseModel):
    name: str
    type: str = NUMERIC
    values: list[str] = []
    description: str = ""


class CreateModelIn(BaseModel):
    name: str
    target: str
    features: list[FeatureIn]
    classes: list[str] = []
    kind: str = "tree"
    params: dict[str, Any] = {}
    description: str = ""


class DecideIn(BaseModel):
    features: dict[str, Any]
    record: bool = True


class FeedbackIn(BaseModel):
    decision_id: str
    label: Any


class LearnIn(BaseModel):
    rows: list[dict[str, Any]] = Field(..., max_length=100_000)


class RulesIn(BaseModel):
    rules: list[dict[str, Any]]


class TrainIn(BaseModel):
    model: str
    target: str | None = None
    features: list[str] | None = None
    kind: str = "tree"
    params: dict[str, Any] = {}
    holdout: float = 0.2
    shuffle: bool = True
    passes: int = 1


class ProviderIn(BaseModel):
    provider: str = "anthropic"
    api_key: str = ""
    model: str = ""
    base_url: str = ""

    def config(self) -> ProviderConfig:
        return ProviderConfig(self.provider, self.api_key.strip(), self.model.strip(), self.base_url.strip())


class DesignIn(ProviderIn):
    description: str = Field(..., min_length=5, max_length=4000)
    n_features: int = Field(6, ge=2, le=20)


class GenerateIn(ProviderIn):
    description: str = Field(..., min_length=5, max_length=4000)
    design: dict[str, Any]
    n_rows: int = Field(200, ge=10, le=5000)
    name: str = ""
    train_model: str = ""  # when set, train (or create) this model on the result
    kind: str = "tree"


# ----------------------------------------------------------------------- app
def create_app(data_dir: str | None = None) -> FastAPI:
    data_dir = data_dir or os.environ.get("DESIC_DATA", "data")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.desic = Desic(data_dir)
        app.state.tasks = set()
        yield
        app.state.desic.persist_all()
        app.state.desic.storage.close()

    app = FastAPI(title="Desic", version=__version__, lifespan=lifespan)

    def svc() -> Desic:
        return app.state.desic

    def spawn(coro) -> None:
        task = asyncio.create_task(coro)
        app.state.tasks.add(task)
        task.add_done_callback(app.state.tasks.discard)

    @app.exception_handler(NotFound)
    async def _nf(_, e: NotFound):
        return JSONResponse({"detail": str(e.args[0])}, status_code=404)

    @app.exception_handler(Conflict)
    async def _cf(_, e: Conflict):
        return JSONResponse({"detail": str(e)}, status_code=409)

    @app.exception_handler(ValueError)
    async def _ve(_, e: ValueError):
        return JSONResponse({"detail": str(e)}, status_code=400)

    @app.exception_handler(GenerationError)
    async def _ge(_, e: GenerationError):
        return JSONResponse({"detail": str(e)}, status_code=502)

    # ---------------------------------------------------------------- basics
    @app.get("/api/health")
    async def health():
        return {"status": "ok", "version": __version__, "models": len(svc().runtimes)}

    # ---------------------------------------------------------------- models
    @app.get("/api/models")
    async def list_models():
        return svc().list_models()

    @app.post("/api/models", status_code=201)
    async def create_model(body: CreateModelIn):
        for f in body.features:
            if f.type not in (NUMERIC, CATEGORICAL):
                raise ValueError(f"feature {f.name!r}: type must be 'numeric' or 'categorical'")
        schema = Schema([Feature(f.name, f.type, list(f.values), f.description) for f in body.features],
                        body.target, list(body.classes))
        rt = svc().create_model(body.name, schema, body.kind, body.params, body.description)
        return svc().model_detail(rt.name)

    @app.get("/api/models/{name}")
    async def get_model(name: str):
        return svc().model_detail(name)

    @app.delete("/api/models/{name}", status_code=204)
    async def delete_model(name: str):
        svc().delete_model(name)

    @app.post("/api/models/{name}/reset")
    async def reset_model(name: str):
        svc().reset_model(name)
        return svc().model_detail(name)

    @app.get("/api/models/{name}/tree")
    async def get_tree(name: str, member: int | None = None):
        return svc().tree(name, member)

    @app.get("/api/models/{name}/learned-rules")
    async def learned_rules(name: str, limit: int = 100):
        return svc().learned_rules(name, limit)

    @app.put("/api/models/{name}/rules")
    async def put_rules(name: str, body: RulesIn):
        return svc().set_rules(name, body.rules)

    @app.post("/api/models/{name}/decide")
    async def decide(name: str, body: DecideIn):
        return svc().decide(name, body.features, body.record)

    @app.post("/api/models/{name}/feedback")
    async def feedback(name: str, body: FeedbackIn):
        return svc().feedback(name, body.decision_id, body.label)

    @app.post("/api/models/{name}/learn")
    async def learn(name: str, body: LearnIn):
        return await asyncio.to_thread(svc().learn, name, body.rows)

    @app.get("/api/models/{name}/decisions")
    async def decisions(name: str, limit: int = 50, pending: bool = False, uncertain_first: bool = False):
        return svc().decisions(name, min(limit, 500), pending, uncertain_first)

    # -------------------------------------------------------------- datasets
    @app.get("/api/datasets")
    async def list_datasets():
        return svc().storage.list_datasets()

    @app.post("/api/datasets", status_code=201)
    async def upload_dataset(file: UploadFile = File(...), name: str = Form("")):
        content = await file.read(MAX_UPLOAD_BYTES + 1)
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "file is larger than 50 MB")
        try:
            columns, rows = parse_dataset(file.filename or "upload", content)
        except DatasetError as e:
            raise HTTPException(400, str(e)) from None
        ds = svc().add_dataset(name or file.filename or "dataset", "upload", columns, rows)
        return svc().dataset_profile(ds["id"])

    @app.get("/api/datasets/{ds_id}")
    async def get_dataset(ds_id: str, preview: int = 20):
        return svc().dataset_profile(ds_id, min(preview, 500))

    @app.delete("/api/datasets/{ds_id}", status_code=204)
    async def delete_dataset(ds_id: str):
        if svc().storage.get_dataset(ds_id) is None:
            raise NotFound(f"dataset {ds_id!r} not found")
        svc().storage.delete_dataset(ds_id)

    @app.post("/api/datasets/{ds_id}/train", status_code=202)
    async def train(ds_id: str, body: TrainIn):
        if svc().storage.get_dataset(ds_id) is None:
            raise NotFound(f"dataset {ds_id!r} not found")
        job = svc().new_job("train", f"training {body.model}")
        spawn(svc().train_from_dataset(job, ds_id, body.model, body.target, body.features, body.kind,
                                       body.params, body.holdout, body.shuffle, body.passes))
        return job.to_dict()

    # ------------------------------------------------------------ generation
    @app.post("/api/generate/design")
    async def generate_design(body: DesignIn):
        provider = make_provider(body.config())
        return await design_schema(provider, body.description, body.n_features)

    @app.post("/api/generate/dataset", status_code=202)
    async def generate_dataset(body: GenerateIn):
        provider = make_provider(body.config())  # validate before starting the job
        desic = svc()
        job = desic.new_job("generate", f"generating {body.n_rows} rows")

        async def run():
            try:
                from .generation import normalize_design

                design = normalize_design(body.design)

                async def progress(done: int, total: int) -> None:
                    desic.update_job(job, progress=round(min(done / total, 1.0) * 0.95, 4),
                                     message=f"{done}/{total} rows generated")

                rows = await generate_rows(provider, design, body.description, body.n_rows, progress)
                columns = [f["name"] for f in design["features"]] + [design["target"]]
                ds = desic.add_dataset(body.name or design["name"], "generated", columns, rows,
                                       {"description": body.description, "design": design})
                result: dict[str, Any] = {"dataset": ds["id"], "rows": len(rows)}
                if body.train_model:
                    train_job = desic.new_job("train", f"training {body.train_model}")
                    result["train_job"] = train_job.id
                    spawn(desic.train_from_dataset(train_job, ds["id"], body.train_model, design["target"],
                                                   None, body.kind))
                desic.update_job(job, status="done", progress=1.0, result=result,
                                 message=f"{len(rows)} rows generated")
            except Exception as e:
                desic.update_job(job, status="error", error=str(e) or type(e).__name__)

        spawn(run())
        return job.to_dict()

    # ------------------------------------------------------------------ jobs
    @app.get("/api/jobs")
    async def list_jobs():
        return sorted((j.to_dict() for j in svc().jobs.values()), key=lambda j: -j["created_at"])[:50]

    @app.get("/api/jobs/{job_id}")
    async def get_job(job_id: str):
        job = svc().jobs.get(job_id)
        if job is None:
            raise NotFound(f"job {job_id!r} not found")
        return job.to_dict()

    # ------------------------------------------------------------- realtime
    @app.websocket("/ws")
    async def ws(websocket: WebSocket):
        await websocket.accept()
        bus = svc().bus
        q = bus.subscribe()

        try:
            async with anyio.create_task_group() as tg:

                async def pump() -> None:
                    await websocket.send_json({"type": "hello", "version": __version__})
                    while True:
                        await websocket.send_json(await q.get())

                async def drain() -> None:  # returns when the client goes away
                    try:
                        while True:
                            await websocket.receive_text()
                    except WebSocketDisconnect:
                        pass
                    tg.cancel_scope.cancel()

                tg.start_soon(pump)
                tg.start_soon(drain)
        except Exception:  # client vanished mid-send (anyio may wrap it in an ExceptionGroup)
            pass
        finally:
            bus.unsubscribe(q)

    # ------------------------------------------------------------- dashboard
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC / "index.html")

    return app


app = create_app()

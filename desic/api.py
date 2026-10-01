"""HTTP + WebSocket API and dashboard hosting.

The decision endpoint follows the Jev request shape so Desic can sit where a
System One model sits:

    POST /v1/decide
    {"state": "...text or JSON...",
     "questions": {"department": {"type": "choice", "instructions": "Which team?",
                                  "criteria": {"billing": "payments, refunds", "technical": "bugs"}},
                   "urgent": {"type": "noul", "instructions": "The customer needs help today."}}}
"""

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
from .core.task import CHOICE, QuestionSpec, SpecError
from .datasets import DatasetError, parse_dataset
from .llm import PROVIDERS, LLMError, ProviderConfig
from .neural.runtime import NeuralError
from .service import Conflict, Desic, NotFound

STATIC = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 50 * 1024 * 1024


# ------------------------------------------------------------------ requests
class DecideIn(BaseModel):
    state: Any
    questions: dict[str, Any] = Field(..., description="question name -> {type, instructions, criteria}; {} reuses a known question")
    explain: bool = False
    record: bool = True
    escalate: str = Field("auto", description="auto (per-question teacher mode) | never | always")
    abstain_threshold: float | None = Field(None, ge=0, le=1)


class FeedbackIn(BaseModel):
    decision_id: str
    answers: dict[str, Any] = Field(..., description="question name -> correct answer")
    source: str = "human"


class CreateTaskIn(BaseModel):
    name: str
    type: str = CHOICE
    instructions: str = ""
    criteria: Any = None
    settings: dict[str, Any] = {}


class PatchTaskIn(BaseModel):
    instructions: str | None = None
    descriptions: dict[str, str] | None = None
    add_options: list[str] | None = None
    settings: dict[str, Any] | None = None


class ExamplesIn(BaseModel):
    examples: list[dict[str, Any]] = Field(..., max_length=100_000)


class RulesIn(BaseModel):
    rules: list[dict[str, Any]]


class RollbackIn(BaseModel):
    version: int


class RebuildIn(BaseModel):
    exclude_sources: list[str] = []


class SnapshotIn(BaseModel):
    note: str = ""


class RetractRecentIn(BaseModel):
    n: int = Field(1, ge=1, le=1000)
    sources: list[str] | None = None


class TeacherIn(BaseModel):
    provider: str = "anthropic"
    api_key: str = ""
    model: str = ""
    base_url: str = ""


class TrainIn(BaseModel):
    task: str
    answer_column: str
    state_columns: list[str] | None = None
    state_mode: str = "auto"  # auto | text | json
    type: str = CHOICE
    levels: list[str] | None = None
    instructions: str = ""
    holdout: float = 0.2


class DistillIn(BaseModel):
    task: str
    state_columns: list[str] | None = None
    state_mode: str = "auto"
    limit: int = Field(200, ge=1, le=5000)


class DesignIn(BaseModel):
    description: str = Field(..., min_length=5, max_length=4000)


class GenerateIn(BaseModel):
    description: str = Field(..., min_length=5, max_length=4000)
    design: dict[str, Any]
    n: int = Field(200, ge=5, le=5000)
    name: str = ""
    train: bool = True


# ----------------------------------------------------------------------- app
def create_app(data_dir: str | None = None, teacher: ProviderConfig | None = None) -> FastAPI:
    data_dir = data_dir or os.environ.get("DESIC_DATA", "data")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.desic = Desic(data_dir, teacher=teacher)
        yield
        app.state.desic.persist_all()
        app.state.desic.storage.close()

    app = FastAPI(title="Desic", version=__version__, lifespan=lifespan,
                  description="Self-learning typed-decision engine (choice / score / noul) with calibrated probabilities.")

    def svc() -> Desic:
        return app.state.desic

    @app.exception_handler(NotFound)
    async def _nf(_, e: NotFound):
        return JSONResponse({"detail": str(e.args[0])}, status_code=404)

    @app.exception_handler(Conflict)
    async def _cf(_, e: Conflict):
        return JSONResponse({"detail": str(e)}, status_code=409)

    @app.exception_handler(SpecError)
    async def _se(_, e: SpecError):
        return JSONResponse({"detail": str(e)}, status_code=422)

    @app.exception_handler(ValueError)
    async def _ve(_, e: ValueError):
        return JSONResponse({"detail": str(e)}, status_code=400)

    @app.exception_handler(LLMError)
    async def _le(_, e: LLMError):
        return JSONResponse({"detail": str(e)}, status_code=502)

    @app.exception_handler(NeuralError)
    async def _ne(_, e: NeuralError):
        return JSONResponse({"detail": str(e)}, status_code=503)

    # ------------------------------------------------------------- decisions
    @app.get("/v1/health")
    async def health():
        return {"status": "ok", "version": __version__, "questions": len(svc().tasks),
                "teacher": svc().teacher_cfg is not None, "neural": svc().neural.active_id}

    @app.post("/v1/decide")
    async def decide(body: DecideIn):
        return await svc().decide(body.state, body.questions, body.record, body.explain, body.escalate, body.abstain_threshold)

    @app.post("/v1/feedback")
    async def feedback(body: FeedbackIn):
        return svc().feedback(body.decision_id, body.answers, body.source)

    @app.get("/v1/decisions/{decision_id}")
    async def get_decision(decision_id: str):
        d = svc().storage.get_decision(decision_id)
        if d is None:
            raise NotFound(f"decision {decision_id!r} not found")
        return d

    # ------------------------------------------------------------- questions
    @app.get("/v1/questions")
    async def list_questions():
        return svc().list_tasks()

    @app.post("/v1/questions", status_code=201)
    async def create_question(body: CreateTaskIn):
        spec = QuestionSpec.parse(body.name, {"type": body.type, "instructions": body.instructions, "criteria": body.criteria})
        svc().create_task(spec)
        if body.settings:
            svc().update_task(spec.name, {"settings": body.settings})
        return svc().task_detail(spec.name)

    @app.get("/v1/questions/{name}")
    async def get_question(name: str):
        return svc().task_detail(name)

    @app.patch("/v1/questions/{name}")
    async def patch_question(name: str, body: PatchTaskIn):
        return svc().update_task(name, body.model_dump(exclude_none=True))

    @app.delete("/v1/questions/{name}", status_code=204)
    async def delete_question(name: str):
        svc().delete_task(name)

    @app.post("/v1/questions/{name}/reset")
    async def reset_question(name: str):
        svc().reset_task(name)
        return svc().task_detail(name)

    @app.post("/v1/questions/{name}/learn")
    async def learn(name: str, body: ExamplesIn):
        return await anyio.to_thread.run_sync(svc().learn, name, body.examples)

    @app.get("/v1/questions/{name}/decisions")
    async def question_decisions(name: str, limit: int = 50, pending: bool = False, uncertain_first: bool = False):
        return svc().answers(name, min(limit, 500), pending, uncertain_first)

    @app.put("/v1/questions/{name}/rules")
    async def put_rules(name: str, body: RulesIn):
        return svc().set_rules(name, body.rules)

    @app.get("/v1/questions/{name}/feedback")
    async def question_feedback(name: str, limit: int = 100, offset: int = 0):
        svc().get(name)
        return svc().storage.list_events(name, min(limit, 500), offset)

    # Retracting applies at once: a label still on probation is dropped from the patch layer,
    # an older one is replayed out from the nearest checkpoint (in a thread: that can take seconds).
    @app.post("/v1/feedback/{event_id}/retract")
    async def retract(event_id: int):
        return await asyncio.to_thread(svc().retract, event_id, True)

    @app.post("/v1/feedback/{event_id}/restore")
    async def restore(event_id: int):
        return await asyncio.to_thread(svc().retract, event_id, False)

    @app.post("/v1/questions/{name}/retract-recent")
    async def retract_recent(name: str, body: RetractRecentIn):
        return await asyncio.to_thread(svc().retract_recent, name, body.n, body.sources)

    @app.get("/v1/questions/{name}/snapshots")
    async def snapshots(name: str):
        svc().get(name)
        return svc().storage.list_snapshots(name)

    @app.post("/v1/questions/{name}/snapshots", status_code=201)
    async def snapshot(name: str, body: SnapshotIn):
        return svc().snapshot(name, body.note)

    @app.post("/v1/questions/{name}/rollback")
    async def rollback(name: str, body: RollbackIn):
        return svc().rollback(name, body.version)

    @app.post("/v1/questions/{name}/rebuild", status_code=202)
    async def rebuild(name: str, body: RebuildIn):
        svc().get(name)
        job = svc().new_job("rebuild", f"rebuilding {name} from the feedback log")
        svc().spawn(svc().rebuild(job, name, body.exclude_sources))
        return job.to_dict()

    # --------------------------------------------------------------- teacher
    @app.get("/v1/teacher")
    async def get_teacher():
        return svc().teacher_public()

    @app.put("/v1/teacher")
    async def put_teacher(body: TeacherIn):
        if body.provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {', '.join(PROVIDERS)}")
        return svc().set_teacher(ProviderConfig(body.provider, body.api_key.strip(), body.model.strip(), body.base_url.strip()))

    @app.delete("/v1/teacher")
    async def delete_teacher():
        return svc().set_teacher(None)

    @app.post("/v1/teacher/test")
    async def test_teacher():
        return await svc().test_teacher()

    # ---------------------------------------------------------------- neural
    @app.get("/v1/neural")
    async def neural_status():
        return svc().neural.status()

    @app.patch("/v1/neural/config")
    async def neural_config(body: dict[str, Any]):
        svc().neural.set_config(body)
        return svc().neural.status()

    @app.post("/v1/neural/train", status_code=202)
    async def neural_train():
        return svc().start_neural_training().to_dict()

    @app.post("/v1/neural/cancel")
    async def neural_cancel():
        svc().neural.cancel = True
        return {"cancelling": bool(svc().neural.training)}

    @app.post("/v1/neural/checkpoints/{ck_id}/{action}")
    async def neural_checkpoint(ck_id: str, action: str):
        return await anyio.to_thread.run_sync(svc().neural_action, ck_id, action)

    # -------------------------------------------------------------- datasets
    @app.get("/v1/datasets")
    async def list_datasets():
        return svc().storage.list_datasets()

    @app.post("/v1/datasets", status_code=201)
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

    @app.get("/v1/datasets/{ds_id}")
    async def get_dataset(ds_id: str, preview: int = 20):
        return svc().dataset_profile(ds_id, min(preview, 500))

    @app.delete("/v1/datasets/{ds_id}", status_code=204)
    async def delete_dataset(ds_id: str):
        if svc().storage.get_dataset(ds_id) is None:
            raise NotFound(f"dataset {ds_id!r} not found")
        svc().storage.delete_dataset(ds_id)

    @app.post("/v1/datasets/{ds_id}/train", status_code=202)
    async def train(ds_id: str, body: TrainIn):
        if svc().storage.get_dataset(ds_id) is None:
            raise NotFound(f"dataset {ds_id!r} not found")
        job = svc().new_job("train", f"training {body.task}")
        svc().spawn(svc().train_from_dataset(job, ds_id, body.task, body.answer_column, body.state_columns,
                                             body.state_mode, body.type, body.levels, body.instructions, body.holdout))
        return job.to_dict()

    @app.post("/v1/datasets/{ds_id}/distill", status_code=202)
    async def distill(ds_id: str, body: DistillIn):
        if svc().storage.get_dataset(ds_id) is None:
            raise NotFound(f"dataset {ds_id!r} not found")
        svc().get(body.task)
        svc().provider()  # fail fast without a teacher
        job = svc().new_job("distill", f"teacher labelling for {body.task}")
        svc().spawn(svc().distill(job, ds_id, body.task, body.state_columns, body.state_mode, body.limit))
        return job.to_dict()

    # ------------------------------------------------------------ generation
    @app.post("/v1/generate/design")
    async def generate_design(body: DesignIn):
        return await svc().design(body.description)

    @app.post("/v1/generate/examples", status_code=202)
    async def generate(body: GenerateIn):
        svc().provider()
        job = svc().new_job("generate", f"writing {body.n} examples")
        svc().spawn(svc().generate(job, body.description, body.design, body.n, body.name, body.train))
        return job.to_dict()

    # ------------------------------------------------------------------ jobs
    @app.get("/v1/jobs")
    async def list_jobs():
        return sorted((j.to_dict() for j in svc().jobs.values()), key=lambda j: -j["created_at"])[:50]

    @app.get("/v1/jobs/{job_id}")
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

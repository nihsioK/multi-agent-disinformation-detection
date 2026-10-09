import json
import os
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, PlainTextResponse

from app import config
from app.llm import LLM, MODEL_SUGGESTIONS, ConfigError
from app.messages import Claim, ClaimInput, new_id
from app.runner import TRACING, run_check
from app.store import Store
from app.tools import SpeakerHistory, get_model

app = FastAPI(title="Claim Check", version="0.1.0")
store = Store(config.DB_PATH)
speakers = SpeakerHistory(config.LIAR_TRAIN_TSV)
STATIC = Path(__file__).parent / "static"


def load_model():
    return get_model(config.HYBRID_CHECKPOINT)


class RunIn(ClaimInput):
    model: str | None = None


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/config")
def api_config() -> dict:
    return {
        "provider": LLM.provider,
        "model": config.LLM_MODEL,
        "suggestions": list(MODEL_SUGGESTIONS),
        "configured": bool(os.environ.get("OPENROUTER_API_KEY")),
        "tracing": TRACING,
    }


@app.post("/api/runs", status_code=202)
def create_run(body: RunIn, background: BackgroundTasks) -> dict:
    try:
        llm = LLM(body.model or None)
    except ConfigError as error:
        raise HTTPException(400, str(error)) from error
    claim = Claim(
        run_id=new_id("run"),
        sender="user",
        recipient="runner",
        statement=body.statement,
        speaker=body.speaker,
        context=body.context,
    )
    store.create_run(claim.run_id, claim.model_dump(), llm.provider, llm.model)
    background.add_task(run_check, store, claim, llm, speakers, load_model, "web")
    return {"run_id": claim.run_id}


@app.get("/api/runs")
def list_runs(limit: int = 20) -> list[dict]:
    return store.list_runs(limit)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, "run not found")
    return {"run": run, "events": store.events(run_id), "workload": store.workload(run_id)}


@app.get("/api/runs/{run_id}/events.jsonl")
def events_jsonl(run_id: str) -> PlainTextResponse:
    lines = [json.dumps(event, ensure_ascii=False) for event in store.events(run_id)]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="application/x-ndjson")

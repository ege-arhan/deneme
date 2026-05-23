from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from song_miner.jobs import JobManager
from song_miner.pipeline import SongPipeline

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

app = FastAPI(title="Song Miner UI", version="1.0.0")
pipeline = SongPipeline(out_dir="data")
jobs = JobManager()


class AnalyzeRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Şarkı adı")
    artist: str | None = Field(default=None, description="Sanatçı")
    audio_seconds: int = Field(default=120, ge=10, le=600)
    no_download: bool = Field(default=False)


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=500, detail="UI dosyası bulunamadı")
    return HTMLResponse(index_file.read_text(encoding="utf-8"))


@app.post("/api/analyze")
def analyze(req: AnalyzeRequest) -> dict[str, Any]:
    result = pipeline.run(
        query=req.query,
        artist=req.artist,
        max_seconds=req.audio_seconds,
        skip_download=req.no_download,
    )
    return {"ok": True, "result": result}






@app.post("/api/analyze_async")
def analyze_async(req: AnalyzeRequest) -> dict[str, Any]:
    job = jobs.create()
    def _progress(stage: str, progress: int) -> None:
        jobs.update(job.id, stage=stage, progress=progress)

    jobs.run_background(
        job.id,
        pipeline.run_with_hooks,
        query=req.query,
        artist=req.artist,
        max_seconds=req.audio_seconds,
        skip_download=req.no_download,
        progress_cb=_progress,
    )
    return {"ok": True, "job_id": job.id, "status": job.status}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return job.__dict__


@app.get("/api/jobs")
def list_jobs(limit: int = 20) -> dict[str, Any]:
    return {"items": [j.__dict__ for j in jobs.list_recent(limit=limit)]}


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    ok = jobs.cancel(job_id)
    if not ok:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    return {"ok": True, "job_id": job_id, "status": "cancelled"}

@app.get("/api/agent/status")
def agent_status() -> dict[str, str]:
    return {
        "model": pipeline.agent.model,
        "ollama_base_url": pipeline.agent.base_url,
    }

@app.get("/api/agent/actions")
def agent_actions(limit: int = 20) -> dict[str, Any]:
    path = Path("data") / "agent_actions.jsonl"
    if not path.exists():
        return {"items": []}
    lines = path.read_text(encoding="utf-8").splitlines()[-max(1, min(limit, 200)): ]
    items=[]
    for line in reversed(lines):
        try: items.append(json.loads(line))
        except Exception: continue
    return {"items": items}


@app.post("/api/review_queue/resolve")
def resolve_review(record_id: str, action: str = "approve") -> dict[str, Any]:
    path = Path("data") / "review_queue.jsonl"
    if not path.exists():
        raise HTTPException(status_code=404, detail="review queue not found")
    lines = path.read_text(encoding="utf-8").splitlines()
    kept=[]
    target=None
    for line in lines:
        try:
            rec=json.loads(line)
        except Exception:
            continue
        if rec.get("record_id") == record_id and target is None:
            target=rec
            continue
        kept.append(rec)
    if target is None:
        raise HTTPException(status_code=404, detail="record not found")
    with path.open("w", encoding="utf-8") as f:
        for rec in kept:
            f.write(json.dumps(rec, ensure_ascii=False)+"\n")
    if action == "approve":
        songs=Path("data")/"songs.jsonl"
        with songs.open("a", encoding="utf-8") as f:
            f.write(json.dumps(target, ensure_ascii=False)+"\n")
    elif action == "retry":
        rerun = pipeline.run(query=target.get("query",""), artist=target.get("artist"), max_seconds=120, skip_download=True)
        return {"ok": True, "action": action, "result": rerun}
    return {"ok": True, "action": action, "record_id": record_id}


@app.get("/api/review_queue")
def review_queue(limit: int = 20) -> dict[str, Any]:
    path = Path("data") / "review_queue.jsonl"
    if not path.exists():
        return {"items": []}
    lines = path.read_text(encoding="utf-8").splitlines()[-max(1, min(limit, 100)): ]
    items=[]
    for line in reversed(lines):
        try: items.append(json.loads(line))
        except Exception: continue
    return {"items": items}


@app.get("/api/history")
def history(limit: int = 20) -> dict[str, Any]:
    path = Path("data") / "songs.jsonl"
    if not path.exists():
        return {"items": []}
    lines = path.read_text(encoding="utf-8").splitlines()[-max(1, min(limit, 100)):]
    items = []
    for line in reversed(lines):
        try:
            items.append(json.loads(line))
        except Exception:
            continue
    return {"items": items}


@app.get("/api/agent/check")
def agent_check() -> dict[str, Any]:
    return pipeline.agent.check_health()


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

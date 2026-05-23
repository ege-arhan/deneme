from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class JobState:
    id: str
    status: str = "queued"
    progress: int = 0
    stage: str = "queued"
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at_utc: str | None = None
    finished_at_utc: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None


class JobManager:
    def __init__(self, data_dir: str = "data") -> None:
        self._jobs: dict[str, JobState] = {}
        self._lock = threading.Lock()
        self._jobs_path = Path(data_dir) / "jobs.jsonl"
        self._jobs_path.parent.mkdir(parents=True, exist_ok=True)
        self._load_jobs()

    def create(self) -> JobState:
        job = JobState(id=uuid.uuid4().hex)
        with self._lock:
            self._jobs[job.id] = job
            self._persist_locked()
        return job

    def get(self, job_id: str) -> JobState | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list_recent(self, limit: int = 20) -> list[JobState]:
        with self._lock:
            items = list(self._jobs.values())
        return sorted(items, key=lambda x: x.created_at_utc, reverse=True)[:limit]

    def update(self, job_id: str, *, stage: str | None = None, progress: int | None = None) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return
            if stage is not None:
                job.stage = stage
            if progress is not None:
                job.progress = max(0, min(100, int(progress)))
            self._persist_locked()

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job or job.status in {"done", "failed", "cancelled"}:
                return False
            job.status = "cancelled"
            job.stage = "cancelled"
            self._persist_locked()
            return True

    def run_background(self, job_id: str, fn, *args, **kwargs) -> None:
        def _runner():
            job = self.get(job_id)
            if not job:
                return
            job.status = "running"
            job.stage = "running"
            job.progress = 10
            job.started_at_utc = datetime.now(timezone.utc).isoformat()
            with self._lock:
                self._persist_locked()
            try:
                if job.status == "cancelled":
                    return
                job.progress = 40
                job.stage = "processing"
                job.result = fn(*args, **kwargs)
                if job.status != "cancelled":
                    job.status = "done"
                    job.stage = "done"
                    job.progress = 100
            except Exception as exc:
                job.status = "failed"
                job.stage = "failed"
                job.error = str(exc)
            finally:
                job.finished_at_utc = datetime.now(timezone.utc).isoformat()
                with self._lock:
                    self._persist_locked()

        t = threading.Thread(target=_runner, daemon=True)
        t.start()

    def _persist_locked(self) -> None:
        with self._jobs_path.open("w", encoding="utf-8") as f:
            for j in self._jobs.values():
                f.write(json.dumps(j.__dict__, ensure_ascii=False) + "\n")

    def _load_jobs(self) -> None:
        if not self._jobs_path.exists():
            return
        for line in self._jobs_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                self._jobs[item["id"]] = JobState(**item)
            except Exception:
                continue

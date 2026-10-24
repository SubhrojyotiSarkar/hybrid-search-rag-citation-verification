"""Background evaluation jobs, so the API never waits on a long run.

One job runs at a time, in a daemon thread inside the backend process: two concurrent
answer runs would only split the provider's per-minute quota. Job state is in memory; the
finished run itself is saved to EVALUATION_DIR/results like a CLI run, so a backend
restart loses only the progress display of an unfinished job (and the job itself).
"""

import logging
import threading
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

logger = logging.getLogger(__name__)

MAX_REMEMBERED_JOBS = 20


class JobConflictError(Exception):
    """Another evaluation job is still running."""


class EvaluationJob(BaseModel):
    job_id: str
    kind: str  # "retrieval" or "answers"
    status: str = "running"  # running | succeeded | failed
    done: int = 0
    total: int = 0
    message: str = ""  # latest progress note
    run_id: str | None = None
    error: str | None = None
    options: dict = {}
    started_at: str
    finished_at: str | None = None


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, EvaluationJob] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, options: dict, work: Callable[[Callable], dict], save: Callable[[dict], Path]) -> EvaluationJob:
        """Run `work(progress)` in the background and save its run record with `save`.

        `progress(done, total, message="")` updates the job; raises JobConflictError if a
        job is already running.
        """
        with self._lock:
            if any(job.status == "running" for job in self._jobs.values()):
                raise JobConflictError("An evaluation job is already running.")
            job = EvaluationJob(job_id=uuid.uuid4().hex[:12], kind=kind, options=options, started_at=_now())
            self._jobs[job.job_id] = job
            self._forget_old()

        def progress(done: int, total: int, message: str = "") -> None:
            with self._lock:
                job.done, job.total, job.message = done, total, message

        def run() -> None:
            try:
                record = work(progress)
                save(record)
                with self._lock:
                    job.run_id, job.status = record["run_id"], "succeeded"
            except Exception as exc:
                logger.exception("Evaluation job %s failed", job.job_id)
                with self._lock:
                    job.status, job.error = "failed", f"{type(exc).__name__}: {exc}"[:500]
            finally:
                with self._lock:
                    job.finished_at = _now()

        threading.Thread(target=run, name=f"evaluation-{job.job_id}", daemon=True).start()
        return job.model_copy()

    def get(self, job_id: str) -> EvaluationJob | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.model_copy() if job else None

    def recent(self) -> list[EvaluationJob]:
        with self._lock:
            return [job.model_copy() for job in sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)]

    def _forget_old(self) -> None:
        finished = sorted((j for j in self._jobs.values() if j.status != "running"), key=lambda j: j.started_at)
        for job in finished[: max(0, len(self._jobs) - MAX_REMEMBERED_JOBS)]:
            del self._jobs[job.job_id]

from threading import Lock
from typing import Any


class JobStore:
    """Small storage boundary that can later be replaced by Redis or a database."""

    def __init__(self) -> None:
        self._jobs: dict[str, dict[str, Any]] = {}
        self._lock = Lock()

    def create(self, job_id: str) -> None:
        with self._lock:
            self._jobs[job_id] = {
                "status": "processing",
                "progress": 0,
                "current_frame": 0,
                "total_frames": 0,
                "processing_fps": None,
                "eta_seconds": None,
            }

    def update(self, job_id: str, **values: Any) -> None:
        with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].update(values)

    def get(self, job_id: str) -> dict[str, Any] | None:
        with self._lock:
            job = self._jobs.get(job_id)
            return job.copy() if job else None


job_store = JobStore()

from typing import Any, Literal

from pydantic import BaseModel, Field


class AlertEvent(BaseModel):
    start_seconds: float
    end_seconds: float


class VideoSummary(BaseModel):
    fps: float
    total_frames: int
    duration_seconds: float
    unique_normal_person_count: int
    unique_potentially_dangerous_person_count: int
    alert_events: list[AlertEvent] = Field(default_factory=list)


class JobStatus(BaseModel):
    status: Literal["processing", "done", "error"]
    progress: int = 0
    current_frame: int = 0
    total_frames: int = 0
    processing_fps: float | None = None
    eta_seconds: float | None = None
    output_video: str | None = None
    summary: VideoSummary | None = None
    error: str | None = None


class UploadResponse(BaseModel):
    job_id: str

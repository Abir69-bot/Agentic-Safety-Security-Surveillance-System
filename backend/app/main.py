from __future__ import annotations

import mimetypes
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from .config import settings
from .email_alerts import send_danger_alert
from .job_store import job_store
from .models import JobStatus, UploadResponse
from .processor import load_model, process_video

app = FastAPI(title="Perimeter Watch API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=settings.allowed_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="perimeter-watch")
ALLOWED_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


@app.on_event("startup")
def startup() -> None:
    load_model(settings)


def _run_job(job_id: str, input_path: Path, output_path: Path) -> None:
    try:
        summary = process_video(input_path, output_path, settings, lambda telemetry: job_store.update(job_id, **telemetry))
        if summary.alert_events:
            try:
                send_danger_alert(job_id, summary, settings)
            except Exception as exc:
                print(f"Could not send dangerous-person email alert: {exc}")
        job_store.update(job_id, status="done", progress=100, current_frame=summary.total_frames, total_frames=summary.total_frames, eta_seconds=0, output_video=output_path.name, summary=summary.model_dump())
    except Exception as exc:
        job_store.update(job_id, status="error", error=str(exc))
    finally:
        input_path.unlink(missing_ok=True)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse, status_code=202)
async def upload_video(video: UploadFile = File(...)) -> UploadResponse:
    extension = Path(video.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported video format. Use mp4, mov, avi, mkv, or webm.")

    job_id = str(uuid.uuid4())
    input_path = settings.upload_dir / f"{job_id}{extension}"
    output_path = settings.output_dir / f"{job_id}.mp4"
    bytes_written = 0
    try:
        with input_path.open("wb") as destination:
            while chunk := await video.read(1024 * 1024):
                bytes_written += len(chunk)
                if bytes_written > settings.max_upload_bytes:
                    raise HTTPException(status_code=413, detail="Video exceeds the 500MB upload limit.")
                destination.write(chunk)
    except HTTPException:
        input_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        input_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Could not save upload: {exc}") from exc
    finally:
        await video.close()

    job_store.create(job_id)
    executor.submit(_run_job, job_id, input_path, output_path)
    return UploadResponse(job_id=job_id)


@app.get("/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str) -> JobStatus:
    job = job_store.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobStatus(**job)


@app.get("/outputs/{filename}")
def get_output(filename: str, range_header: str | None = Header(default=None, alias="Range")) -> StreamingResponse:
    safe_name = Path(filename).name
    path = settings.output_dir / safe_name
    if safe_name != filename or not path.is_file():
        raise HTTPException(status_code=404, detail="Output not found.")

    file_size = path.stat().st_size
    start = 0
    end = file_size - 1
    status_code = 200
    headers = {"Accept-Ranges": "bytes", "Content-Length": str(file_size), "Content-Type": mimetypes.guess_type(path.name)[0] or "video/mp4"}
    if range_header:
        try:
            unit, value = range_header.split("=", 1)
            if unit != "bytes":
                raise ValueError
            start_text, end_text = value.split("-", 1)
            start = int(start_text) if start_text else max(0, file_size - int(end_text))
            end = int(end_text) if end_text else end
            end = min(end, file_size - 1)
            if start > end or start < 0:
                raise ValueError
        except ValueError as exc:
            raise HTTPException(status_code=416, detail="Invalid byte range.") from exc
        status_code = 206
        headers["Content-Range"] = f"bytes {start}-{end}/{file_size}"
        headers["Content-Length"] = str(end - start + 1)

    def iterator():
        with path.open("rb") as source:
            source.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                chunk = source.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
                yield chunk

    return StreamingResponse(iterator(), status_code=status_code, headers=headers, media_type="video/mp4")

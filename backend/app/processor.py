from __future__ import annotations

import math
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable

import cv2
import imageio_ffmpeg
import torch
from ultralytics import YOLO

from .config import Settings
from .models import AlertEvent, VideoSummary

NORMAL_CLASS = 0
DANGEROUS_CLASS = 1
CLASS_NAMES = {0: "normal_person", 1: "potentially_dangerous_person"}
_model: YOLO | None = None
_model_lock = threading.Lock()


def load_model(settings: Settings) -> None:
    """Load model weights once when the API starts."""
    global _model
    if _model is not None:
        return
    model_path = settings.resolved_model_path
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLO model not found: {model_path}")
    started = time.perf_counter()
    try:
        _model = YOLO(str(model_path))
    except Exception as exc:
        raise RuntimeError(f"Could not load YOLO model at {model_path}: {exc}") from exc
    elapsed = time.perf_counter() - started
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    if device == "CPU":
        print("WARNING: CUDA is unavailable; YOLO inference is running on CPU and will be slower.")
    print(f"YOLO model loaded in {elapsed:.2f}s on {device}: {model_path}")
    print(f"YOLO classes: {_model.names}")


def _get_model(settings: Settings) -> YOLO:
    if _model is None:
        load_model(settings)
    if _model is None:
        raise RuntimeError("YOLO model failed to load.")
    return _model


def _merge_alert_frames(flagged_frames: list[int], fps: float, gap_seconds: float) -> list[AlertEvent]:
    if not flagged_frames or fps <= 0:
        return []

    max_gap = max(0, math.ceil(gap_seconds * fps))
    events: list[AlertEvent] = []
    start = previous = flagged_frames[0]
    for frame_number in flagged_frames[1:]:
        if frame_number - previous > max_gap + 1:
            events.append(AlertEvent(start_seconds=round(start / fps, 2), end_seconds=round(previous / fps, 2)))
            start = frame_number
        previous = frame_number
    events.append(AlertEvent(start_seconds=round(start / fps, 2), end_seconds=round(previous / fps, 2)))
    return events


def process_video(
    input_path: Path,
    output_path: Path,
    settings: Settings,
    update_progress: Callable[[dict[str, int | float | None]], None],
) -> VideoSummary:
    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError("OpenCV could not open the uploaded video.")

    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("The uploaded video has no readable frame dimensions.")

    raw_output_path = output_path.with_name(f"{output_path.stem}.raw.mp4")
    writer = cv2.VideoWriter(str(raw_output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("OpenCV could not create the annotated output video.")

    model = _get_model(settings)
    normal_ids: set[int] = set()
    dangerous_ids: set[int] = set()
    flagged_frames: list[int] = []
    frame_index = 0
    started = time.perf_counter()
    inference_seconds = 0.0
    write_seconds = 0.0
    last_detections: list[tuple[list[float], float, int, int | None]] = []
    skip = max(1, settings.frame_skip)

    try:
        while True:
            success, frame = capture.read()
            if not success:
                break

            if frame_index % skip == 0:
                inference_started = time.perf_counter()
                with _model_lock:
                    results = model.track(
                        frame,
                        tracker="bytetrack.yaml",
                        persist=True,
                        stream=True,
                        conf=settings.confidence_threshold,
                        iou=settings.iou_threshold,
                        verbose=False,
                    )
                    result = next(iter(results), None)
                inference_seconds += time.perf_counter() - inference_started
                last_detections = []
                if result is not None and result.boxes is not None:
                    boxes = result.boxes
                    xyxy = boxes.xyxy.cpu().tolist()
                    confidences = boxes.conf.cpu().tolist()
                    classes = boxes.cls.int().cpu().tolist()
                    track_ids = boxes.id.int().cpu().tolist() if boxes.id is not None else [None] * len(classes)
                    last_detections = list(zip(xyxy, confidences, classes, track_ids))
            dangerous_present = False

            for coordinates, confidence, class_id, track_id in last_detections:
                class_id = int(class_id)
                if class_id not in CLASS_NAMES:
                    continue
                if class_id == DANGEROUS_CLASS and confidence < settings.dangerous_confidence_threshold:
                    continue
                if frame_index % skip == 0 and track_id is not None:
                    (dangerous_ids if class_id == DANGEROUS_CLASS else normal_ids).add(int(track_id))
                if class_id == DANGEROUS_CLASS:
                    dangerous_present = True
                color = (0, 0, 255) if class_id == DANGEROUS_CLASS else (0, 190, 70)
                x1, y1, x2, y2 = (int(value) for value in coordinates)
                label = f"{CLASS_NAMES[class_id]} #{track_id if track_id is not None else '-'} {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

            if dangerous_present:
                flagged_frames.append(frame_index)
                cv2.rectangle(frame, (4, 4), (width - 5, height - 5), (0, 0, 255), 6)
                cv2.putText(frame, "ALERT", (24, 48), cv2.FONT_HERSHEY_DUPLEX, 1.1, (0, 0, 255), 3, cv2.LINE_AA)

            write_started = time.perf_counter()
            writer.write(frame)
            write_seconds += time.perf_counter() - write_started
            frame_index += 1
            if total_frames > 0:
                elapsed = time.perf_counter() - started
                processing_fps = frame_index / elapsed if elapsed else 0.0
                remaining = max(0, total_frames - frame_index)
                eta_seconds = remaining / processing_fps if processing_fps else None
                update_progress({
                    "progress": min(99, int(frame_index / total_frames * 100)),
                    "current_frame": frame_index,
                    "total_frames": total_frames,
                    "processing_fps": round(processing_fps, 2),
                    "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
                })
                if frame_index % 30 == 0:
                    print(f"frame {frame_index}/{total_frames} | inference {inference_seconds:.2f}s | write {write_seconds:.2f}s")
    finally:
        capture.release()
        writer.release()

    try:
        ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run(
            [
                ffmpeg_path,
                "-y",
                "-i",
                str(raw_output_path),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = exc.stderr[-1000:] if isinstance(exc, subprocess.CalledProcessError) else str(exc)
        raise RuntimeError(f"Could not encode H.264 output video: {detail}") from exc
    finally:
        raw_output_path.unlink(missing_ok=True)

    print(f"Video stages complete | inference {inference_seconds:.2f}s | write {write_seconds:.2f}s | frames {frame_index}")

    duration = frame_index / fps if fps else 0
    return VideoSummary(
        fps=round(fps, 3),
        total_frames=frame_index,
        duration_seconds=round(duration, 3),
        unique_normal_person_count=len(normal_ids),
        unique_potentially_dangerous_person_count=len(dangerous_ids),
        alert_events=_merge_alert_frames(flagged_frames, fps, settings.alert_gap_seconds),
    )

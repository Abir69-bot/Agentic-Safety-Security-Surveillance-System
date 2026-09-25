# Agentic Safety & Security Surveillance System

A real-world AI-assisted surveillance prototype built for detecting and reviewing potentially dangerous-person activity in video footage.

This repository combines a Python backend, a YOLO11-based detector, and a React monitoring interface into one system that can upload video, run inference, track people across frames, and flag suspicious events for human review.

## What this project actually does

This project is not a fully autonomous surveillance system. It is a review-support application designed to help operators:

- upload security or perimeter footage
- detect people with a trained YOLO11s model
- track each person across the video using ByteTrack
- classify detections into normal vs. potentially dangerous person events
- generate an annotated output video with bounding boxes and labels
- summarize alert time windows for review
- send email alerts when risky events are detected

The main logic is implemented in the backend pipeline under `backend/app/`, while the front-end dashboard sits in `frontend/src/`.

## Real system analysis

From the codebase, the system is structured around a practical monitoring workflow:

1. A user uploads a video through the FastAPI API.
2. The backend saves the file and starts a background processing job.
3. The model loads once at startup using `YOLO(str(model_path))`.
4. Each frame is processed with `model.track(...)` using ByteTrack.
5. Detections are filtered and counted by class:
   - `normal_person`
   - `potentially_dangerous_person`
6. Dangerous detections are marked as `alert_events` and grouped into time windows.
7. The annotated output video is generated and stored in `data/outputs`.
8. The frontend polls `/status/{job_id}` to display progress and then shows the processed result.

This means the app is best understood as an AI-assisted alerting and evidence-review tool rather than a direct enforcement system.

## Key Results

| Measure | YOLO11s result |
| --- | ---: |
| Precision | 0.87 |
| Recall | 0.69 |
| F1 score | 0.77 |
| mAP50 | 0.80 |
| mAP50-95 | 0.64 |
| Test images | 314 |

These values reflect the reported held-out evaluation for the selected YOLO11s configuration in the project workflow.

## Dataset

The final project dataset includes labeled person instances for normal and potentially dangerous-person classes.

| Split | Images | Normal-person instances | Potentially-dangerous-person instances | Total instances |
| --- | ---: | ---: | ---: | ---: |
| Train | 2,512 | 6,170 | 2,298 | 8,468 |
| Validation | 314 | 968 | 356 | 1,324 |
| Test | 314 | 539 | 264 | 803 |
| Total | 3,140 | 7,677 | 2,918 | 10,595 |

This repository is built around a YOLO-format dataset designed for person-context safety review.

## Architecture

This project follows a job-based surveillance pipeline with a human-review layer.

### End-to-end system flow

1. [frontend/src/App.jsx](frontend/src/App.jsx) provides the operator dashboard, upload flow, result viewer, history, and analytics pages.
2. [frontend/src/api.js](frontend/src/api.js) calls the FastAPI backend endpoints for upload, status polling, and output retrieval.
3. [backend/app/main.py](backend/app/main.py) validates the uploaded video, saves it under the upload directory, creates a unique job ID, and starts a background processing task.
4. [backend/app/job_store.py](backend/app/job_store.py) keeps track of each job's progress, frame counts, and output metadata in memory.
5. [backend/app/processor.py](backend/app/processor.py) loads the trained YOLO11s model, applies ByteTrack tracking, classifies `normal_person` and `potentially_dangerous_person`, merges alert frames into time windows, and writes the annotated MP4 output.
6. [backend/app/config.py](backend/app/config.py) provides the thresholds, model path, upload limits, CORS settings, and optional email configuration used by the processing pipeline.
7. [backend/app/models.py](backend/app/models.py) defines the API contract for upload responses, job status, alert events, and summaries.
8. [backend/app/email_alerts.py](backend/app/email_alerts.py) optionally sends a warning email when dangerous-person alerts are detected.
9. [frontend/src/App.jsx](frontend/src/App.jsx) polls `/status/{job_id}` until the job is complete, then renders the processed output video and alert summary for human review.

### Component responsibilities

#### Frontend

- [frontend/src/App.jsx](frontend/src/App.jsx) – main UI and operational dashboard
- [frontend/src/api.js](frontend/src/api.js) – backend communication layer
- [frontend/src/components/three/OrbitScene.jsx](frontend/src/components/three/OrbitScene.jsx) – visual shell for the dashboard view

#### Backend

- [backend/app/main.py](backend/app/main.py) – API entry point and job orchestration
- [backend/app/processor.py](backend/app/processor.py) – YOLO detection, tracking, annotation, and video export
- [backend/app/config.py](backend/app/config.py) – runtime configuration and model settings
- [backend/app/job_store.py](backend/app/job_store.py) – in-memory job state manager
- [backend/app/models.py](backend/app/models.py) – typed data models for the backend API
- [backend/app/email_alerts.py](backend/app/email_alerts.py) – optional notification layer for critical events

This system is designed as an AI-assisted review platform rather than a fully autonomous enforcement engine.

## Model behavior

The project uses a YOLO11s model file located at:

- `backend/models/best_phone_bag_hard_negative_yolo11s.pt`

In the backend config, the default model path is:

- `models/best_phone_bag_hard_negative_yolo11s.pt`

The model is configured with:

- confidence threshold: `0.30`
- dangerous threshold: `0.55`
- IoU threshold: `0.50`
- alert gap: `1.5` seconds
- frame skip: `1` by default

The processor code uses:

- `CLASS_NAMES = {0: "normal_person", 1: "potentially_dangerous_person"}`
- `DANGEROUS_CLASS = 1`

This confirms the project is focused on a person-centric safety alerting model rather than general object detection.

## Features in this repository

- Video upload endpoint using FastAPI
- Job-based processing for each uploaded clip
- Progress tracking and result polling in the frontend
- Output video generation with boxes and labels burned into frames
- People tracking across frames
- Unique person counting and alert event windows
- Optional Gmail notification support
- Local browser history for completed analyses

## Limitations and honest assessment

The repository is a prototype and has real operational limits:

- the job store is in-memory and resets on restart
- CPU inference can be slow when CUDA is unavailable
- live camera streaming is not a complete real-time inference endpoint yet
- image detection is intentionally limited in the current UI
- alerting is a review signal, not a legal or factual determination
- output encoding depends on browser and codec support

This is important context: the app is a decision-support prototype, not a standalone autonomous security authority.

## Repository

GitHub: https://github.com/Abir69-bot/Agentic-Safety-Security-Surveillance-System

## Local setup

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:MODEL_PATH="models\best_phone_bag_hard_negative_yolo11s.pt"
uvicorn app.main:app --reload --port 8000
```

### Frontend

```powershell
cd frontend
npm install
$env:VITE_API_BASE_URL="http://localhost:8000"
npm run dev
```

Open:

- http://localhost:5173

## Responsible use

This project should be used only in lawful, privacy-aware, human-reviewed workflows.

The code and UI clearly communicate that outputs are not proof of wrongdoing. They are alerts for operator review.

Use it responsibly by:

- keeping a human decision-maker in the loop
- securing footage and access
- validating performance in the real environment
- avoiding automated punitive action
- treating false positives and false negatives as operational risks

## Final project summary

This project is an AI-powered perimeter and safety monitoring prototype designed to support operators in reviewing footage for potentially dangerous activity. Its real value is in combining detection, tracking, annotation, alert grouping, and human-review workflow into a single end-to-end system.

It is practical, usable, and explainable as a decision-support tool, while still clearly emphasizing that it is not a final authority and must be used responsibly.


# Perimeter Watch

A lightweight AI surveillance project for monitoring video footage, tracking people across frames, and flagging potentially dangerous-person events for human review.

This repository includes a FastAPI backend, a React frontend, and a YOLO11-based detection pipeline designed for a real-world safety and perimeter monitoring workflow.

## Project idea

The goal of this system is to help operators review video streams more efficiently by automatically:

- detecting people in uploaded footage
- tracking them across the whole clip
- distinguishing normal persons from potential alert events
- generating an annotated output video
- summarizing alert windows for review
- sending email notifications when dangerous events are detected

This is a decision-support system, not an autonomous enforcement system.

## Features

- FastAPI backend for video uploads and processing jobs
- React dashboard for viewing results and status
- YOLO11s model with ByteTrack tracking
- Annotated output video export
- Alert summary and event window detection
- Gmail alert support for dangerous events
- Browser-based history for processed runs

## Tech stack

- Python
- FastAPI
- Ultralytics YOLO
- OpenCV
- ByteTrack
- React
- Vite
- JavaScript / JSX

## Repository

GitHub: https://github.com/Abir69-bot/Agentic-Safety-Security-Surveillance-System

## Run locally

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:MODEL_PATH="models\best_phone_bag_hard_negative_yolo11s.pt"
uvicorn app.main:app --reload --port 8000
```

Frontend:

```powershell
cd frontend
npm install
$env:VITE_API_BASE_URL="http://localhost:8000"
npm run dev
```

Then open:

- http://localhost:5173

## Configuration

The backend supports environment settings such as:

- `MODEL_PATH`
- `CONFIDENCE_THRESHOLD`
- `DANGEROUS_CONFIDENCE_THRESHOLD`
- `IOU_THRESHOLD`
- `ALERT_GAP_SECONDS`
- `FRAME_SKIP`
- `GMAIL_ADDRESS`
- `GMAIL_APP_PASSWORD`
- `ALERT_RECIPIENT`

These can be set in your shell or local environment before starting the app.

## Responsible use

- Keep a human reviewer in the loop
- Treat detections as alerts, not proof of wrongdoing
- Validate in real deployment conditions before production use
- Protect privacy and secure access to footage
- Use this system to support safety operations, not to replace judgment

## Notes

This project is intended as an AI-assisted surveillance prototype for research, development, and operational review. It is designed to make monitoring workflows more efficient while still requiring human oversight.

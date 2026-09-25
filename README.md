# Agentic Safety Security Surveillance System

A full-stack AI surveillance prototype for reviewing video footage, tracking people across frames, and flagging potentially dangerous-person events for human review.

This project combines a Python FastAPI backend, a YOLO11-based detection model, and a React monitoring dashboard to create a practical safety and perimeter monitoring workflow.

## Overview

The system is designed to:

- receive uploaded video files from a web client
- analyze frames with a trained YOLO11 model
- track people across the clip with ByteTrack
- mark normal vs. potentially dangerous detections
- generate an annotated output video
- surface suspicious windows for operator review
- optionally send alert emails when danger events are found

This is a review-oriented safety system, not an autonomous decision-maker. The model highlights possible risk and keeps a human in the loop.

## Key Features

- FastAPI backend for video upload and status polling
- React frontend dashboard for operations and review
- YOLO11s model for person-context detection
- ByteTrack-based multi-object tracking
- Alert event grouping over time
- Output video export with annotated boxes and labels
- Email notification support for dangerous event summaries
- Local browser-based history and results review

## Architecture

- Backend: `backend/`
  - `app/main.py` exposes the API
  - `app/processor.py` runs video analysis and model inference
  - `app/config.py` stores environment configuration
  - `app/email_alerts.py` sends alert emails
- Frontend: `frontend/`
  - React + Vite dashboard for uploading videos and viewing results
- Model: `backend/models/best_phone_bag_hard_negative_yolo11s.pt`

## Project Structure

```text
.
├── backend/
│   ├── app/
│   ├── data/
│   ├── models/
│   └── requirements.txt
├── frontend/
│   ├── src/
│   ├── package.json
│   └── vite.config.js
├── README.md
├── data/
├── index.html
└── LICENSE
```

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Abir69-bot/Agentic-Safety-Security-Surveillance-System.git
cd Agentic-Safety-Security-Surveillance-System
```

### 2. Start the backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Set the model path if needed and start the API:

```powershell
$env:MODEL_PATH="models\best_phone_bag_hard_negative_yolo11s.pt"
uvicorn app.main:app --reload --port 8000
```

The API runs on:

- http://localhost:8000

### 3. Start the frontend

```powershell
cd frontend
npm install
$env:VITE_API_BASE_URL="http://localhost:8000"
npm run dev
```

Then open:

- http://localhost:5173

## Supported Uploads

The backend accepts these video types:

- MP4
- MOV
- AVI
- MKV
- WEBM

Maximum upload size is 500MB.

## Configuration

The backend reads settings from environment variables and defaults from `backend/app/config.py`.

Example:

```powershell
$env:MODEL_PATH="models\best_phone_bag_hard_negative_yolo11s.pt"
$env:CONFIDENCE_THRESHOLD="0.30"
$env:DANGEROUS_CONFIDENCE_THRESHOLD="0.55"
$env:IOU_THRESHOLD="0.50"
$env:ALERT_GAP_SECONDS="1.5"
$env:FRAME_SKIP="1"
```

## Gmail Alerts

For email notifications, configure the Gmail account with an app password:

```powershell
$env:GMAIL_ADDRESS="owner@gmail.com"
$env:GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
$env:ALERT_RECIPIENT="owner@gmail.com"
```

> Do not commit real credentials. Use environment variables or a local `.env` file that is excluded from version control.

## Responsible Use

This project is intended for lawful monitoring and human-reviewed safety workflows only.

Please use it responsibly:

- keep a human reviewer in the loop
- avoid autonomous punitive action
- validate performance in real deployment settings
- protect privacy and access control
- treat model outputs as decision-support signals, not proof of wrongdoing

## Notes

This repository is a prototype for AI-assisted perimeter and safety monitoring. It is meant to support operations teams, investigators, and security reviewers with evidence-based alerts instead of fully automated enforcement.

## GitHub

Repository:

https://github.com/Abir69-bot/Agentic-Safety-Security-Surveillance-System


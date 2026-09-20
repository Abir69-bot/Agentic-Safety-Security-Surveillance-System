# Perimeter Watch

A FastAPI + React person-detection module for the larger Agentic Safety & Security Surveillance System. It runs YOLO11s with ByteTrack, returns an annotated video and summary, and can email the owner when dangerous-person events are found. The processor/job-store boundaries leave room for warehouse-safety and DCSASS anomaly models later.

## Run locally

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:MODEL_PATH="models\best_phone_bag_hard_negative_yolo11s.pt"
$env:GMAIL_ADDRESS="owner@gmail.com"
$env:GMAIL_APP_PASSWORD="your-google-app-password"
$env:ALERT_RECIPIENT="owner@gmail.com"
uvicorn app.main:app --reload --port 8000
```

Use a Gmail App Password created under Google 2-Step Verification, never the normal Gmail password. Alerts are sent only when dangerous-person event windows exist; mail failure does not fail the video job. Leave the Gmail variables unset to disable email.

Frontend, in a second terminal:

```powershell
cd frontend
npm install
$env:VITE_API_BASE_URL="http://localhost:8000"
npm run dev
```

Open `http://localhost:5173`.

## Configuration and limitations

`MODEL_PATH` defaults to the included workspace weight. `CONFIDENCE_THRESHOLD` defaults to `0.30`; `DANGEROUS_CONFIDENCE_THRESHOLD` defaults to `0.55` to reduce false dangerous-person alerts; `IOU_THRESHOLD` and `ALERT_GAP_SECONDS` default to `0.50` and `1.5`. `FRAME_SKIP=1` analyzes every frame; `FRAME_SKIP=2` or `3` can speed up long clips by reusing recent boxes, trading some tracking freshness for speed. Uploads are limited to 500MB and support mp4, mov, avi, mkv, and webm. The model loads once when FastAPI starts, and the backend logs CPU/GPU, model-load, inference, and video-write timings.

The development job store is in memory and resets on restart. The model loads per job to isolate tracker state, so overlapping jobs use more memory. CPU inference may be slow. `mp4v` output playback depends on browser codec support; H.264 transcoding may be needed for some browsers.

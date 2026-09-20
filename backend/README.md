# Perimeter Watch API

## Setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

The app uses the existing workspace weights by default. You can also copy the trained YOLO weights into this directory as `best.pt`, or point the app at a different file:

```powershell
$env:MODEL_PATH="models\best_phone_bag_hard_negative_yolo11s.pt"
uvicorn app.main:app --reload --port 8000
```

The API accepts `mp4`, `mov`, `avi`, `mkv`, and `webm` uploads up to 500MB. Configuration is available through `MODEL_PATH`, `CONFIDENCE_THRESHOLD`, `DANGEROUS_CONFIDENCE_THRESHOLD`, `IOU_THRESHOLD`, `ALERT_GAP_SECONDS`, and `FRAME_SKIP` environment variables. Dangerous predictions default to a stricter `0.55` confidence threshold to reduce false alarms; increase it for stricter alerts, or lower it if valid dangerous detections are being missed. `FRAME_SKIP=1` (default) runs YOLO on every frame; setting `FRAME_SKIP=2` or `3` reduces inference time by reusing the last boxes on skipped frames, with a corresponding reduction in box freshness and tracking accuracy.

## Gmail alerts

After enabling 2-Step Verification on the Gmail account, create a Google App Password. Do not use the normal Gmail password and do not commit credentials. Set these variables before starting the API:

```powershell
$env:GMAIL_ADDRESS="owner@gmail.com"
$env:GMAIL_APP_PASSWORD="xxxx xxxx xxxx xxxx"
$env:ALERT_RECIPIENT="owner@gmail.com" # optional; defaults to GMAIL_ADDRESS
```

When a clip contains at least one dangerous-person alert event, the API sends one email listing the job ID and event time windows through Gmail SMTP. If Gmail is unavailable, the video job still completes and the delivery failure is printed by the backend.

The tracker and processing code live behind a job store and processor boundary so this person-detection module can later share a pipeline with warehouse-safety and DCSASS anomaly models. The development job store is in memory and is not durable across restarts.

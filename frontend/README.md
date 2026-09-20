# Perimeter Watch Console

```powershell
cd frontend
npm install
$env:VITE_API_BASE_URL="http://localhost:8000"
npm run dev
```

Open `http://localhost:5173`. The client polls the FastAPI job status every second and displays the annotated video and alert windows when processing completes.

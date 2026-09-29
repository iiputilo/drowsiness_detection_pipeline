# 03_demo — Backend

FastAPI backend for the drowsiness detection demo application.

## Setup

### 1. Install requirements

```bash
pip install -r requirements.txt
```

### 2. Ensure trained models are available

Place the following files at the paths specified in `config.yaml` under `paths.models_dir` (default: `../../01_training/models`):
- `face_detector.pt`
- `states_detector.pt`

Run `01_training` first to produce these models.

### 3. Run the server

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The server starts even if models are not yet present; it will return an error through the WebSocket when a streaming session is started without models.

## Endpoints

- `GET /health` — returns server status and model availability
- `POST /upload_video` — uploads a video file, returns `video_id`
- `WS /stream` — WebSocket endpoint for real-time frame streaming

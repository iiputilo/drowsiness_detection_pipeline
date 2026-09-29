# 03_demo

Real-time drowsiness detection demonstration application — the functional prototype built on top of the pipelines from `01_training` and `02_testing`.

## What it does

The application streams video from your webcam or an uploaded video file, runs either the YOLO-based pipeline or the MediaPipe-based pipeline on each frame, and displays the result in real time. A sliding window logic aggregates frame-level predictions into alarm events.

## Running the application

### Step 1 — Start the backend

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Step 2 — Start the frontend

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

### Step 3 — Open in browser

Navigate to: http://localhost:5173

## Usage

1. Select a pipeline (YOLO or MediaPipe) using the pipeline cards
2. Select a source: Webcam (uses your default camera) or Video File (upload a video)
3. Toggle the overlay switch to enable/disable visual annotations on the frame
4. Press **Start** to begin streaming
5. The status panel shows the current drowsiness state, sliding window state, and performance metrics
6. Press **Stop** to end the session

## Prerequisites

The four trained YOLO models are committed at the path configured in `backend/config.yaml` under `paths.models_dir` (default: `../../01_training/models`), so no training run is required.

The MediaPipe pipeline additionally needs `face_landmarker.task` in that directory — it is not redistributed here:

```bash
curl -L -o ../../01_training/models/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Configuration

All backend parameters are in `backend/config.yaml`:
- `pipeline.window_size` and `pipeline.window_threshold`: sliding window parameters
- `pipeline.ear_threshold` and `pipeline.mar_threshold`: MediaPipe thresholds
- `stream.max_frame_width`: resize frames for performance
- `paths.models_dir`: location of trained model files

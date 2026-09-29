import asyncio
import base64
import json
import logging
import os
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import yaml
from fastapi import FastAPI, File, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


CONFIG = load_config()
PATHS = CONFIG["paths"]
STREAM_CFG = CONFIG["stream"]
PIPELINE_CFG = CONFIG["pipeline"]
SERVER_CFG = CONFIG["server"]

models_dir = Path(PATHS["models_dir"])
face_model_path = models_dir / "face_detector.pt"
eyes_model_path = models_dir / "eyes_detector.pt"
eyes_classifier_path = models_dir / "eyes_classifier.pt"
mouth_model_path = models_dir / "mouth_detector.pt"
models_loaded = (
    face_model_path.exists()
    and eyes_model_path.exists()
    and eyes_classifier_path.exists()
    and mouth_model_path.exists()
)

temp_video_dir = Path(PATHS["temp_video_dir"])
temp_video_dir.mkdir(parents=True, exist_ok=True)

if models_loaded:
    logger.info(f"Face model found: {face_model_path}")
    logger.info(f"Eyes detector found: {eyes_model_path}")
    logger.info(f"Eyes classifier found: {eyes_classifier_path}")
    logger.info(f"Mouth model found: {mouth_model_path}")
else:
    logger.warning(f"Models not found in: {models_dir}. Start pipeline after models are placed there.")

app = FastAPI(title="Drowsiness Detection Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return JSONResponse({
        "status": "ok",
        "models_loaded": models_loaded,
        "face_model_path": str(face_model_path),
        "eyes_model_path": str(eyes_model_path),
        "mouth_model_path": str(mouth_model_path),
    })


@app.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    try:
        video_id = str(uuid.uuid4())
        suffix = Path(file.filename).suffix or ".mp4"
        save_path = temp_video_dir / f"{video_id}{suffix}"
        content = await file.read()
        with open(str(save_path), "wb") as f:
            f.write(content)
        logger.info(f"Saved uploaded video: {save_path}")
        return JSONResponse({"video_id": video_id})
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)


def encode_frame_base64(frame: np.ndarray, quality: int) -> str:
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, buffer = cv2.imencode(".jpg", frame, encode_params)
    return base64.b64encode(buffer).decode("utf-8")


def resize_frame(frame: np.ndarray, max_width: int) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    ratio = max_width / w
    new_h = int(h * ratio)
    return cv2.resize(frame, (max_width, new_h))


@app.websocket("/stream")
async def stream_ws(websocket: WebSocket):
    await websocket.accept()
    logger.info("WebSocket client connected")

    source_obj = None
    pipeline_obj = None

    try:
        init_msg = await websocket.receive_text()
        config_msg = json.loads(init_msg)

        source_type = config_msg.get("source", "webcam")
        pipeline_type = config_msg.get("pipeline", "yolo")
        show_overlay = config_msg.get("show_overlay", True)
        gaze_tracking = bool(config_msg.get("gaze_tracking", False))
        video_id = config_msg.get("video_id")

        if not models_loaded:
            await websocket.send_text(json.dumps({
                "error": "Models not loaded. Place face_detector.pt, eyes_detector.pt, eyes_classifier.pt and mouth_detector.pt in models_dir."
            }))
            await websocket.close()
            return

        from video_source import VideoSource
        from sliding_window import SlidingWindow

        if source_type == "webcam":
            source_obj = VideoSource(0)
        else:
            if not video_id:
                await websocket.send_text(json.dumps({"error": "video_id required for video source"}))
                await websocket.close()
                return
            video_files = list(temp_video_dir.glob(f"{video_id}*"))
            if not video_files:
                await websocket.send_text(json.dumps({"error": f"Video not found: {video_id}"}))
                await websocket.close()
                return
            source_obj = VideoSource(str(video_files[0]))

        if pipeline_type == "yolo":
            from pipelines.yolo_pipeline import YOLOPipeline
            pipeline_obj = YOLOPipeline(str(face_model_path), str(eyes_model_path), str(eyes_classifier_path), str(mouth_model_path))
            gaze_tracking = False
        else:
            from pipelines.mediapipe_pipeline import MediaPipePipeline
            pipeline_obj = MediaPipePipeline(
                ear_threshold=PIPELINE_CFG["ear_threshold"],
                mar_threshold=PIPELINE_CFG["mar_threshold"],
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                model_path=PATHS["face_landmarker_model"],
            )

        _pw = PIPELINE_CFG.get("pipelines", {}).get(pipeline_type, {})
        _window_duration = _pw.get("window_duration", PIPELINE_CFG["window_duration"])
        _drowsy_fraction = _pw.get("drowsy_fraction", PIPELINE_CFG["drowsy_fraction"])
        sw = SlidingWindow(_window_duration, _drowsy_fraction)
        logger.info(f"Window: {_window_duration}s, drowsy_fraction={_drowsy_fraction}")

        gaze_threshold = PIPELINE_CFG.get("gaze_threshold", 0.35)
        gaze_window_duration = PIPELINE_CFG.get("gaze_window_duration", _window_duration)
        gaze_fraction = PIPELINE_CFG.get("gaze_fraction", _drowsy_fraction)
        sw_gaze = SlidingWindow(gaze_window_duration, gaze_fraction) if gaze_tracking else None
        quality = STREAM_CFG["jpeg_quality"]
        max_width = STREAM_CFG["max_frame_width"]

        logger.info(f"Streaming started: source={source_type}, pipeline={pipeline_type}")

        should_stop = False

        async def receive_controls() -> None:
            nonlocal should_stop
            while not should_stop:
                try:
                    msg_text = await asyncio.wait_for(websocket.receive_text(), timeout=0.01)
                    ctrl = json.loads(msg_text)
                    action = ctrl.get("action")
                    if action == "pause":
                        source_obj.pause()
                    elif action == "resume":
                        source_obj.resume()
                    elif action == "reset":
                        source_obj.reset()
                        sw.reset()
                    elif action == "stop":
                        should_stop = True
                except asyncio.TimeoutError:
                    pass
                except Exception:
                    should_stop = True
                    break

        control_task = asyncio.create_task(receive_controls())

        while not should_stop:
            ret, frame = source_obj.get_frame()
            if not ret:
                if source_obj.is_paused:
                    await asyncio.sleep(0.05)
                    continue
                if source_type == "video":
                    await websocket.send_text(json.dumps({"event": "video_ended"}))
                should_stop = True
                break

            frame = resize_frame(frame, max_width)

            try:
                result = pipeline_obj.predict(frame)
            except Exception as e:
                logger.warning(f"Prediction error: {e}")
                continue

            is_drowsy = result.get("state") == "drowsy"
            drowsy_alarm = sw.update(is_drowsy)
            window_state = sw.get_current_state()

            gaze = result.get("gaze")
            gaze_alarm = False
            gaze_window_state = None
            if gaze_tracking and sw_gaze is not None:
                gaze_alarm = sw_gaze.update(gaze is not None and gaze >= gaze_threshold)
                gaze_window_state = sw_gaze.get_current_state()

            alarm = drowsy_alarm or gaze_alarm

            frame_main = frame
            frame_face = None

            if pipeline_type == "yolo":
                frame_main, frame_face = pipeline_obj.render(frame, result, show_overlay)
            else:
                frame_main = pipeline_obj.render(frame, result, show_overlay)

            main_b64 = encode_frame_base64(frame_main, quality)
            face_b64 = encode_frame_base64(frame_face, quality) if frame_face is not None else None

            ear = result.get("ear")
            mar = result.get("mar")

            state = result.get("state", "no_face")
            if state != "no_face":
                if gaze_tracking and gaze_alarm:
                    state = "distracted"
                elif drowsy_alarm:
                    state = "drowsy"
                else:
                    state = "alert"

            payload = {
                "frame_main": main_b64,
                "frame_face": face_b64,
                "state": state,
                "alarm": alarm,
                "metrics": {
                    "ear": round(ear, 4) if ear is not None else None,
                    "mar": round(mar, 4) if mar is not None else None,
                    "gaze": round(gaze, 4) if gaze is not None else None,
                    "gaze_enabled": gaze_tracking,
                    "gaze_window_state": gaze_window_state,
                    "inference_ms": round(result.get("inference_ms", 0), 2),
                    "window_state": window_state,
                    "alarm_count": sw.alarm_count,
                },
                "timestamp": time.time(),
            }

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(0)

        control_task.cancel()
        try:
            await control_task
        except asyncio.CancelledError:
            pass

    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if source_obj:
            source_obj.release()
        logger.info("WebSocket streaming stopped")

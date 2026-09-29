import logging
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
# [left_corner, upper_left, upper_center, upper_right, right_corner, lower_right, lower_center, lower_left]
MOUTH_INDICES = [61, 39, 0, 269, 291, 405, 17, 181]

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)


def _ensure_model(model_path: str) -> str:
    path = Path(model_path)
    if not path.exists():
        logger.info(f"Downloading face_landmarker.task → {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_MODEL_URL, path)
        logger.info("Download complete")
    return str(path)


class MediaPipePipeline:
    def __init__(
        self,
        ear_threshold: float,
        mar_threshold: float,
        min_detection_confidence: float,
        min_tracking_confidence: float,
        model_path: str = "face_landmarker.task",
    ) -> None:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision

        self.ear_threshold = ear_threshold
        self.mar_threshold = mar_threshold
        self._mp = mp

        model_path = _ensure_model(model_path)
        base_options = mp_tasks.BaseOptions(model_asset_path=model_path)
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=min_detection_confidence,
            min_face_presence_confidence=min_tracking_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(options)

    def compute_ear(self, landmarks: list, indices: list, w: int, h: int) -> float:
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
        a = np.linalg.norm(pts[1] - pts[5])
        b = np.linalg.norm(pts[2] - pts[4])
        c = np.linalg.norm(pts[0] - pts[3])
        if c < 1e-6:
            return 0.0
        return (a + b) / (2.0 * c)

    def compute_mar(self, landmarks: list, indices: list, w: int, h: int) -> float:
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
        a = np.linalg.norm(pts[1] - pts[7])
        b = np.linalg.norm(pts[2] - pts[6])
        c = np.linalg.norm(pts[3] - pts[5])
        d = np.linalg.norm(pts[0] - pts[4])
        if d < 1e-6:
            return 0.0
        return (a + b + c) / (2.0 * d)

    def predict(self, frame: np.ndarray) -> dict:
        t_start = time.perf_counter()
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        try:
            mp_image = self._mp.Image(
                image_format=self._mp.ImageFormat.SRGB, data=rgb_frame
            )
            results = self.face_landmarker.detect(mp_image)
        except Exception as e:
            logger.warning(f"FaceLandmarker processing failed: {e}")
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return {"state": "no_face", "ear": None, "mar": None, "landmarks": None, "inference_ms": elapsed}

        if not results.face_landmarks:
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return {"state": "no_face", "ear": None, "mar": None, "landmarks": None, "inference_ms": elapsed}

        lms = results.face_landmarks[0]
        landmark_coords = [[int(lm.x * w), int(lm.y * h)] for lm in lms]

        try:
            ear_left = self.compute_ear(lms, LEFT_EYE_INDICES, w, h)
            ear_right = self.compute_ear(lms, RIGHT_EYE_INDICES, w, h)
            ear = (ear_left + ear_right) / 2.0
            mar = self.compute_mar(lms, MOUTH_INDICES, w, h)
        except Exception as e:
            logger.warning(f"EAR/MAR computation failed: {e}")
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return {"state": "alert", "ear": None, "mar": None, "landmarks": landmark_coords, "inference_ms": elapsed}

        state = "drowsy" if (ear < self.ear_threshold or mar > self.mar_threshold) else "alert"
        elapsed = (time.perf_counter() - t_start) * 1000.0

        return {
            "state": state,
            "ear": ear,
            "mar": mar,
            "landmarks": landmark_coords,
            "inference_ms": elapsed,
        }

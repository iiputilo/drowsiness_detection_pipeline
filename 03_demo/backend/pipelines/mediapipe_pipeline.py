import logging
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

logger = logging.getLogger(__name__)

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
# MediaPipe Face Landmarker / Face Mesh iris landmarks:
# 468-472 belong to the subject's right eye, 473-477 to the subject's left eye.
# Mixing these with the opposite eye corners projects the iris outside the eye and
# makes the normalized gaze score clamp to 1.0 on almost every frame.
RIGHT_IRIS_INDICES = [468, 469, 470, 471, 472]
LEFT_IRIS_INDICES = [473, 474, 475, 476, 477]
LEFT_EYE_CORNERS = (362, 263)
RIGHT_EYE_CORNERS = (33, 133)
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

    def compute_ear(self, landmarks, indices: list, w: int, h: int) -> float:
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
        a = np.linalg.norm(pts[1] - pts[5])
        b = np.linalg.norm(pts[2] - pts[4])
        c = np.linalg.norm(pts[0] - pts[3])
        if c < 1e-6:
            return 0.0
        return (a + b) / (2.0 * c)

    def compute_mar(self, landmarks, indices: list, w: int, h: int) -> float:
        pts = np.array([[landmarks[i].x * w, landmarks[i].y * h] for i in indices])
        a = np.linalg.norm(pts[1] - pts[7])
        b = np.linalg.norm(pts[2] - pts[6])
        c = np.linalg.norm(pts[3] - pts[5])
        d = np.linalg.norm(pts[0] - pts[4])
        if d < 1e-6:
            return 0.0
        return (a + b + c) / (2.0 * d)

    def _landmark_xy(self, landmarks, idx: int, w: int, h: int) -> np.ndarray:
        return np.array([landmarks[idx].x * w, landmarks[idx].y * h], dtype=np.float32)

    def _iris_center(self, landmarks, indices: list[int], w: int, h: int) -> np.ndarray:
        return np.mean(
            np.array([self._landmark_xy(landmarks, i, w, h) for i in indices], dtype=np.float32),
            axis=0,
        )

    def _horizontal_eye_ratio(
        self,
        iris: np.ndarray,
        corner_a: np.ndarray,
        corner_b: np.ndarray,
    ) -> float | None:
        """Return iris position on the eye corner axis: 0=corner_a, 1=corner_b."""
        eye_vec = corner_b - corner_a
        denom = float(np.dot(eye_vec, eye_vec))
        if denom < 1e-6:
            return None
        return float(np.dot(iris - corner_a, eye_vec) / denom)

    def compute_gaze_score(self, landmarks, w: int, h: int) -> float | None:
        try:
            if len(landmarks) <= max(LEFT_IRIS_INDICES + RIGHT_IRIS_INDICES):
                return None

            left_corner_a = self._landmark_xy(landmarks, LEFT_EYE_CORNERS[0], w, h)
            left_corner_b = self._landmark_xy(landmarks, LEFT_EYE_CORNERS[1], w, h)
            right_corner_a = self._landmark_xy(landmarks, RIGHT_EYE_CORNERS[0], w, h)
            right_corner_b = self._landmark_xy(landmarks, RIGHT_EYE_CORNERS[1], w, h)

            left_iris = self._iris_center(landmarks, LEFT_IRIS_INDICES, w, h)
            right_iris = self._iris_center(landmarks, RIGHT_IRIS_INDICES, w, h)

            left_ratio = self._horizontal_eye_ratio(left_iris, left_corner_a, left_corner_b)
            right_ratio = self._horizontal_eye_ratio(right_iris, right_corner_a, right_corner_b)
            if left_ratio is None or right_ratio is None:
                return None

            # Correctly paired iris/corners should normally stay near [0, 1]. If it is
            # far outside, landmarks are unreliable for this frame; do not convert it
            # into a permanent max distraction score.
            if not (-0.25 <= left_ratio <= 1.25 and -0.25 <= right_ratio <= 1.25):
                logger.debug(
                    "Unreliable gaze landmarks: left_ratio=%.3f right_ratio=%.3f",
                    left_ratio,
                    right_ratio,
                )
                return None

            left_ratio = float(np.clip(left_ratio, 0.0, 1.0))
            right_ratio = float(np.clip(right_ratio, 0.0, 1.0))
            horizontal_ratio = (left_ratio + right_ratio) / 2.0

            # 0.0 means both irises are close to eye centers; 1.0 means they are near
            # the side corners. Threshold is configured in config.yaml.
            return float(np.clip(abs(horizontal_ratio - 0.5) * 2.0, 0.0, 1.0))
        except Exception:
            return None

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
        iris_centers = None

        try:
            ear_left = self.compute_ear(lms, LEFT_EYE_INDICES, w, h)
            ear_right = self.compute_ear(lms, RIGHT_EYE_INDICES, w, h)
            ear = (ear_left + ear_right) / 2.0
            mar = self.compute_mar(lms, MOUTH_INDICES, w, h)
            gaze = self.compute_gaze_score(lms, w, h)
            if len(lms) > max(LEFT_IRIS_INDICES + RIGHT_IRIS_INDICES):
                iris_centers = [
                    self._iris_center(lms, LEFT_IRIS_INDICES, w, h).astype(int).tolist(),
                    self._iris_center(lms, RIGHT_IRIS_INDICES, w, h).astype(int).tolist(),
                ]
        except Exception as e:
            logger.warning(f"EAR/MAR computation failed: {e}")
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return {
                "state": "alert",
                "ear": None,
                "mar": None,
                "gaze": None,
                "landmarks": landmark_coords,
                "iris_centers": iris_centers,
                "inference_ms": elapsed,
            }

        state = "drowsy" if (ear < self.ear_threshold or mar > self.mar_threshold) else "alert"
        elapsed = (time.perf_counter() - t_start) * 1000.0

        return {
            "state": state,
            "ear": ear,
            "mar": mar,
            "gaze": gaze,
            "landmarks": landmark_coords,
            "iris_centers": iris_centers,
            "inference_ms": elapsed,
        }

    def render(self, frame: np.ndarray, result: dict, show_overlay: bool) -> np.ndarray:
        annotated = frame.copy()

        if not show_overlay:
            return annotated

        lms = result.get("landmarks")
        if lms:
            for pt in lms:
                cv2.circle(annotated, tuple(pt), 1, (0, 255, 0), -1)
            for idx in LEFT_EYE_INDICES + RIGHT_EYE_INDICES:
                if idx < len(lms):
                    cv2.circle(annotated, tuple(lms[idx]), 2, (255, 255, 0), -1)
            for idx in MOUTH_INDICES:
                if idx < len(lms):
                    cv2.circle(annotated, tuple(lms[idx]), 2, (0, 255, 255), -1)
            for idx in LEFT_IRIS_INDICES + RIGHT_IRIS_INDICES:
                if idx < len(lms):
                    cv2.circle(annotated, tuple(lms[idx]), 2, (255, 0, 255), -1)
            for pt in result.get("iris_centers") or []:
                cv2.circle(annotated, tuple(pt), 4, (0, 0, 255), -1)

        state = result.get("state", "alert")
        ear = result.get("ear")
        mar = result.get("mar")
        gaze = result.get("gaze")
        color = (0, 0, 255) if state == "drowsy" else (0, 255, 0)

        y = 25
        if ear is not None:
            cv2.putText(annotated, f"EAR: {ear:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25
        if mar is not None:
            cv2.putText(annotated, f"MAR: {mar:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25
        if gaze is not None:
            cv2.putText(annotated, f"GAZE: {gaze:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y += 25
        cv2.putText(annotated, f"STATE: {state.upper()}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        return annotated

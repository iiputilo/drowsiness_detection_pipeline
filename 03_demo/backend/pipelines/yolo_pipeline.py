import logging
import time

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

EYES_DROWSY_CLASSES = {"closed"}
MOUTH_DROWSY_CLASSES = {"open_mouth"}
DROWSY_CONFIDENCE_THRESHOLD = 0.5


class YOLOPipeline:
    def __init__(
        self,
        face_model_path: str,
        eyes_detector_path: str,
        eyes_classifier_path: str,
        mouth_model_path: str,
    ) -> None:
        from ultralytics import YOLO

        if torch.cuda.is_available():
            self.device = "cuda:0"
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = "cpu"
            logger.warning("CUDA not available, falling back to CPU")

        logger.info(f"Loading face model: {face_model_path}")
        self.face_model = YOLO(face_model_path)
        self.face_model.to(self.device)

        logger.info(f"Loading eyes detector: {eyes_detector_path}")
        self.eyes_detector = YOLO(eyes_detector_path)
        self.eyes_detector.to(self.device)

        logger.info(f"Loading eyes classifier: {eyes_classifier_path}")
        self.eyes_classifier = YOLO(eyes_classifier_path)
        self.eyes_classifier.to(self.device)

        logger.info(f"Loading mouth model: {mouth_model_path}")
        self.mouth_model = YOLO(mouth_model_path)
        self.mouth_model.to(self.device)

        actual_device = next(self.face_model.model.parameters()).device
        logger.info(f"Models loaded on device: {actual_device}")

    def _classify_eye(self, eye_crop: np.ndarray) -> tuple[str, float]:
        # conf=0.0 disables YOLO's default 0.25 threshold so detect-type classifiers
        # always return at least one prediction on a small eye crop
        results = self.eyes_classifier(eye_crop, verbose=False, device=self.device, conf=0.0)
        r = results[0]
        # YOLO classify model
        if hasattr(r, "probs") and r.probs is not None:
            top_idx = int(r.probs.top1)
            return r.names[top_idx], float(r.probs.top1conf)
        # YOLO detect model
        boxes = getattr(r, "boxes", None)
        if boxes is not None and len(boxes) > 0:
            best = int(boxes.conf.argmax())
            cls = int(boxes.cls[best])
            return r.names[cls], float(boxes.conf[best])
        return "open", 0.0

    def predict(self, frame: np.ndarray) -> dict:
        t_start = time.perf_counter()

        try:
            face_results = self.face_model(frame, verbose=False, device=self.device)
        except Exception as e:
            logger.warning(f"Face detection failed: {e}")
            return {
                "state": "no_face",
                "face_box": None,
                "eyes_detections": [],
                "mouth_detections": [],
                "inference_ms": 0.0,
            }

        face_box = None
        if face_results and len(face_results[0].boxes) > 0:
            boxes = face_results[0].boxes.xyxy.cpu().numpy()
            confs = face_results[0].boxes.conf.cpu().numpy()
            best_idx = int(np.argmax(confs))
            x1, y1, x2, y2 = boxes[best_idx].astype(int)

            h, w = frame.shape[:2]
            pad_x = int((x2 - x1) * 0.10)
            pad_y = int((y2 - y1) * 0.10)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            face_box = [x1, y1, x2, y2]

        if face_box is None:
            elapsed = (time.perf_counter() - t_start) * 1000.0
            return {
                "state": "no_face",
                "face_box": None,
                "eyes_detections": [],
                "mouth_detections": [],
                "inference_ms": elapsed,
            }

        x1, y1, x2, y2 = face_box
        face_crop = frame[y1:y2, x1:x2]

        # Stage 1: detect eye regions in the face crop
        eyes_detections = []
        is_eyes_drowsy = False
        try:
            det_results = self.eyes_detector(face_crop, verbose=False, device=self.device)
            _boxes = det_results[0].boxes if det_results else None
            if _boxes is not None and len(_boxes) > 0:
                eye_boxes = _boxes.xyxy.cpu().numpy().astype(int)
                fh, fw = face_crop.shape[:2]
                for box in eye_boxes:
                    bx1 = max(0, box[0])
                    bx2 = min(fw, box[2])
                    w = bx2 - bx1
                    if w <= 0:
                        continue
                    # Make a square crop: height = width, centred on original bbox
                    cy = (box[1] + box[3]) // 2
                    by1 = max(0, cy - w // 2)
                    by2 = min(fh, by1 + w)
                    # Re-anchor if clipped at bottom
                    if by2 - by1 < w:
                        by1 = max(0, by2 - w)
                    eye_crop = face_crop[by1:by2, bx1:bx2]
                    if eye_crop.size == 0:
                        continue
                    # Stage 2: classify each detected eye region
                    class_name, confidence = self._classify_eye(eye_crop)
                    eyes_detections.append({
                        "class_name": class_name,
                        "confidence": confidence,
                        "box": [int(bx1), int(by1), int(bx2), int(by2)],
                    })
                    if class_name in EYES_DROWSY_CLASSES and confidence >= DROWSY_CONFIDENCE_THRESHOLD:
                        is_eyes_drowsy = True
        except Exception as e:
            logger.warning(f"Eyes detection/classification failed: {e}")

        mouth_detections = []
        is_mouth_drowsy = False
        try:
            mouth_results = self.mouth_model(face_crop, verbose=False, device=self.device)
            if mouth_results and len(mouth_results[0].boxes) > 0:
                boxes = mouth_results[0].boxes.xyxy.cpu().numpy()
                confs = mouth_results[0].boxes.conf.cpu().numpy()
                classes = mouth_results[0].boxes.cls.cpu().numpy().astype(int)
                names = mouth_results[0].names
                for i in range(len(boxes)):
                    class_name = names.get(classes[i], str(classes[i]))
                    confidence = float(confs[i])
                    mouth_detections.append({
                        "class_name": class_name,
                        "confidence": confidence,
                        "box": boxes[i].tolist(),
                    })
                    if class_name in MOUTH_DROWSY_CLASSES and confidence >= DROWSY_CONFIDENCE_THRESHOLD:
                        is_mouth_drowsy = True
        except Exception as e:
            logger.warning(f"Mouth detection failed: {e}")

        state = "drowsy" if (is_eyes_drowsy or is_mouth_drowsy) else "alert"
        elapsed = (time.perf_counter() - t_start) * 1000.0

        return {
            "state": state,
            "face_box": face_box,
            "eyes_detections": eyes_detections,
            "mouth_detections": mouth_detections,
            "inference_ms": elapsed,
        }

    def render(self, frame: np.ndarray, result: dict, show_overlay: bool) -> tuple:
        main_frame = frame.copy()

        face_box = result.get("face_box")
        if face_box is None:
            return main_frame, None

        x1, y1, x2, y2 = face_box
        face_frame = frame[y1:y2, x1:x2].copy()

        if show_overlay:
            state = result.get("state", "alert")
            box_color = (0, 0, 255) if state == "drowsy" else (0, 255, 0)

            cv2.rectangle(main_frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                main_frame, state.upper(),
                (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2
            )

            for det in result.get("eyes_detections", []):
                class_name = det["class_name"]
                confidence = det["confidence"]
                bx1, by1, bx2, by2 = [int(v) for v in det["box"]]
                color = (0, 0, 255) if class_name in EYES_DROWSY_CLASSES else (0, 255, 0)
                cv2.rectangle(face_frame, (bx1, by1), (bx2, by2), color, 1)
                cv2.putText(
                    face_frame, f"{class_name} {confidence:.2f}",
                    (bx1, max(by1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

            for det in result.get("mouth_detections", []):
                class_name = det["class_name"]
                confidence = det["confidence"]
                bx1, by1, bx2, by2 = [int(v) for v in det["box"]]
                color = (0, 0, 255) if class_name in MOUTH_DROWSY_CLASSES else (0, 255, 0)
                cv2.rectangle(face_frame, (bx1, by1), (bx2, by2), color, 1)
                cv2.putText(
                    face_frame, f"{class_name} {confidence:.2f}",
                    (bx1, max(by1 - 4, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1
                )

        return main_frame, face_frame

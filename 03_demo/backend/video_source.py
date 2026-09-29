import logging
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)

TARGET_FPS = 30.0
FRAME_DURATION = 1.0 / TARGET_FPS


class VideoSource:
    def __init__(self, source) -> None:
        self._source = source
        self._is_webcam = source == 0 or source == "0"
        self._cap = cv2.VideoCapture(int(source) if self._is_webcam else source)
        self._paused = False
        self._last_frame_time = 0.0

        if not self._cap.isOpened():
            logger.error(f"Could not open video source: {source}")

    @property
    def fps(self) -> float:
        raw = self._cap.get(cv2.CAP_PROP_FPS)
        return raw if raw and raw > 0 else TARGET_FPS

    @property
    def is_paused(self) -> bool:
        return self._paused

    def pause(self) -> None:
        self._paused = True
        logger.debug("VideoSource paused")

    def resume(self) -> None:
        self._paused = False
        self._last_frame_time = 0.0
        logger.debug("VideoSource resumed")

    def get_frame(self) -> tuple:
        if self._paused:
            return False, None

        if not self._is_webcam:
            now = time.perf_counter()
            elapsed = now - self._last_frame_time
            if self._last_frame_time > 0 and elapsed < FRAME_DURATION:
                time.sleep(FRAME_DURATION - elapsed)
            self._last_frame_time = time.perf_counter()

        if not self._cap.isOpened():
            return False, None

        ret, frame = self._cap.read()
        return ret, frame

    def reset(self) -> None:
        if not self._is_webcam:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._last_frame_time = 0.0
            logger.debug("VideoSource reset to beginning")

    def release(self) -> None:
        if self._cap.isOpened():
            self._cap.release()
        logger.debug("VideoSource released")

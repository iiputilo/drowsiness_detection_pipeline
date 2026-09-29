import time
from collections import deque


class SlidingWindow:
    def __init__(self, window_duration: float, drowsy_fraction: float) -> None:
        self.window_duration = window_duration
        self.drowsy_fraction = drowsy_fraction
        self._entries: deque = deque()  # (timestamp, is_drowsy)
        self._start_time: float | None = None
        self._alarm_count: int = 0
        self._prev_alarm: bool = False

    def update(self, is_drowsy: bool) -> bool:
        ts = time.monotonic()

        if self._start_time is None:
            self._start_time = ts

        self._entries.append((ts, is_drowsy))

        cutoff = ts - self.window_duration
        while self._entries and self._entries[0][0] < cutoff:
            self._entries.popleft()

        alarm = self._compute_alarm(ts)
        if alarm and not self._prev_alarm:
            self._alarm_count += 1
        self._prev_alarm = alarm
        return alarm

    def _compute_alarm(self, current_ts: float) -> bool:
        if not self._entries or self._start_time is None:
            return False
        if current_ts - self._start_time < self.window_duration:
            return False
        drowsy_count = sum(1 for _, d in self._entries if d)
        return (drowsy_count / len(self._entries)) >= self.drowsy_fraction

    def reset(self) -> None:
        self._entries.clear()
        self._start_time = None
        self._prev_alarm = False

    @property
    def alarm_count(self) -> int:
        return self._alarm_count

    def get_current_state(self) -> dict:
        if not self._entries or self._start_time is None:
            return {"fraction": 0.0, "alarm": False, "fill_ratio": 0.0, "frame_count": 0}
        current_ts = self._entries[-1][0]
        fill_ratio = min((current_ts - self._start_time) / self.window_duration, 1.0)
        drowsy_count = sum(1 for _, d in self._entries if d)
        fraction = drowsy_count / len(self._entries)
        return {
            "fraction": round(fraction, 3),
            "alarm": self._compute_alarm(current_ts),
            "fill_ratio": round(fill_ratio, 3),
            "frame_count": len(self._entries),
        }

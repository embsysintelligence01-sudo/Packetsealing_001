"""Bounded display persistence; cached drawings are never inference evidence."""
import threading

import cv2
import numpy as np


class DetectionPreview:
    def __init__(self, max_age=2.5, fresh_age=.75, missed_hold=.6):
        self.max_age, self.fresh_age, self.missed_hold = max_age, fresh_age, missed_hold
        self.lock = threading.Lock()
        self.layers = {}
        self._amber = None

    def publish(self, image, captured_at, *, key='default', hold_empty=False):
        mask = cv2.bitwise_not(cv2.inRange(image, (0, 0, 0), (0, 0, 0)))
        with self.lock:
            previous = self.layers.get(key)
            if not np.any(mask):
                if hold_empty and previous and 0 <= captured_at-previous[2] <= self.missed_hold:
                    self.layers[key] = (*previous[:3], True)
                else:
                    self.layers.pop(key, None)
            else:
                self.layers[key] = (image, mask, captured_at, False)

    def clear(self):
        with self.lock:
            self.layers.clear()

    def render(self, frame, captured_at):
        with self.lock:
            layers = list(self.layers.values())
        held = False
        for image, mask, ts, missed in layers:
            age = captured_at-ts
            limit = self.missed_hold if missed else self.max_age
            if image.shape != frame.shape or not 0 <= age <= limit:
                continue
            if missed or age > self.fresh_age:
                # Amber communicates that geometry is held, rather than fresh.
                if self._amber is None or self._amber.shape != image.shape:
                    self._amber = np.full_like(image, (0, 170, 255))
                cv2.copyTo(self._amber, mask, frame)
                held = True
            else:
                cv2.copyTo(image, mask, frame)
        if held:
            cv2.putText(frame, 'Last detection - awaiting update', (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, (0, 170, 255), 2)
        return frame

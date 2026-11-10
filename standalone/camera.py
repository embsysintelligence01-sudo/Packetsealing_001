"""Real-time file replay alongside the tracker's USB/RTSP capture."""
from pathlib import Path
import threading
import time

import cv2
import jewel_tracker_5 as original


class VideoFileGrabber(original.RTSPGrabber):
    def __init__(self, source, camera=None):
        super().__init__(source, camera)
        self.ended = False
        self.stop_event = threading.Event()

    def run(self):
        cap = cv2.VideoCapture(str(self.source))
        try:
            if not cap.isOpened():
                self.problem = "Could not open video source."
                return
            fps = cap.get(cv2.CAP_PROP_FPS)
            fps = fps if 0 < fps <= 240 else 25.
            self.connected = True
            deadline = time.monotonic()
            while self.running:
                ok, frame = cap.read()
                if not ok:
                    break
                if self.stop_event.wait(max(0., deadline - time.monotonic())):
                    break
                t = time.time()
                if self.on_frame:
                    self.on_frame(frame, t)
                with self.lock:
                    self.frame, self.ts = frame, t
                deadline += 1 / fps
        finally:
            cap.release()
            self.connected, self.ended = False, True

    def stop(self):
        self.running = False
        self.stop_event.set()


def make_grabber(source, camera):
    if isinstance(source, str) and Path(source).is_file():
        return VideoFileGrabber(source, camera)
    return original.RTSPGrabber(source, camera)

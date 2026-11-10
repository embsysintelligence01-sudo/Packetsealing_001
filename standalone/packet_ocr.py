"""Post-sealing number capture, with Paddle isolated in another interpreter."""
import base64
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import time

import cv2
import numpy as np
import torch
from ultralytics.data.augment import LetterBox
from ultralytics.utils import nms, ops

from .models import session


class NumberDetector:
    """Decode the supplied YOLOv8 OBB export, not a segmentation model."""
    def __init__(self, model):
        self.session = session(model)
        inp = self.session.get_inputs()[0]
        if inp.shape != [1, 3, 640, 640]:
            raise ValueError("Packet-number model must have a 640x640 RGB input")
        self.input = inp.name
        self.letterbox = LetterBox((640, 640), auto=False)

    def crop(self, frame, roi):
        padded = self.letterbox(image=frame)
        x = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.
        raw = self.session.run(None, {self.input: x})[0]
        pred = nms.non_max_suppression(torch.from_numpy(raw), .5, .4,
                                       nc=1, rotated=True, max_det=8)[0]
        if not len(pred):
            return None, None
        boxes = torch.cat((pred[:, :4], pred[:, -1:]), dim=1)
        polygons = ops.xywhr2xyxyxyxy(boxes).numpy()
        # Undo the exact LetterBox padding/scale before perspective rectification.
        h, w = frame.shape[:2]
        gain = min(640 / h, 640 / w)
        pad = np.array([round((640-round(w*gain))/2-.1),
                        round((640-round(h*gain))/2-.1)], np.float32)
        polygons = (polygons-pad)/gain
        valid = []
        for polygon in polygons:
            center = polygon.mean(axis=0)
            if cv2.pointPolygonTest(np.asarray(roi, np.float32), tuple(center), False) < 0:
                continue
            if (polygon[:, 0].min() < 0 or polygon[:, 1].min() < 0 or
                    polygon[:, 0].max() >= w or polygon[:, 1].max() >= h):
                continue  # A clipped number is not evidence of the whole number.
            valid.append(polygon.astype(np.float32))
        if len(valid) != 1:
            return None, None  # Do not choose arbitrarily between two labels.
        # Ultralytics returns BR, TR, TL, BL at zero angle. Reorder to
        # TL, TR, BR, BL so rectification does not mirror the printed text.
        polygon = valid[0][[2, 1, 0, 3]]
        width = round(np.linalg.norm(polygon[1]-polygon[0]))
        height = round(np.linalg.norm(polygon[2]-polygon[1]))
        if min(width, height) < 12 or max(width, height) < 24:
            return None, None
        target = np.float32([[0, 0], [width-1, 0], [width-1, height-1], [0, height-1]])
        crop = cv2.warpPerspective(frame, cv2.getPerspectiveTransform(polygon, target), (width, height))
        if crop.shape[0] > crop.shape[1]:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)
        return crop, polygon


class NumberConsensus:
    def __init__(self):
        self.text = None
        self.hits = 0
        self.first = self.last = None
        self.confirmed = False
        self.confidence = None

    def observe(self, text, confidence, t):
        if self.confirmed or (self.last is not None and t <= self.last):
            return self.confirmed
        normalized = re.sub(r"\s+", " ", text).strip().upper()
        valid = len(normalized) >= 2 and any(c.isdigit() for c in normalized) and confidence >= .85
        if not valid:
            self.text, self.hits, self.first = None, 0, None
        elif normalized != self.text or (self.last is not None and t-self.last > 15):
            self.text, self.hits, self.first = normalized, 1, t
            self.confidence = confidence
        else:
            self.hits += 1
            self.confidence = min(self.confidence, confidence)
        self.last = t
        self.confirmed = self.hits >= 3 and t-self.first >= 1.
        return self.confirmed


class PacketOCR:
    """One bounded background request; camera preview/recording never waits."""
    def __init__(self, root):
        self.root = Path(root)
        self.python = Path(os.environ.get("PACKET_OCR_PYTHON", self.root / ".venv-ocr/bin/python"))
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="packet-ocr")
        self.future = self.process = self.detector = None
        self.closed = False
        self.last_submit = -float('inf')
        self.consensus = NumberConsensus()
        self.result = dict(status="pending", text=None, confidence=None, confirmed_at=None,
                           method="pcnumber_obb_paddleocr", message="Rotate the packet to show its number.")

    def _read(self, frame, roi, t):
        if self.detector is None:
            self.detector = NumberDetector(self.root / "model onnx/pcnumber.onnx")
        crop, polygon = self.detector.crop(frame, roi)
        if crop is None:
            return dict(text="", confidence=0., t=t, crop=None, polygon=None)
        if not self.python.is_file():
            raise RuntimeError("OCR environment missing. Run setup_packet_ocr.sh.")
        if self.closed:
            return None
        if self.process is None:
            env = dict(os.environ, OMP_NUM_THREADS="2", PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True")
            self.process = subprocess.Popen([str(self.python), "-u", str(self.root / "standalone/ocr_service.py")],
                                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                            stderr=subprocess.DEVNULL, text=True, env=env)
        if self.closed:
            self.process.terminate()
            return None
        ok, encoded = cv2.imencode('.png', crop)
        if not ok:
            raise RuntimeError("Unable to encode packet number crop")
        request = json.dumps(dict(image=base64.b64encode(encoded).decode('ascii')))
        self.process.stdin.write(request+'\n')
        self.process.stdin.flush()
        # First request can download/load OCR weights; bounded and interruptible.
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            deadline = time.monotonic()+180
            while not self.closed:
                if time.monotonic() > deadline:
                    raise TimeoutError("PaddleOCR initialization/read timed out; finish to save evidence and retry.")
                if not selector.select(.2):
                    continue
                line = self.process.stdout.readline()
                if not line:
                    raise RuntimeError("PaddleOCR process stopped unexpectedly. Run the OCR self-check.")
                if not line.startswith('PACKET_OCR:'):
                    continue
                response = json.loads(line[len('PACKET_OCR:'):])
                if response.get('error'):
                    raise RuntimeError(response['error'])
                return dict(**response, t=t, crop=crop, polygon=polygon, scene=frame)
        return None

    def update(self, frame, roi, t):
        accepted = None
        if self.future is not None and self.future.done():
            try:
                read = self.future.result()
                if read is not None:
                    confirmed = self.consensus.observe(read['text'], read['confidence'], read['t'])
                    self.result.update(status="confirmed" if confirmed else "reading", text=self.consensus.text,
                                       confidence=self.consensus.confidence,
                                       message=("Packet number confirmed. Saving PDF and videos." if confirmed else
                                                "Rotate/show the packet number clearly; waiting for three matching reads."))
                    if confirmed:
                        self.result['confirmed_at'] = read['t']
                        accepted = read
            except Exception as exc:
                self.result.update(status="error", message=f"Packet sealed; OCR unavailable: {exc}. Finish / Save retains the session.")
            self.future = None
        if self.result['status'] not in ('confirmed', 'error') and self.future is None and t-self.last_submit >= .6:
            self.last_submit = t
            self.future = self.pool.submit(self._read, frame.copy(), list(roi), t)
            self.result['message'] = "Rotate the packet to show its number. Detecting/reading the number area…"
        return self.result, accepted

    def close(self):
        self.closed = True
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
        self.pool.shutdown(wait=True, cancel_futures=True)


def check_ocr(root):
    """Exercise the actual isolated interpreter, model downloads and recognition."""
    root = Path(root)
    detector = NumberDetector(root / 'model onnx/pcnumber.onnx')
    image = np.full((100, 520, 3), 255, np.uint8)
    cv2.putText(image, 'PC123456', (15, 72), cv2.FONT_HERSHEY_SIMPLEX, 2., (0, 0, 0), 3, cv2.LINE_AA)
    detector.crop(image, [(0, 0), (519, 0), (519, 99), (0, 99)])
    _, encoded = cv2.imencode('.png', image)
    python = os.environ.get('PACKET_OCR_PYTHON', str(root / '.venv-ocr/bin/python'))
    request = json.dumps(dict(image=base64.b64encode(encoded).decode('ascii')))+'\n'
    result = subprocess.run([python, '-u', str(root / 'standalone/ocr_service.py')],
                            input=request, capture_output=True, text=True, timeout=180)
    messages = [json.loads(line[len('PACKET_OCR:'):]) for line in result.stdout.splitlines()
                if line.startswith('PACKET_OCR:')]
    if not messages or messages[-1].get('error'):
        raise RuntimeError(str(messages[-1] if messages else result.stderr[-4000:]))
    read = messages[-1]
    if '123456' not in read['text'] or read['confidence'] < .85:
        raise RuntimeError(f'OCR self-check failed: {read}')
    print(f"PASS: pcnumber.onnx CPU inference and isolated PaddleOCR: {read['text']} ({read['confidence']:.1%})")

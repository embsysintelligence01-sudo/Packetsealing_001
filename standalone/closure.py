"""Explicitly configured packet-closure classifier; missing weights never pass."""
from dataclasses import dataclass
import json
from pathlib import Path

import cv2
import numpy as np
from .models import session


@dataclass(frozen=True)
class ClosureResult:
    status: str
    confidence: float | None = None
    message: str = ""


class ClosureClassifier:
    def __init__(self, config_path):
        self.ready = False
        self.problem = "Closure model required: provide a trained open/closed/unclear ONNX classifier."
        self.error_status = "model_missing"
        path = Path(config_path)
        if not path.is_file():
            return
        try:
            config = json.loads(path.read_text())
            model_path = path.parent / config["model"]
            if not model_path.is_file():
                self.problem = f"Closure model weights not found: {model_path.name}"
                return
            self.labels = config["classes"]
            if sorted(self.labels) != ["closed", "open", "unclear"]:
                raise ValueError("Closure classes must contain open, closed, unclear exactly once in output order.")
            self.threshold = float(config["confidence"])
            self.mean = np.asarray(config["mean"], np.float32)
            self.std = np.asarray(config["std"], np.float32)
            self.output_kind = config["output"]
            if not .5 <= self.threshold <= 1 or self.mean.shape != (3,) or self.std.shape != (3,) or np.any(self.std <= 0):
                raise ValueError("Invalid closure confidence or normalization.")
            if not np.isfinite(self.mean).all() or not np.isfinite(self.std).all() or self.output_kind not in ("logits", "probabilities"):
                raise ValueError("Invalid closure normalization/output contract.")
            self.session = session(model_path)
            inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
            shape = inputs[0].shape
            if len(inputs) != 1 or len(shape) != 4 or shape[1] != 3 or inputs[0].type != "tensor(float)":
                raise ValueError("Closure input must be one float32 NCHW RGB tensor.")
            if shape[0] not in (1, "batch", None) or any(not isinstance(v, int) or v < 16 for v in shape[2:]):
                raise ValueError("Closure input needs batch 1 and fixed image dimensions.")
            if len(outputs) != 1 or len(outputs[0].shape) != 2 or outputs[0].shape[1] != 3:
                raise ValueError("Closure output must be [batch, 3].")
            self.input_name = inputs[0].name
            self.size = (shape[3], shape[2])
            self.ready = True
        except Exception as exc:
            self.problem = f"Closure model configuration error: {exc}"
            self.error_status = "model_error"

    def classify(self, frame, bag):
        if not self.ready:
            return ClosureResult(self.error_status, message=self.problem)
        if bag is None:
            return ClosureResult("unclear", message="Packet is not available for closure verification.")
        try:
            x1, y1, x2, y2 = map(int, bag.box)
            h, w = frame.shape[:2]
            crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
            if crop.size == 0:
                return ClosureResult("unclear", message="Packet crop is empty.")
            rgb = cv2.cvtColor(cv2.resize(crop, self.size), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
            tensor = np.ascontiguousarray(((rgb - self.mean) / self.std).transpose(2, 0, 1)[None])
            values = np.asarray(self.session.run(None, {self.input_name: tensor})[0], np.float32)
            if values.shape != (1, 3) or not np.isfinite(values).all():
                raise ValueError("Closure output is not finite [1,3].")
            values = values[0]
            if self.output_kind == "logits":
                values = np.exp(values - values.max())
                values /= values.sum()
            elif np.any(values < 0) or np.any(values > 1) or abs(float(values.sum()) - 1.) > .01:
                raise ValueError("Invalid closure probabilities.")
            index = int(values.argmax())
            confidence = float(values[index])
            label = self.labels[index] if confidence >= self.threshold else "unclear"
            return ClosureResult(label, confidence, f"Closure: {label} ({confidence:.0%}).")
        except Exception as exc:
            return ClosureResult("model_error", message=f"Closure inference failed: {exc}")

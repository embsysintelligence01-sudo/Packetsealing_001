"""CPU ONNX segmentation and the original packet false-positive filters."""
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from torch import nn
from ultralytics.data.augment import LetterBox
from ultralytics.utils import nms, ops


def session(path):
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


@dataclass
class Detection:
    mask: np.ndarray
    box: tuple
    confidence: float


class Segmenter:
    def __init__(self, path, confidence):
        self.session = session(path)
        self.input = self.session.get_inputs()[0].name
        if self.session.get_inputs()[0].shape != [1, 3, 320, 320]:
            raise ValueError(f"Unsupported segmentation input: {path}")
        self.confidence = confidence
        self.letterbox = LetterBox((320, 320), auto=False)

    def detect(self, frame):
        padded = self.letterbox(image=frame)
        x = np.ascontiguousarray(padded[:, :, ::-1].transpose(2, 0, 1)[None], dtype=np.float32) / 255.
        raw, proto = self.session.run(None, {self.input: x})
        prediction = nms.non_max_suppression(torch.from_numpy(raw), self.confidence, .45, nc=1, max_det=12)[0]
        if not len(prediction):
            return []
        prediction[:, :4] = ops.scale_boxes((320, 320), prediction[:, :4], frame.shape)
        masks = ops.process_mask_native(torch.from_numpy(proto[0]), prediction[:, 6:],
                                        prediction[:, :4], frame.shape[:2]).numpy()
        return [Detection(m.astype(np.uint8) * 255, tuple(p[:4].numpy().astype(int)), float(p[4]))
                for p, m in zip(prediction, masks) if np.count_nonzero(m) >= 25]


class StripCNN(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for a, b in ((3, 16), (16, 32), (32, 64)):
            layers.extend((nn.Conv2d(a, b, 3, padding=1), nn.BatchNorm2d(b), nn.ReLU(), nn.MaxPool2d(2)))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64 * 8 * 8, 128), nn.ReLU(),
                                  nn.Dropout(.4), nn.Linear(128, 2))

    def forward(self, x):
        return self.head(self.features(x))


class PacketModels:
    def __init__(self, directory: Path, bag_tp_class=1):
        self.bag = Segmenter(directory / "bagnewmodel.onnx", .5)
        self.strip = Segmenter(directory / "stripmodel.onnx", .4)
        self.hand = Segmenter(directory / "handseg model.onnx", .3)
        self.bag_filter = session(directory / "fp_filter_resnet18_bag.onnx")
        self.bag_input = self.bag_filter.get_inputs()[0].name
        self.bag_tp_class = bag_tp_class  # Original FPFilter uses class 1 = true positive.
        self.strip_filter = StripCNN()
        self.strip_filter.load_state_dict(torch.load(directory / "hsv_fp_filter_strip.pt",
                                                     map_location="cpu", weights_only=True))
        self.strip_filter.eval()

    def filter_bag(self, crop):
        rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        rgb = (rgb - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        logits = self.bag_filter.run(None, {self.bag_input: np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])})[0][0]
        probabilities = np.exp(logits - logits.max())
        return float(probabilities[self.bag_tp_class] / probabilities.sum())

    def filter_strip(self, crop):
        hsv = cv2.cvtColor(cv2.resize(crop, (64, 64)), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv /= np.array([180., 255., 255.], np.float32)
        with torch.inference_mode():
            logits = self.strip_filter(torch.from_numpy(hsv.transpose(2, 0, 1).copy())[None])
            return float(logits.softmax(1)[0, 1])

    @staticmethod
    def crop(frame, detection):
        x1, y1, x2, y2 = detection.box
        return frame[y1:y2, x1:x2]

    def inspect(self, frame, roi_mask, context_mask=None):
        # All exports infer at 320px. Decode their masks at a bounded working
        # resolution, instead of running torch mask decoding at camera 1440p.
        height, width = frame.shape[:2]
        if width <= 960:
            return self._inspect_native(frame, roi_mask, context_mask) if context_mask is not None else self._inspect_native(frame, roi_mask)
        size = (960, max(1, round(height * 960 / width)))
        small = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        roi = cv2.resize(roi_mask, size, interpolation=cv2.INTER_NEAREST)
        context = cv2.resize(context_mask, size, interpolation=cv2.INTER_NEAREST) if context_mask is not None else None
        bag, strip, clear, strip_mask, hand_mask = (self._inspect_native(small, roi, context) if context is not None
                                                   else self._inspect_native(small, roi))
        if bag is not None:
            sx, sy = width / size[0], height / size[1]
            bag = Detection(cv2.resize(bag.mask, (width, height), interpolation=cv2.INTER_NEAREST),
                            tuple(round(v * scale) for v, scale in zip(bag.box, (sx, sy, sx, sy))), bag.confidence)
        return (bag, strip, clear,
                cv2.resize(strip_mask, (width, height), interpolation=cv2.INTER_NEAREST),
                cv2.resize(hand_mask, (width, height), interpolation=cv2.INTER_NEAREST))

    def _inspect_native(self, frame, roi_mask, context_mask=None):
        bags = self.bag.detect(frame)
        accepted = []
        for d in bags:
            if np.count_nonzero((d.mask > 0) & (roi_mask > 0)) < .9 * np.count_nonzero(d.mask):
                continue
            crop = self.crop(frame, d)
            if crop.size and self.filter_bag(crop) >= .6:
                accepted.append(d)
        bag = accepted[0] if len(accepted) == 1 else None
        strips = self.strip.detect(frame)
        inspection_mask = bag.mask if bag is not None else context_mask
        strip_state = "absent" if inspection_mask is not None else "unknown"
        strip_mask = np.zeros(frame.shape[:2], np.uint8)
        for d in strips:
            # A rejected raw candidate is uncertainty, never proof of removal.
            if inspection_mask is None:
                strip_state = "uncertain"
                continue
            if np.count_nonzero((d.mask > 0) & (inspection_mask > 0)) < .6 * np.count_nonzero(d.mask):
                continue
            crop = self.crop(frame, d)
            probability = self.filter_strip(crop) if crop.size else 0.
            if probability >= .6:
                strip_mask |= d.mask
                strip_state = "present"
            elif strip_state != "present":
                strip_state = "uncertain"
        hands = self.hand.detect(frame)
        hand_mask = np.zeros(frame.shape[:2], np.uint8)
        for d in hands:
            hand_mask |= d.mask & roi_mask
        # A hand/fixture at the other end of the bench does not cover this packet.
        scope = roi_mask
        if inspection_mask is not None:
            scope = cv2.dilate(inspection_mask, np.ones((25, 25), np.uint8)) & roi_mask
        hands_clear = np.count_nonzero((hand_mask > 0) & (scope > 0)) < max(100, np.count_nonzero(scope) * .001)
        return bag, strip_state, hands_clear, strip_mask, hand_mask


def mask_iou(a, b):
    if a is None or b is None or a.shape != b.shape:
        return 0.
    return np.count_nonzero((a > 0) & (b > 0)) / max(1, np.count_nonzero((a > 0) | (b > 0)))

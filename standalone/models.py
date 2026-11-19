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


class PacketBagDetector:
    """Packet-only inference shared by production inspection and mouth collection."""
    bag_filter_threshold = .20

    def __init__(self, directory: Path, bag_tp_class=1, bag_filter_threshold=.20):
        if not 0 < bag_filter_threshold <= 1:
            raise ValueError('Bag filter threshold must be greater than zero and at most one')
        self.bag_filter_threshold = float(bag_filter_threshold)
        self.bag = Segmenter(directory / "bagnewmodel.onnx", .5)
        self.bag_filter = session(directory / "fp_filter_resnet18_bag.onnx")
        self.bag_input = self.bag_filter.get_inputs()[0].name
        self.bag_tp_class = bag_tp_class

    def detect(self, frame):
        accepted = []
        for detection in self.bag.detect(frame):
            x1,y1,x2,y2 = detection.box
            x1,y1 = max(0,x1),max(0,y1)
            x2,y2 = min(frame.shape[1],x2),min(frame.shape[0],y2)
            crop = frame[y1:y2,x1:x2]
            if crop.size and self.filter_bag(crop) >= self.bag_filter_threshold:
                accepted.append(Detection(detection.mask, tuple(map(int,(x1,y1,x2,y2))), detection.confidence))
        return accepted

    def filter_bag(self, crop):
        rgb = cv2.cvtColor(cv2.resize(crop, (224, 224)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.
        rgb = (rgb - np.array([.485, .456, .406], np.float32)) / np.array([.229, .224, .225], np.float32)
        logits = self.bag_filter.run(None, {self.bag_input: np.ascontiguousarray(rgb.transpose(2, 0, 1)[None])})[0][0]
        probabilities = np.exp(logits - logits.max())
        return float(probabilities[self.bag_tp_class] / probabilities.sum())

class PacketModels(PacketBagDetector):
    def __init__(self, directory: Path, bag_tp_class=1, bag_filter_threshold=.20):
        super().__init__(directory, bag_tp_class, bag_filter_threshold)
        self.strip = Segmenter(directory / "stripmodel.onnx", .4)
        self.hand = Segmenter(directory / "handseg model.onnx", .3)
        self.strip_filter = StripCNN()
        self.strip_filter.load_state_dict(torch.load(directory / "hsv_fp_filter_strip.pt",
                                                     map_location="cpu", weights_only=True))
        self.strip_filter.eval()

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

    def inspect(self, frame, roi_mask, context_mask=None, inspection_zone=None):
        # Keep the source image and filter crops at camera resolution. Each
        # Segmenter letterboxes directly from this image to its 320x320 input.
        return self._inspect_native(frame, roi_mask, context_mask, inspection_zone)

    def _inspect_native(self, frame, roi_mask, context_mask=None, inspection_zone=None):
        bags = self.bag.detect(frame)
        accepted = []
        bag_candidates = []
        for d in bags:
            overlap = float(np.count_nonzero((d.mask > 0) & (roi_mask > 0)) / max(1, np.count_nonzero(d.mask)))
            candidate = dict(confidence=float(d.confidence), bench_roi_overlap=overlap)
            bag_candidates.append(candidate)
            if overlap < .9:
                candidate['decision'] = 'outside_bench_roi'
                continue
            crop = self.crop(frame, d)
            probability = self.filter_bag(crop) if crop.size else 0.
            candidate.update(filter_probability=float(probability),
                decision='accepted' if crop.size and probability >= self.bag_filter_threshold else 'filter_rejected')
            if crop.size and probability >= self.bag_filter_threshold:
                accepted.append(d)
        bag = accepted[0] if len(accepted) == 1 else None
        strips = self.strip.detect(frame)
        candidates = []
        inspection_mask = bag.mask if bag is not None else context_mask
        if inspection_zone is not None:
            inspection_mask = inspection_zone
        strip_state = "absent" if inspection_mask is not None else "unknown"
        strip_mask = np.zeros(frame.shape[:2], np.uint8)
        for d in strips:
            # A rejected raw candidate is uncertainty, never proof of removal.
            if inspection_mask is None:
                strip_state = "uncertain"
                continue
            overlap = float(np.count_nonzero((d.mask > 0) & (inspection_mask > 0)) / max(1, np.count_nonzero(d.mask)))
            candidate = dict(confidence=float(d.confidence), inspection_overlap=overlap)
            candidates.append(candidate)
            if overlap < .6:
                candidate['decision'] = 'outside_attachment_area' if inspection_zone is not None else 'outside_packet'
                continue
            crop = self.crop(frame, d)
            probability = self.filter_strip(crop) if crop.size else 0.
            candidate.update(filter_probability=float(probability), decision='accepted' if probability >= .6 else 'uncertain')
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
        hand_pixels = int(np.count_nonzero((hand_mask > 0) & (scope > 0)))
        hand_limit = max(100, np.count_nonzero(scope) * .001)
        hands_clear = hand_pixels < hand_limit
        self.last_diagnostic = dict(inspection_scope='tracked_attachment_area' if inspection_zone is not None else 'full_packet',
            bag_filter_threshold=self.bag_filter_threshold, bag_detector_threshold=getattr(self.bag, 'confidence', .5) if isinstance(getattr(self.bag, 'confidence', .5), (int, float)) else .5,
            raw_bag_candidates=len(bags), bag_candidates=bag_candidates,
            raw_strip_candidates=len(strips), strip_candidates=candidates, hand_pixels_in_scope=hand_pixels,
            hand_pixel_limit=float(hand_limit), accepted_packet_count=len(accepted))
        return bag, strip_state, hands_clear, strip_mask, hand_mask


def mask_iou(a, b):
    if a is None or b is None or a.shape != b.shape:
        return 0.
    return np.count_nonzero((a > 0) & (b > 0)) / max(1, np.count_nonzero((a > 0) | (b > 0)))

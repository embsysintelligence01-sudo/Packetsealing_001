"""CPU-only packet mouth ROI tracker, HOG features, and guarded temporal decisions.

A tracked ROI is NOT proof of an open mouth or jewellery insertion.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Optional

import cv2
import numpy as np

LABELS = ("OPEN_CAMERA", "OPEN_SIDE", "CLOSED_FOLDED", "UNCLEAR")


def to_working_size(frame: np.ndarray, max_width: int = 960) -> np.ndarray:
    h, w = frame.shape[:2]
    if w > max_width:
        return cv2.resize(frame, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)
    return frame


def clamp_roi(roi, frame_shape):
    h, w = frame_shape[:2]
    x, y, rw, rh = [int(round(z)) for z in roi]
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(w, x + rw), min(h, y + rh)
    if x2-x1 < 22 or y2-y1 < 22:
        return None
    return x1, y1, x2-x1, y2-y1


def crop_roi(frame, roi):
    rect = clamp_roi(roi, frame.shape) if roi is not None else None
    if rect is None:
        return None
    x, y, w, h = rect
    return frame[y:y+h, x:x+w]


@dataclass
class TrackState:
    roi: Optional[tuple[int, int, int, int]] = None
    status: str = "UNSEEDED"
    quality: float = 0.0
    inliers: int = 0
    reason: str = "Draw a rectangle around the actual mouth."


class MouthROITracker:
    """Seed ONCE at a visible mouth, then track nearby texture across frames.

    Correctly exposes TRACK_LOST instead of silently reusing stale mouth geometry.
    Every coordinate is in working-frame pixels, not source-video pixels.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.gray = None
        self.points = None
        self.corners = None
        self.state = TrackState()

    def _gray(self, frame):
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _points(self, gray, roi):
        mask = np.zeros(gray.shape, dtype=np.uint8)
        x, y, w, h = roi
        pad = min(5, max(1, min(w, h) // 10))
        cv2.rectangle(mask, (x+pad,y+pad), (x+w-pad,y+h-pad), 255, -1)
        return cv2.goodFeaturesToTrack(gray, maxCorners=130, qualityLevel=.008,
                                       minDistance=4, blockSize=3, mask=mask)

    def seed(self, frame, roi):
        self.reset()
        rect = clamp_roi(roi, frame.shape)
        if rect is None:
            self.state = TrackState(status="INVALID_ROI", reason="Mouth selection too small or outside image.")
            return self.state
        self.gray = self._gray(frame)
        x, y, w, h = rect
        self.corners = np.float32([[x,y],[x+w,y],[x+w,y+h],[x,y+h]]).reshape(-1,1,2)
        self.points = self._points(self.gray, rect)
        n = 0 if self.points is None else len(self.points)
        self.state = TrackState(rect, "SEEDED", 0.5 if n >= 8 else 0., n,
                                "Mouth seeded. Tracking is not evidence of an open mouth." if n >= 8 else
                                "Too few textured features: tracking may fail; choose a wider mouth ROI.")
        return self.state

    def update(self, frame):
        if self.state.roi is None or self.gray is None:
            return self.state
        if self.state.status == "TRACK_LOST":
            return self.state
        new_gray = self._gray(frame)
        if self.gray.shape != new_gray.shape:
            self.state.status, self.state.reason = "TRACK_LOST", "Frame resolution changed. Draw the mouth ROI again."
            return self.state
        if self.points is None or len(self.points) < 8:
            self.state.status, self.state.reason = "TRACK_LOST", "Too few points. Manually reselect the mouth."
            return self.state
        p1, s1, _ = cv2.calcOpticalFlowPyrLK(self.gray, new_gray, self.points, None,
                                              winSize=(21,21), maxLevel=3,
                                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 24, .02))
        if p1 is None or s1 is None:
            self.state.status, self.state.reason = "TRACK_LOST", "Optical flow unavailable."
            return self.state
        p0_back, s2, _ = cv2.calcOpticalFlowPyrLK(new_gray, self.gray, p1, None,
                                                   winSize=(21,21), maxLevel=3)
        if p0_back is None or s2 is None:
            self.state.status, self.state.reason = "TRACK_LOST", "Reverse-flow validation failed."
            return self.state
        err = np.linalg.norm(self.points.reshape(-1,2) - p0_back.reshape(-1,2), axis=1)
        valid = ((s1.reshape(-1)>0) & (s2.reshape(-1)>0) & (err < 1.8))
        if np.count_nonzero(valid) < 8:
            self.state.status, self.state.reason = "TRACK_LOST", "Mouth texture lost or covered. Reselect the ROI."
            return self.state
        before = self.points.reshape(-1,2)[valid]
        after = p1.reshape(-1,2)[valid]
        M, inlier_mask = cv2.estimateAffinePartial2D(before, after, method=cv2.RANSAC, ransacReprojThreshold=2.5)
        if M is None or inlier_mask is None:
            self.state.status, self.state.reason = "TRACK_LOST", "Could not estimate stable mouth motion."
            return self.state
        inliers = inlier_mask.reshape(-1).astype(bool)
        n = int(np.sum(inliers))
        q = float(n / len(before))
        scale = math.hypot(float(M[0,0]), float(M[0,1]))
        shift = math.hypot(float(M[0,2]), float(M[1,2]))
        if n < 8 or q < .55 or not .78 <= scale <= 1.3 or shift > frame.shape[1]*.20:
            self.state.status, self.state.reason = "TRACK_LOST", "Unstable geometry (fold, hand or large turn). Reselect mouth."
            return self.state
        # Transform the original quadrilateral, not its expanding axis-aligned box.
        new_corners = cv2.transform(self.corners, M).reshape(-1,2)
        rx,ry,rw,rh = cv2.boundingRect(new_corners.astype(np.float32))
        next_roi = clamp_roi((rx,ry,rw,rh), frame.shape)
        if next_roi is None:
            self.state.status, self.state.reason = "TRACK_LOST", "Tracked mouth moved off-screen."
            return self.state
        self.gray = new_gray
        self.corners = new_corners.reshape(-1,1,2)
        self.points = self._points(new_gray, next_roi)
        if self.points is None or len(self.points) < 8:
            self.state.status, self.state.reason = "TRACK_LOST", "New mouth region has insufficient trackable details."
            return self.state
        self.state = TrackState(next_roi, "TRACKED", q, n, "Tracking current mouth candidate (not opening proof).")
        return self.state


_HOG = cv2.HOGDescriptor((64,64), (16,16), (8,8), (8,8), 9)


def mouth_features(image):
    """HOG + colour histogram + edge summaries of manually selected mouth crop."""
    if image is None or image.size == 0:
        raise ValueError("Empty mouth crop")
    bgr = cv2.resize(image, (64,64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hog = _HOG.compute(gray).reshape(-1).astype(np.float32)
    hist = cv2.calcHist([hsv],[0,1],None,[12,8],[0,180,0,256]).reshape(-1).astype(np.float32)
    hist /= max(1, hist.sum())
    edges = cv2.Canny(gray, 65, 155)
    metrics = np.array([gray.mean()/255, gray.std()/128, np.mean(edges > 0),
                        np.mean(gray < 70), np.mean(hsv[:,:,1] > 80)], np.float32)
    return np.concatenate([hog, hist, metrics])


def diagnostic_metrics(image):
    if image is None:
        return {}
    gray = cv2.cvtColor(cv2.resize(image,(192,128)),cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,65,155)
    return {'edge_density':round(float(np.mean(edges>0)),3),
            'dark_fraction':round(float(np.mean(gray<70)),3),
            'brightness':round(float(np.mean(gray)/255),3)}


def visible_gap_candidate(image):
    """Model-free dark-gap proposal within an operator-selected mouth crop.

    Measures local contrast and a two-dimensional interior dark contour. This
    cannot distinguish every printed mark, shadow or fold from a physical hole.
    A negative result means UNCLEAR, never proof of a closed/sealed packet.
    Returned contour coordinates refer to the supplied crop, not the full frame.
    """
    result = {'label': 'UNCLEAR', 'reason': 'No distinct interior gap visible.',
              'contour': None, 'contrast': 0.0, 'area_fraction': 0.0}
    if image is None or image.size == 0 or min(image.shape[:2]) < 22:
        return result
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)
    low, high = np.percentile(gray, (2, 85))
    if high-low < 30:
        result['reason'] = 'Insufficient contrast to locate a visible gap.'
        return result
    threshold = min(100., float(low + .35*(high-low)))
    mask = np.uint8(gray < threshold)*255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    height, width = gray.shape
    best_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        fraction = area / gray.size
        x,y,w,h = cv2.boundingRect(contour)
        if not .025 <= fraction <= .55 or x <= 1 or y <= 1 or x+w >= width-1 or y+h >= height-1:
            continue
        (_, _), (a,b), _ = cv2.minAreaRect(contour)
        if min(a,b) < max(4, min(width,height)*.07) or max(a,b)/max(1,min(a,b)) > 9:
            continue
        inside = np.zeros_like(gray)
        cv2.drawContours(inside, [contour], -1, 255, -1)
        ring = cv2.dilate(inside, np.ones((9,9),np.uint8)) - inside
        if not np.any(ring):
            continue
        contrast = float(gray[ring > 0].mean()-gray[inside > 0].mean())
        if contrast >= 25 and area > best_area:
            best_area = area
            result = {'label': 'GAP_CANDIDATE',
                      'reason': 'Visible dark gap candidate; folds, print and shadows can imitate it.',
                      'contour': contour.reshape(-1,2).tolist(),
                      'contrast': round(contrast,2), 'area_fraction': round(fraction,4)}
    return result


class TemporalConsensus:
    """Consecutive independent frames, separated by timestamps, required for confirmation."""
    def __init__(self, min_samples=3, min_span=.4, min_prob=.78, max_gap=1.1):
        self.min_samples = min_samples
        self.min_span = min_span
        self.min_prob = min_prob
        self.max_gap = max_gap
        self.reset()

    def reset(self):
        self.samples = deque(maxlen=6)
        self.last_t = None

    def add(self, label, probability, timestamp):
        if self.last_t is not None and timestamp <= self.last_t:
            return "UNCONFIRMED"
        if self.last_t is not None and timestamp-self.last_t > self.max_gap:
            self.samples.clear()
        if probability < self.min_prob or label == 'UNCLEAR':
            self.samples.clear()
            self.last_t = timestamp
            return "UNCONFIRMED"
        if self.samples and self.samples[-1][1] != label:
            self.samples.clear()
        # Process an independent observation at no more than 10 Hz.
        if not self.samples or timestamp-self.samples[-1][0] >= .095:
            self.samples.append((timestamp,label,probability))
        self.last_t = timestamp
        if len(self.samples) >= self.min_samples and self.samples[-1][0]-self.samples[0][0] >= self.min_span:
            return f'{label}_CONSISTENT'
        return "UNCONFIRMED"

"""Stable full-bag ROI; no manually drawn opening or fixed mouth geometry."""
from .models import mask_iou
import numpy as np
import cv2
import math


class PacketRegion:
    def __init__(self):
        self.previous = None
        self.reference = None
        self.stable = 0
        self.last_good = None
        self.loss_reported = False
        self.stationary = False
        self.still_since = None
        self.still_reference = None

    def _stationary(self, mask, t, valid, hands_clear):
        if not valid:
            self.stationary, self.still_since, self.still_reference = False, None, None
            return
        moments = cv2.moments(mask, binaryImage=True)
        area = moments["m00"]
        if area <= 0:
            self.stationary = False
            return
        center = (moments["m10"] / area, moments["m01"] / area)
        _, _, width, height = cv2.boundingRect(mask)
        tolerance = max(3., max(width, height) * .01)
        if self.still_reference is None or math.dist(center, self.still_reference[0]) > tolerance or abs(area / self.still_reference[1] - 1.) > .08:
            self.still_reference, self.still_since = (center, area), t
        self.stationary = t - self.still_since >= .7

    def update(self, mask, t, hands_clear):
        if mask is None:
            self._stationary(None, t, False, hands_clear)
            self.previous, self.stable = None, 0
            timeout = 3. if hands_clear else 10.
            changed = self.last_good is not None and t - self.last_good > timeout and not self.loss_reported
            if changed:
                self.loss_reported = True
            return False, changed
        self.stable = self.stable + 1 if mask_iou(self.previous, mask) >= .65 else 1
        self.previous = mask.copy()
        if self.reference is None:
            if self.stable < 3:
                self._stationary(mask, t, False, hands_clear)
                return False, False
            self.reference = mask.copy()
        intersection = np.count_nonzero((self.reference > 0) & (mask > 0))
        areas = (np.count_nonzero(self.reference), np.count_nonzero(mask))
        # Folding the full, trained bag changes its silhouette. A stable mask
        # contained in the same footprint may still be that bag, not a swap.
        contained_fold = (intersection / max(1, min(areas)) >= .85 and
                          min(areas) / max(1, max(areas)) >= .25)
        matches = mask_iou(self.reference, mask) >= .5 or contained_fold
        changed = not matches and self.stable >= 3
        if changed:
            self.reference = mask.copy()
            matches = True
        valid = matches and self.stable >= 2
        if valid:
            self.last_good = t
            self.loss_reported = False
        self._stationary(mask, t, valid, hands_clear)
        return valid, changed

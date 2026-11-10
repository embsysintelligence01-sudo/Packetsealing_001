"""Keep the armed packet's visual context without requiring full-bag detection."""
import cv2
import numpy as np
from .models import Detection


class SealContext:
    def __init__(self):
        self.orb = cv2.ORB_create(nfeatures=900, edgeThreshold=10, fastThreshold=10)
        self.points = self.descriptors = self.mask = self.zone = None
        self.size = None
        self.current = None
        self.strip_view_ok = False

    @property
    def ready(self):
        return self.descriptors is not None

    def arm(self, frame, bag, strip_mask):
        if self.ready or bag is None:
            return
        h, w = frame.shape[:2]
        self.size = (min(640, w), max(1, round(h*min(640, w)/w)))
        gray = cv2.cvtColor(cv2.resize(frame, self.size), cv2.COLOR_BGR2GRAY)
        mask = cv2.resize(bag.mask, self.size, interpolation=cv2.INTER_NEAREST)
        feature_mask = cv2.erode(mask, np.ones((5, 5), np.uint8))
        zone = cv2.resize(strip_mask, self.size, interpolation=cv2.INTER_NEAREST)
        if not np.any(zone):
            return  # Arming needs an actually observed strip, not an empty ROI.
        zone = cv2.dilate(zone, np.ones((13, 13), np.uint8))
        feature_mask[zone > 0] = 0  # Removing the strip must not remove our reference features.
        keys, descriptors = self.orb.detectAndCompute(gray, feature_mask)
        if descriptors is not None and len(keys) >= 10:
            self.points = np.float32([key.pt for key in keys])
            self.descriptors, self.mask, self.zone = descriptors, mask, zone

    def locate(self, frame):
        self.current, self.strip_view_ok = None, False
        if not self.ready:
            return None
        gray = cv2.cvtColor(cv2.resize(frame, self.size), cv2.COLOR_BGR2GRAY)
        keys, descriptors = self.orb.detectAndCompute(gray, None)
        if descriptors is None or len(keys) < 10:
            return None
        pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(self.descriptors, descriptors, k=2)
        matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .72*pair[1].distance]
        if len(matches) < 10:
            return None
        source = np.float32([self.points[m.queryIdx] for m in matches])
        target = np.float32([keys[m.trainIdx].pt for m in matches])
        matrix, inliers = cv2.estimateAffinePartial2D(source, target, method=cv2.RANSAC, ransacReprojThreshold=2.5)
        if matrix is None or inliers is None or inliers.sum() < 8 or inliers.mean() < .6:
            return None
        scale = np.hypot(matrix[0, 0], matrix[0, 1])
        if not .5 <= scale <= 1.8:
            return None
        mask = cv2.warpAffine(self.mask, matrix, self.size, flags=cv2.INTER_NEAREST)
        zone = cv2.warpAffine(self.zone, matrix, self.size, flags=cv2.INTER_NEAREST)
        # Only the strip inspection zone needs to stay in view, not the full bag.
        expected = max(1., np.count_nonzero(self.zone)*scale*scale)
        self.strip_view_ok = bool(np.count_nonzero(zone) / expected >= .8)
        if np.count_nonzero(mask) < 100:
            return None
        h, w = frame.shape[:2]
        full = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        x, y, bw, bh = cv2.boundingRect(full)
        self.current = Detection(full, (x, y, x+bw, y+bh), 0.)
        return self.current

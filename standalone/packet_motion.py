"""Follow a moving packet between detector hits, for transfer context only."""
import cv2
import numpy as np

from .models import Detection


class PacketMotion:
    def __init__(self, max_gap=2.5):
        self.max_gap = max_gap
        self.gray = self.mask = None
        self.last_detection = self.last_t = None
        self.current = None
        self.source = 'unavailable'
        self.last_bag = None

    def update(self, frame, bag, t):
        height, width = frame.shape[:2]
        size = (min(width, 640), max(1, round(height * min(width, 640) / width)))
        gray = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2GRAY)
        mask = None
        self.current, self.source = None, 'unavailable'
        if bag is not None:
            mask = cv2.resize(bag.mask, size, interpolation=cv2.INTER_NEAREST)
            self.last_detection = t
            self.current, self.source = bag, 'model'
            self.last_bag = bag
        elif (self.gray is not None and self.gray.shape == gray.shape and self.mask is not None
              and self.last_t is not None and 0 < t-self.last_t <= 1.5
              and self.last_detection is not None and t-self.last_detection <= self.max_gap):
            points = cv2.goodFeaturesToTrack(self.gray, 120, .01, 5, mask=self.mask)
            if points is not None and len(points) >= 8:
                target, ok, _ = cv2.calcOpticalFlowPyrLK(self.gray, gray, points, None)
                if target is not None:
                    back, back_ok, _ = cv2.calcOpticalFlowPyrLK(gray, self.gray, target, None)
                    if back is not None:
                        good = (ok.ravel() > 0) & (back_ok.ravel() > 0)
                        good &= np.linalg.norm(back-points, axis=2).ravel() < 1.5
                        if good.sum() >= 8:
                            matrix, inliers = cv2.estimateAffinePartial2D(points[good], target[good],
                                                                         method=cv2.RANSAC, ransacReprojThreshold=2.)
                            if matrix is not None and inliers is not None:
                                scale = np.hypot(matrix[0, 0], matrix[0, 1])
                                shift = np.hypot(matrix[0, 2], matrix[1, 2])
                                if .65 <= scale <= 1.5 and inliers.mean() >= .65 and shift <= size[0] * .3:
                                    mask = cv2.warpAffine(self.mask, matrix, size, flags=cv2.INTER_NEAREST)
                                    if np.count_nonzero(mask) >= 100:
                                        full = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
                                        x, y, w, h = cv2.boundingRect(full)
                                        self.current = Detection(full, (x, y, x+w, y+h), 0.)
                                        self.source = 'motion'
                                    else:
                                        mask = None
        if (self.current is None and self.last_bag is not None and self.last_detection is not None
                and 0 < t-self.last_detection <= .45 and self.last_bag.mask.shape == (height, width)):
            # A single detector/flow miss at a fast insertion must not erase
            # the just-observed opening. Never used for strip/closure proof.
            self.current, self.source = self.last_bag, 'recent'
        self.gray, self.mask, self.last_t = gray, mask, t
        return self.current

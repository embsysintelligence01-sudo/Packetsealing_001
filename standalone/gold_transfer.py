"""Pickup-owned gold motion, independent of the hand model and SLIC identity."""
import cv2
import numpy as np

from .flow import Observation


def interaction_region(bag):
    """The full packet footprint includes its opening, not only visible film."""
    if bag is None:
        return None
    contours, _ = cv2.findContours(bag.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    mask = np.zeros_like(bag.mask)
    hull = cv2.convexHull(np.concatenate(contours))
    cv2.fillConvexPoly(mask, hull, 255)
    _, _, w, h = cv2.boundingRect(mask)
    margin = max(3, round(max(w, h)*.04))
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*margin+1, 2*margin+1)))
    from types import SimpleNamespace
    return SimpleNamespace(mask=mask)


def packet_position(point, bag, *, allow_box=False):
    if bag is None:
        return None
    x, y = map(int, point)
    h, w = bag.mask.shape[:2]
    if not (0 <= x < w and 0 <= y < h) or (not allow_box and not bag.mask[y, x]):
        return None
    bx, by, bw, bh = cv2.boundingRect(bag.mask)
    if not (bx <= x < bx+bw and by <= y < by+bh):
        return None
    return ((x-bx)/max(1, bw), (y-by)/max(1, bh))


def gold_overlap(detection, bag):
    if bag is None or getattr(detection, 'mask', None) is None:
        return 0.
    mask = detection.mask
    ox, oy = map(int, detection.origin)
    h, w = bag.mask.shape[:2]
    x1, y1, x2, y2 = max(0, ox), max(0, oy), min(w, ox+mask.shape[1]), min(h, oy+mask.shape[0])
    if x2 <= x1 or y2 <= y1:
        return 0.
    local = mask[y1-oy:y2-oy, x1-ox:x2-ox] > 0
    return np.count_nonzero(local & (bag.mask[y1:y2, x1:x2] > 0)) / max(1, np.count_nonzero(mask))


class GoldTransfer:
    def __init__(self, owner, anchor, t):
        self.owner = owner
        self.anchor = np.array(anchor, dtype=float)
        self.point = self.anchor.copy()
        self.last_t = t
        self.velocity = np.zeros(2)
        self.hits = 0
        self.outside_seen = False
        self.blocked_packet_positions = ()
        self.blocked_packet_points = ()
        self.used_at = None
        self.last_detection = None
        self.last_interaction = None
        self.last_interaction_t = None

    def observe(self, t, detections, bag, shape, other_bench_points=()):
        if self.used_at is not None and t <= self.used_at:
            return None
        self.used_at = t
        self.last_detection = None
        dt = t-self.last_t
        if dt < 0:
            return None
        reacquiring = dt > 1.2
        diagonal = float(np.hypot(*shape[:2]))
        gate = min(diagonal*.38, max(diagonal*.30, np.linalg.norm(self.velocity)*dt*1.7 + diagonal*.04))
        predicted = self.point if reacquiring else self.point + self.velocity * dt
        if reacquiring:
            gate = max(16., diagonal*.06)
        candidates = []
        for det in detections:
            point = np.asarray(det.c, dtype=float)
            if any(np.linalg.norm(point-p) < diagonal*.025 for p in other_bench_points):
                continue
            position = packet_position(point, bag)
            if reacquiring and (position is not None or gold_overlap(det, bag) >= .1):
                continue  # Recovery must occur outside the packet, not on old contents.
            relative = packet_position(point, bag, allow_box=True)
            old_position = relative is not None and any(np.linalg.norm(np.asarray(relative)-p) < .12
                                                         for p in self.blocked_packet_positions)
            old_pixels = any(np.linalg.norm(point-p) < diagonal*.035 for p in self.blocked_packet_points)
            if old_position or old_pixels:
                continue  # Visible gold already in the bag is not this transfer.
            distance = np.linalg.norm(point-predicted)
            if distance <= gate and np.linalg.norm(point-self.point) <= diagonal*.40:
                candidates.append((distance, det, point, position))
        candidates.sort(key=lambda item: item[0])
        if not candidates:
            return None
        if len(candidates) > 1 and candidates[1][0]-candidates[0][0] < diagonal*.012:
            return None  # No arbitrary choice between similar gold candidates.
        _, detection, point, position = candidates[0]
        if reacquiring:
            self.velocity = np.zeros(2)
            self.hits = 0
        elif dt > .001:
            self.velocity = .3*self.velocity + .7*(point-self.point)/dt
        self.point, self.last_t = point, t
        self.hits += 1
        self.last_detection = detection
        overlap = gold_overlap(detection, bag)
        inside = position is not None or overlap >= .1
        if bag is not None and not inside:
            self.outside_seen = True
            self.last_interaction = self.last_interaction_t = None
        moved = np.linalg.norm(point-self.anchor) >= max(12., diagonal*.012)
        if inside and moved and self.hits >= 2 and self.outside_seen:
            self.last_interaction = Observation(tuple(point), packet_overlap=max(overlap, .1), near_packet=True,
                                                identity_ok=False, method='pickup_owned_gold_packet_interaction')
            self.last_interaction_t = t
            return self.last_interaction
        return None

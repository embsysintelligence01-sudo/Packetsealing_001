"""Follow the pickup hand geometrically; never classify hidden bag contents."""
import cv2
import numpy as np

from .flow import Observation


class TransferFollower:
    def __init__(self):
        self.owner = None
        self.point = None
        self.last_t = None
        self.hand_acquired = False

    def start(self, owner, point, t):
        self.owner, self.point, self.last_t = owner, tuple(map(float, point)), t
        self.hand_acquired = False

    def observe(self, t, hand_mask, bag, shape):
        if self.point is None or hand_mask is None:
            return None
        diagonal = float(np.hypot(*shape[:2]))
        contours, _ = cv2.findContours((hand_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            if cv2.contourArea(contour) < 40:
                continue
            distance = max(0., -cv2.pointPolygonTest(contour, self.point, True))
            candidates.append((distance, contour))
        if not candidates:
            return None
        distance, contour = min(candidates, key=lambda v: v[0])
        elapsed = max(0., t - self.last_t)
        # A missing hand cannot jump to the other operator hand holding the bag.
        reacquiring = self.hand_acquired and elapsed > 1.5
        gate = .04 if reacquiring else .065 if not self.hand_acquired else .12
        if distance > diagonal * gate:
            return None
        if reacquiring and bag is not None:
            candidate = np.zeros(shape[:2], np.uint8)
            cv2.drawContours(candidate, [contour], -1, 255, -1)
            if np.any((candidate > 0) & (bag.mask > 0)):
                return None  # Do not jump onto the packet-holding hand after a gap.
        moments = cv2.moments(contour)
        if not moments['m00']:
            return None
        self.point = (moments['m10'] / moments['m00'], moments['m01'] / moments['m00'])
        self.last_t, self.hand_acquired = t, True
        if bag is None:
            return None
        courier = np.zeros(shape[:2], np.uint8)
        cv2.drawContours(courier, [contour], -1, 255, -1)
        # A small proximity band permits arrival at the opening boundary. The
        # hand must first have been associated with this pickup, not any hand.
        margin = max(3, round(diagonal * .008))
        near = cv2.dilate(bag.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2))
        overlap = np.count_nonzero((courier > 0) & (near > 0))
        if overlap < max(20, np.count_nonzero(courier) * .015):
            return None
        return Observation(self.point, near_packet=True, method="pickup_owned_hand_packet_interaction")

"""Pickup-owned hand approach evidence; contact alone never proves placement."""
import cv2
import numpy as np

from .flow import Observation


class TransferFollower:
    def __init__(self):
        self.owner = None
        self.point = None
        self.last_t = None
        self.hand_acquired = False
        self.used_at = None
        self.hand_area = None
        self._reset_arrival()

    def _reset_arrival(self):
        self.outside_point = self.outside_bag = None
        self.outside_at = self.last_outside = None
        self.outside_hits = 0
        self.entry_at = None
        self.entry_hits = 0
        self.bag_center = self.bag_area = None

    def start(self, owner, point, t):
        self.__init__()
        self.owner, self.point, self.last_t = owner, tuple(map(float, point)), t

    def observe(self, t, hand_mask, bag, shape):
        if self.point is None or hand_mask is None or (self.used_at is not None and t <= self.used_at):
            return None
        self.used_at = t
        diagonal = float(np.hypot(*shape[:2]))
        elapsed = max(0., t-self.last_t)
        reacquiring = self.hand_acquired and elapsed > .8
        if reacquiring:
            self._reset_arrival()
        contours, _ = cv2.findContours((hand_mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = sorted((max(0., -cv2.pointPolygonTest(c, self.point, True)), index, c)
                            for index, c in enumerate(contours) if cv2.contourArea(c) >= 40)
        if not candidates:
            self._reset_arrival()
            return None
        distance, _, contour = candidates[0]
        gate = .04 if reacquiring else .065 if not self.hand_acquired else .12
        if distance > diagonal*gate:
            self._reset_arrival()
            return None
        # Crossing/merged hands cannot inherit an unambiguous courier path.
        if len(candidates) > 1 and candidates[1][0]-distance < diagonal*.012:
            self._reset_arrival()
            return None
        area = cv2.contourArea(contour)
        moments = cv2.moments(contour)
        if not moments['m00']:
            self._reset_arrival()
            return None
        point = (moments['m10']/moments['m00'], moments['m01']/moments['m00'])
        if self.hand_acquired and (np.linalg.norm(np.asarray(point)-self.point) > diagonal*.14 or
                (not reacquiring and self.hand_area and not .4 <= area/self.hand_area <= 2.5)):
            self._reset_arrival()
            return None
        courier = np.zeros(shape[:2], np.uint8)
        cv2.drawContours(courier, [contour], -1, 255, -1)
        if reacquiring and bag is not None and np.any((courier > 0) & (bag.mask > 0)):
            return None  # A blind gap cannot reconnect directly onto packet handling.
        self.point, self.last_t, self.hand_area, self.hand_acquired = point, t, area, True
        if bag is None:
            self._reset_arrival()
            return None
        contours, _ = cv2.findContours(bag.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            self._reset_arrival()
            return None
        # The supplied model locates the packet body, not its physical mouth.
        # Hull/boundary arrival is supporting evidence, never verified insertion.
        hull = cv2.convexHull(np.concatenate(contours))
        bag_area = cv2.contourArea(hull)
        m = cv2.moments(hull)
        if not m['m00']:
            self._reset_arrival()
            return None
        bag_center = np.array([m['m10']/m['m00'], m['m01']/m['m00']])
        if self.bag_area and (not .65 <= bag_area/self.bag_area <= 1.5 or
                np.linalg.norm(bag_center-self.bag_center) > diagonal*.08):
            # Flattening, rotating or bringing the bag into view invalidates
            # the old approach; require fresh outside evidence afterward.
            self._reset_arrival()
        self.bag_center, self.bag_area = bag_center, bag_area
        margin = max(3., diagonal*.008)
        signed = cv2.pointPolygonTest(hull, point, True)
        footprint = np.zeros(shape[:2], np.uint8)
        cv2.fillConvexPoly(footprint, hull, 255)
        overlap = np.count_nonzero((courier > 0) & (footprint > 0))/max(1, np.count_nonzero(courier))
        if signed < -margin and overlap < .05:
            if self.outside_point is None:
                self.outside_point, self.outside_bag = np.asarray(point), bag_center.copy()
                self.outside_at = t
            self.outside_hits += 1
            self.last_outside = t
            self.entry_hits, self.entry_at = 0, None
            return None
        contact = signed >= -margin or overlap >= .015
        if not contact:
            self.entry_hits, self.entry_at = 0, None
            return None
        evidence = dict(kind='packet_contact_only', opening_verified=False)
        result = Observation(point, near_packet=True, method='pickup_owned_hand_contact', evidence=evidence)
        if self.outside_point is None or self.outside_hits < 2 or t-self.outside_at < .15:
            return result
        if self.entry_at is None and (t-self.last_outside > .8 or signed > max(8., np.sqrt(bag_area)*.12)):
            return result  # Missed boundary arrival cannot be reconstructed from a later press.
        axis = self.outside_bag-self.outside_point
        length = np.linalg.norm(axis)
        if length < 1:
            return result
        axis /= length
        hand_delta = np.asarray(point)-self.outside_point
        bag_delta = bag_center-self.outside_bag
        hand_progress = float(hand_delta @ axis)
        relative_progress = float((hand_delta-bag_delta) @ axis)
        packet_toward_hand = max(0., -float(bag_delta @ axis))
        threshold = max(8., diagonal*.012)
        entered = (signed >= -max(2., diagonal*.003) and overlap >= .15 and
                   hand_progress >= threshold and relative_progress >= threshold and
                   hand_progress >= packet_toward_hand*1.5+threshold)
        if not entered:
            self.entry_hits, self.entry_at = 0, None
            return result
        if self.entry_at is None:
            self.entry_at = t
        self.entry_hits += 1
        if self.entry_hits >= 2 and t-self.entry_at >= .15:
            result.transfer_verified = True
            result.method = 'pickup_owned_hand_packet_interaction'
            result.evidence = dict(kind='continuous_outside_to_packet_boundary', opening_verified=False,
                                   outside_at=self.outside_at, arrival_started_at=self.entry_at,
                                   confirmed_at=t, outside_observations=self.outside_hits,
                                   arrival_observations=self.entry_hits,
                                   hand_progress_px=hand_progress, relative_progress_px=relative_progress)
        return result

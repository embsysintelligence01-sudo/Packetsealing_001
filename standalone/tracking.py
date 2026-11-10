"""Reuse version 5's actual SLIC/ResNet tracking without running its testing workflow."""
import cv2
import numpy as np
import jewel_tracker_5 as original
from int8_models import OpenVINOYolo
from continuous_testing import visibility, choose_hand_jewel, PICKUP_STATES, HAND
from .flow import Pickup, INSIDE, PICKED, UNVERIFIED, INTERACTING
from .transfer import TransferFollower


class GoldPredictor:
    def __init__(self):
        self.model = OpenVINOYolo(original.MODEL_PATH)

    def predict(self, frame, **kwargs):
        kwargs.pop("_live_job", None)
        kwargs.pop("_captured_at", None)
        return self.model.predict(frame, **kwargs)


class TrackingBridge(original.Worker):
    def __init__(self, roi):
        super().__init__(0, roi, continuous=True)
        if not self.identity_slic:
            raise RuntimeError("SLIC is unavailable. Start with run_packet_sealing.sh and .venv-slic.")
        self.model = GoldPredictor()
        self.identity_error = ""
        self.event.connect(self._event, original.QtCore.Qt.DirectConnection)
        self.inactive_ids = set()
        self.transfer_active = False
        self.returners = {}
        self.detections = []
        self.recent_hand = {}
        self.continuous_identity = {}
        self.bench_anchors = {}
        self.taken_ids = set()
        self.bench_absence_hits = {}
        self.bench_contact = {}
        self.pickup_followers = {}
        self.pickup_seeds = {}
        self.raw_detections = []

    def _event(self, message, level):
        if "gallery unavailable" in message:
            self.identity_error = message

    def stone_present(self, frame):
        return True  # Use the same stable-count assignment without an acid-test stone.

    def update_rubbing(self, t):
        pass

    def update_acid(self, t):
        pass

    def check_done(self, tr, seen, t):
        pass

    def evaluate(self, tr, seen, t):
        super().evaluate(tr, seen, t)

    def save_capture(self, frame, jewels):
        pass

    def process(self, frame, t, capture=True):
        self.frame_wh = (frame.shape[1], frame.shape[0])
        self.ensure_roi()
        all_taken = self.mode == "LOCKED" and self.tracks and all(tr.id in self.inactive_ids for tr in self.tracks)
        if all_taken and not self.transfer_active:
            self.detections, self.returners, self.raw_detections = [], {}, []
            return []  # No uncompleted transfer needs gold inference.
        detections, self.rejected = self.detect(self.model, frame, self.roi, captured_at=t)
        self.detections = detections
        self.raw_detections = detections
        self.raw_timestamp = t
        if all_taken:
            self.detections, self.returners = [], {}
            return detections  # Raw gold still follows the final pickup; no ID reassignment.
        if self.mode == "ARMING":
            if capture:
                self.try_lock(detections, t, frame)
                if self.mode == "LOCKED":
                    self.continuous_identity = {tr.id: True for tr in self.tracks}
                    self.bench_anchors = {tr.id: (tuple(map(float, tr.pos)), tuple(map(float, tr.box))) for tr in self.tracks}
        else:
            with self.lock:
                all_tracks = self.tracks
                previous = {tr.id: (tr.det_seen, self.continuous_identity.get(tr.id, True)) for tr in all_tracks}
                self.returners = {}
                # Only unhandled bench positions participate in identity
                # matching. Carried/placed gold is never classified again.
                active_detections = detections
                if self.inactive_ids and self.bench_anchors:
                    anchors = [(point, box) for jid, (point, box) in self.bench_anchors.items()
                               if jid not in self.inactive_ids]
                    active_detections = [det for det in detections if any(
                        np.linalg.norm(np.asarray(det.c) - point) <= max(36.,
                            np.hypot(box[2]-box[0], box[3]-box[1]) * .65)
                        for point, box in anchors)]
                self.detections = active_detections
                self.tracks = [tr for tr in all_tracks if tr.id not in self.inactive_ids]
                try:
                    self.update_tracks(active_detections, t, frame)
                finally:
                    self.tracks = all_tracks
                for tr in self.tracks:
                    was_seen, was_known = previous[tr.id]
                    fresh = abs(tr.det_seen - t) < .0001
                    known = was_known and 0 <= t - was_seen <= .75
                    for det in active_detections:
                        if np.array_equal(det.box, tr.det_box) and det.idres is not None:
                            jid, probability = det.idres
                            if jid is not None and probability >= original.ID_MIN_PROB:
                                known = jid == tr.id
                    self.continuous_identity[tr.id] = fresh and known
        return detections

    def pickup_event(self, t, hand_mask=None):
        """Use the original recorder's visibility and handled-ID selection rules."""
        if self.mode != "LOCKED":
            return Pickup()
        tracks = [dict(id=tr.id, state=tr.state, spot=tr.spot, det_seen=tr.det_seen,
                       in_hand=getattr(tr, "in_hand", False), acid_done=False)
                  for tr in self.tracks if tr.id not in self.inactive_ids]
        moving = bool(getattr(self, "moving_now", False))
        states = {tr["id"]: visibility(tr, t, moving) for tr in tracks}
        for jid, state in states.items():
            if state in PICKUP_STATES:
                self.recent_hand[jid] = t
        active = any(state in PICKUP_STATES for state in states.values())
        candidate = choose_hand_jewel(tracks, t, moving, self.recent_hand) if active else None
        if hand_mask is not None:
            # Original HSV hand contact can mistake gold/skin-colored bench
            # pixels for a pickup. Supply segmented contact to the same handled-ID rules at the fixed
            # assigned bench spot using the separate hand segmentation model.
            held = []
            for jid, (point, box) in self.bench_anchors.items():
                if jid in self.taken_ids or jid in self.inactive_ids:
                    continue
                radius = max(12., max(box[2] - box[0], box[3] - box[1]) * .6)
                x, y = map(int, point)
                x1, y1 = max(0, int(x-radius)), max(0, int(y-radius))
                x2, y2 = min(hand_mask.shape[1], int(x+radius)), min(hand_mask.shape[0], int(y+radius))
                # A courier passing above another bangle must not start its
                # cycle while that bangle is still detected at its bench spot.
                tolerance = max(10., max(box[2]-box[0], box[3]-box[1]) * .22)
                still_on_bench = any(np.linalg.norm(np.asarray(det.c) - point) <= tolerance
                                     for det in self.detections)
                self.bench_absence_hits[jid] = 0 if still_on_bench else self.bench_absence_hits.get(jid, 0) + 1
                contact = np.count_nonzero(hand_mask[y1:y2, x1:x2]) >= max(20, radius * radius * .03)
                if contact:
                    self.bench_contact[jid] = t
                    if jid not in self.pickup_followers:
                        follower = TransferFollower()
                        follower.start(jid, point, t)
                        self.pickup_followers[jid] = follower
                follower = self.pickup_followers.get(jid)
                if follower is not None:
                    follower.observe(t, hand_mask, None, (*hand_mask.shape, 3))
                if t - self.bench_contact.get(jid, -float("inf")) <= .8 and self.bench_absence_hits[jid] >= 2:
                    held.append(jid)
                    candidates = self.raw_detections or self.detections
                    nearest = min(candidates, key=lambda d: np.linalg.norm(np.asarray(d.c)-point), default=None)
                    if follower is not None and follower.hand_acquired and t - follower.last_t <= .5:
                        self.pickup_seeds[jid] = follower.point
                        continue
                    self.pickup_seeds[jid] = (tuple(map(float, nearest.c)) if nearest is not None and
                        np.linalg.norm(np.asarray(nearest.c)-point) < np.hypot(*hand_mask.shape) * .3 else point)
            if not held:
                return Pickup()
            bench_tracks = [dict(id=jid, in_hand=True, state="OK", spot="hand", det_seen=t, acid_done=False)
                            for jid in held]
            candidate = choose_hand_jewel(bench_tracks, t, moving, self.recent_hand)
            return Pickup(True, candidate, candidate is None, tuple(held))
        return Pickup(active, candidate, active and candidate is None,
                      tuple(jid for jid, state in states.items() if state in HAND))

    def reset_assignment(self):
        # Use the tracker's own reset, invalidating background gallery publication.
        self.arm()
        self.identity_training_token = None
        self.id_model.ready = False
        self.identity_error = ""
        self.inactive_ids.clear()
        self.transfer_active = False
        self.bench_anchors.clear()
        self.taken_ids.clear()
        self.bench_absence_hits.clear()
        self.bench_contact.clear()
        self.pickup_followers.clear()
        self.pickup_seeds.clear()
        self.raw_detections = []
        self.returners, self.detections = {}, []
        self.prev_stable, self.prev_dets = [], []
        self.recent_hand, self.continuous_identity = {}, {}

    def close(self):
        self.running = False
        self.flow_revision += 1
        self.identity_training_token = None


def point_in(poly, point, margin=0):
    return cv2.pointPolygonTest(np.asarray(poly, np.int32), tuple(map(float, point)), True) >= -margin


def draw_gold(frame, tracker, t, statuses=None):
    """Show accepted INT8 detections before lock, then the original stable IDs.

    Missing tracks have dashed last-known boxes, matching the tracker's visual
    distinction between a current detection and an occluded jewel.
    """
    statuses = statuses or {}
    locked = tracker.mode == "LOCKED"
    for det in ([] if locked else getattr(tracker, "detections", [])):
        if tracker.roi is not None and not point_in(tracker.roi, det.c):
            continue
        color = (130, 130, 130) if locked else (0, 210, 255)
        if det.mask is not None:
            contours, _ = cv2.findContours(det.mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            offset = np.asarray(det.origin, np.int32).reshape(1, 1, 2)
            cv2.drawContours(frame, [c + offset for c in contours], -1, color, 1 if locked else 2)
        if not locked:
            x1, y1, x2, y2 = map(int, det.box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"GOLD {det.conf:.0%}", (x1, max(22, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
    if locked:
        for tr in tracker.tracks:
            status = statuses.get(tr.id, "")
            if status in (INSIDE, PICKED, UNVERIFIED, INTERACTING):
                continue  # Retained sidebar result; never draw an old bench box as live tracking.
            x1, y1, x2, y2 = map(int, tr.box)
            seen = abs(t - tr.det_seen) < .0001
            recent = 0 <= t - tr.det_seen <= .6
            if recent and getattr(tr, "det_mask", None) is not None:
                contours, _ = cv2.findContours(tr.det_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                offset = np.asarray(tr.det_origin, np.int32).reshape(1, 1, 2)
                cv2.drawContours(frame, [c + offset for c in contours], -1, (130, 130, 130), 1)
            color = (0, 130, 255) if status == "Uncertain" else (230, 210, 30)
            label = f"J{tr.id}" + (f" {status}" if status else "")
            if recent:
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            else:
                original.dashed_rect(frame, (x1, y1), (x2, y2), color)
                label += " - not seen"
            cv2.putText(frame, label, (x1, max(22, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
    return frame


def packet_overlap(mask, origin, packet_mask):
    """Share of the jewel's segmented pixels within the full bag mask."""
    if mask is None or packet_mask is None:
        return 0.
    ox, oy = map(int, origin)
    height, width = mask.shape[:2]
    fh, fw = packet_mask.shape[:2]
    x1, y1, x2, y2 = max(0, ox), max(0, oy), min(fw, ox + width), min(fh, oy + height)
    if x2 <= x1 or y2 <= y1:
        return 0.
    jewel = mask[y1 - oy:y2 - oy, x1 - ox:x2 - ox] > 0
    return float(np.count_nonzero(jewel & (packet_mask[y1:y2, x1:x2] > 0)) / max(1, np.count_nonzero(mask)))

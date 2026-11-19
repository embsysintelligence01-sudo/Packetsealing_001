#!/usr/bin/env python3
"""Isolated pickup experiment. No changes to the main packing workflow.

Run with the project's .venv-slic Python. Use --self-test for deterministic
state-machine checks; --check-models for a real-model CPU smoke check.
Tracking and reference coordinates use the original capture resolution. Confirmed means the
original spot was independently verified empty; it does not prove insertion.
PICKUP_DETECTED requires gold motion plus an exposed empty anchor; PICKUP_CONFIRMED is later original-spot
verification. Observed pickup, detection and verification timestamps are separate.
An Otsu foreground/edge verification mask is captured once with each hand-clear
reference. The model's segmentation mask remains unchanged for live tracking.
Masks and quality metadata are saved beside the original reference crop.

The third mode uses hand_landmarker.task fingertips and coherent visible gold
movement beyond --cog-threshold; no packet or empty-home inference is used.
Previous logic loads no hand model. Home-position logic uses hand segmentation,
fixed home masks/COG, provisional contact recording and exposed-home verification.
The original jewellery Worker provides
persistent initial assignment, the existing SLIC/identity gallery, positional
matching, reacquisition, ID correction, motion and IN_HAND / UNDER_HAND cues.
Only stone/rubbing/acid workflows are bypassed in this separate pickup test.
Otsu verification observes fixed references independently: it never changes
tracker boxes, IDs, detections, or eligibility after a confirmed removal.
Camera motion / insufficient background texture pauses verification, not tracking.
Removal from the original spot does not distinguish lifting from sliding.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import struct
from types import SimpleNamespace
from zoneinfo import ZoneInfo

os.environ.setdefault('YOLO_AUTOINSTALL', 'False')
os.environ.setdefault('OMP_WAIT_POLICY', 'PASSIVE')
ROOT = Path(__file__).resolve().parent
TIMEZONE = ZoneInfo('Asia/Kolkata')
TRACKER = Path(os.environ.get('JEWEL_TRACKER_DIR', ROOT.parent/'jewel_tracker'/'jewel_tracker'))
sys.path.insert(0, str(TRACKER))

import cv2
import numpy as np


def capture_devices():
    """Enumerate Linux capture nodes; exclude webcam metadata-only endpoints."""
    if not sys.platform.startswith('linux'):
        return []
    import fcntl
    devices = []
    for path in sorted(Path('/dev').glob('video[0-9]*')):
        try:
            index = int(path.name[5:])
            data = bytearray(104)  # struct v4l2_capability
            with path.open('rb', buffering=0) as handle:
                fcntl.ioctl(handle, 0x80685600, data, True)  # VIDIOC_QUERYCAP
            _, capabilities, device_caps = struct.unpack_from('III', data, 80)
            caps = device_caps if capabilities & 0x80000000 else capabilities
            if caps & (0x1 | 0x1000):  # VIDEO_CAPTURE / VIDEO_CAPTURE_MPLANE
                name = bytes(data[16:48]).split(b'\0')[0].decode(errors='replace')
                devices.append((index, name))
        except (OSError, ValueError):
            continue
    return devices


def small_frame(frame):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (960, round(h*960/w)), interpolation=cv2.INTER_AREA) if w > 960 else frame.copy()


def padded(box, shape, pad=12):
    x1, y1, x2, y2 = map(int, box)
    return max(0, x1-pad), max(0, y1-pad), min(shape[1], x2+pad), min(shape[0], y2+pad)


def region(image, box):
    x1, y1, x2, y2 = map(int, box)
    return image[y1:y2, x1:x2]


def center(box):
    return np.array([(box[0]+box[2])/2, (box[1]+box[3])/2], dtype=float)


def mask_box(mask):
    x, y, w, h = cv2.boundingRect(mask)
    return x, y, x+w, y+h


def coverage(mask, box):
    patch = region(mask, box)
    return float(np.mean(patch > 0)) if patch.size else 0.


def union_masks(detections, shape):
    result = np.zeros(shape[:2], np.uint8)
    for detection in detections:
        result |= detection.mask
    return result


def verification_reference(reference, model_mask):
    """Capture an Otsu foreground mask once; never fill holes or learn from absence."""
    started = time.perf_counter()
    gray = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    support = model_mask > 0
    background = cv2.dilate(model_mask, np.ones((9, 9), np.uint8)) == 0
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    threshold, dark = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    bed = float(np.median(gray[background])) if np.any(background) else float(np.median(gray))
    candidate = (dark if bed >= threshold else cv2.bitwise_not(dark)) & model_mask
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidate, 8)
    keep = np.zeros(count, np.uint8)
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= max(8, round(np.count_nonzero(support)*.001)):
            keep[index] = 255
    candidate = keep[labels]
    foreground = candidate > 0
    pixels = int(np.count_nonzero(foreground))
    contrast = float(np.mean(np.abs(gray[foreground].astype(float)-bed))) if pixels else 0.
    edges = cv2.Canny(gray, 50, 120)
    edges &= cv2.dilate(candidate, np.ones((9, 9), np.uint8))
    edge_pixels = int(np.count_nonzero(edges))
    reason = ('Not enough clear bed background around the model mask.' if np.count_nonzero(background) < 30 else
              'Otsu foreground is empty or too small.' if pixels < max(15, .01*np.count_nonzero(support)) else
              'Foreground/bed contrast is insufficient.' if contrast <= 18 else
              'Too few reference edges for independent verification.' if edge_pixels < 8 else '')
    info = dict(method='otsu_foreground', threshold=float(threshold),
                polarity='dark_on_light' if bed >= threshold else 'light_on_dark',
                model_pixels=int(np.count_nonzero(support)), verification_pixels=pixels,
                reference_contrast=round(contrast, 3), edge_pixels=edge_pixels,
                valid=not bool(reason), reason=reason,
                build_ms=round((time.perf_counter()-started)*1000, 3))
    return candidate, edges, info


@dataclass
class Jewel:
    id: int
    box: tuple
    position: np.ndarray
    crop_box: tuple
    reference: np.ndarray
    gold_mask: np.ndarray
    state: str = 'ON_BENCH'
    at: float = 0.
    reason: str = 'Original position retained.'
    tracker_visibility: str = 'VISIBLE'
    contact_at: float | None = None
    departed: bool = False
    votes: int = 0
    vote_since: float | None = None
    vote_kind: str = ''
    clear_since: float | None = None
    metrics: dict | None = None
    live_box: tuple | None = None
    live_seen: float = -math.inf
    live_source: str = 'unavailable'
    trigger_source: str = ''
    packet_visit: bool = False
    release_since: float | None = None
    release_votes: int = 0
    verification_mask: np.ndarray | None = None
    verification_edges: np.ndarray | None = None
    verification_info: dict | None = None
    pickup_at: float | None = None
    detected_at: float | None = None
    verified_at: float | None = None
    verification_status: str = 'not_started'
    motion_since: float | None = None
    motion_hits: int = 0
    motion_previous: tuple | None = None
    pickup_basis: str = ''
    home_cog: np.ndarray | None = None
    current_cog: np.ndarray | None = None
    displacement: float | None = None
    velocity: float | None = None
    cog_previous: tuple | None = None
    associated_hand: int | None = None
    hand_position: list | None = None
    destination_hint: str = ''
    contact_active: bool = False
    verification_cache: dict | None = None

    def __post_init__(self):
        ys, xs = np.nonzero(self.gold_mask)
        self.home_cog = (np.array([xs.mean()+self.crop_box[0], ys.mean()+self.crop_box[1]])
                         if len(xs) else self.position.copy())
        self.verification_mask, self.verification_edges, self.verification_info = verification_reference(self.reference, self.gold_mask)
        # Immutable geometry/reference statistics are prepared once per jewel.
        # Per-frame checks use the same full-resolution pixels and thresholds.
        ring = cv2.dilate(self.gold_mask, np.ones((9, 9), np.uint8)) == 0
        gray = cv2.cvtColor(self.reference, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
        ring_idx = np.flatnonzero(ring)
        metal_idx = np.flatnonzero(self.verification_mask)
        ring_bgr = self.reference.reshape(-1, 3)[ring_idx].astype(np.float32)
        bed = np.median(ring_bgr, axis=0) if len(ring_bgr) else np.zeros(3)
        noise = float(np.percentile(np.max(np.abs(ring_bgr-bed), axis=1), 90)) if len(ring_bgr) else 0.
        bed_gray = float(np.median(gray[ring_idx])) if len(ring_idx) else 0.
        self.verification_cache = dict(
            ring=ring_idx, metal=metal_idx, support=np.flatnonzero(self.gold_mask),
            edges=np.flatnonzero(self.verification_edges), gray_ring=gray[ring_idx],
            gray_metal=gray[metal_idx], bed_tolerance=min(32., max(16., noise+8.)),
            old_contrast=float(np.mean(np.abs(gray[metal_idx]-bed_gray))) if len(metal_idx) else 0.)



class PickupEngine:
    def __init__(self, emit, save_reference=None):
        self.emit = emit
        self.save_reference = save_reference or (lambda jewel: None)
        self.jewels = []
        self.baseline = None
        self.stable_since = None
        self.stable_frames = 0
        self.last_t = None
        self.reference_gray = None
        self.scene_points = None
        self.global_uncertain = False
        self.message = 'Lay jewels separately; remove hands for initial assignment.'

    def change(self, jewel, state, t, reason):
        if jewel.state != state:
            self.emit(dict(time=t, jewel=f'J{jewel.id}', state=state, tracker_visibility=jewel.tracker_visibility, reason=reason,
                           contact_at=jewel.contact_at, pickup_at=jewel.pickup_at,
                           detected_at=jewel.detected_at, verified_at=jewel.verified_at,
                           verification_status=jewel.verification_status, pickup_basis=jewel.pickup_basis))
            jewel.at = t
        jewel.state, jewel.reason = state, reason

    def verification_pending(self, jewel, t, reason):
        jewel.verification_status = 'uncertain'
        if jewel.pickup_at is not None:
            jewel.reason = 'Pickup detected; verification pending. '+reason
        else:
            self.change(jewel, 'UNCERTAIN', t, reason)

    def assign(self, frame, gold, hands, bags, roi_mask, t):
        all_hands = union_masks(hands, frame.shape)
        bag_mask = union_masks(bags, frame.shape)
        gold = [d for d in gold if coverage(roi_mask, d.box) > .9 and
                coverage(bag_mask, d.box) < .10 and np.count_nonzero(d.mask) >= 15]
        gold.sort(key=lambda d: (center(d.box)[1], center(d.box)[0]))
        clear = bool(gold) and all(coverage(all_hands, padded(d.box, frame.shape, 18)) < .005 for d in gold)
        boxes = [d.box for d in gold]
        stable = clear and self.baseline is not None and len(boxes) == len(self.baseline)
        if stable:
            stable = all(np.linalg.norm(center(a)-center(b)) < 9 for a, b in zip(boxes, self.baseline))
        if not stable:
            self.baseline = boxes if clear else None
            self.stable_since = t if clear else None
            self.stable_frames = 1 if clear else 0
            return
        self.stable_frames += 1
        self.message = f'Initial assignment: {len(boxes)} jewels stable for {t-self.stable_since:.1f}/1.5 s'
        if t-self.stable_since < 1.5 or self.stable_frames < 4:
            return
        self.set_references(frame, list(enumerate(gold, 1)), bags, roi_mask, t)

    def adopt_tracks(self, frame, tracks, bags, roi_mask, t):
        """Freeze original Worker assignment for verification without reseeding it."""
        if self.jewels:
            return
        indexed_gold = []
        for track in tracks:
            full = np.zeros(frame.shape[:2], np.uint8)
            mask = track.det_mask
            if mask is not None:
                ox, oy = map(int, track.det_origin)
                h, w = mask.shape
                x1, y1 = max(0, ox), max(0, oy)
                x2, y2 = min(frame.shape[1], ox+w), min(frame.shape[0], oy+h)
                if x2 > x1 and y2 > y1:
                    full[y1:y2, x1:x2] = mask[y1-oy:y2-oy, x1-ox:x2-ox]
            indexed_gold.append((track.id, SimpleNamespace(box=tuple(map(int, track.det_box)), mask=full)))
        self.set_references(frame, indexed_gold, bags, roi_mask, t)

    def set_references(self, frame, indexed_gold, bags, roi_mask, t):
        bag_mask = union_masks(bags, frame.shape)
        for index, detection in indexed_gold:
            box = tuple(map(int, detection.box))
            crop_box = padded(box, frame.shape, max(12, round(np.linalg.norm(center(box)-np.array(box[:2]))*.3)))
            jewel = Jewel(index, box, center(box), crop_box, region(frame, crop_box).copy(),
                          region(detection.mask, crop_box).copy(), at=t, clear_since=self.stable_since)
            jewel.live_box, jewel.live_seen, jewel.live_source = box, t, 'gold'
            self.jewels.append(jewel)
            self.save_reference(jewel)
            self.emit(dict(time=t, jewel=f'J{index}', state='ASSIGNED', box=list(box), position=jewel.position.tolist(),
                           verification=jewel.verification_info,
                           reason=f"Otsu verification: {jewel.verification_info['verification_pixels']}/{jewel.verification_info['model_pixels']} pixels; "
                                  f"reference build {jewel.verification_info['build_ms']:.2f} ms."))
            if not jewel.verification_info['valid']:
                jewel.reason = 'Verification reference unsuitable: '+jewel.verification_info['reason']
                self.emit(dict(time=t, jewel=f'J{index}', state='REFERENCE_UNSUITABLE', reason=jewel.reason))
        self.reference_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        background = roi_mask.copy()
        background[bag_mask > 0] = 0
        for jewel in self.jewels:
            x1, y1, x2, y2 = padded(jewel.box, frame.shape, 24)
            background[y1:y2, x1:x2] = 0
        self.scene_points = cv2.goodFeaturesToTrack(self.reference_gray, 100, .008, 8, mask=background)
        self.message = 'Assigned. Original tracker visibility, gold movement and verified empty-home evidence are active; no hand model.'

    def scene_stable(self, gray):
        if self.scene_points is None or len(self.scene_points) < 8:
            return False
        target, ok, _ = cv2.calcOpticalFlowPyrLK(self.reference_gray, gray, self.scene_points, None, maxLevel=3)
        if target is None:
            return False
        back, reverse, _ = cv2.calcOpticalFlowPyrLK(gray, self.reference_gray, target, None, maxLevel=3)
        if back is None:
            return False
        valid = (ok.ravel() > 0) & (reverse.ravel() > 0)
        valid &= np.linalg.norm(back-self.scene_points, axis=2).ravel() < 1.5
        if valid.sum() < 8:
            return False
        before, after = self.scene_points[valid], target[valid]
        matrix, inliers = cv2.estimateAffinePartial2D(before, after, method=cv2.RANSAC, ransacReprojThreshold=1.5)
        if matrix is None or inliers is None or inliers.mean() < .7:
            return False
        motion = np.median(np.linalg.norm(after-before, axis=2))
        scale = math.hypot(matrix[0, 0], matrix[0, 1])
        angle = abs(math.degrees(math.atan2(matrix[1, 0], matrix[0, 0])))
        return motion < 2.5 and .99 < scale < 1.01 and angle < 1.

    def compare(self, jewel, frame):
        current = region(frame, jewel.crop_box)
        reference = jewel.reference
        if not jewel.verification_info['valid']:
            return dict(present=False, removed=False, reliable=False,
                        verification='otsu_foreground', reason=jewel.verification_info['reason'])
        cache = jewel.verification_cache
        metal, ring, edges = cache['metal'], cache['ring'], cache['edges']
        if len(metal) < 15 or len(ring) < 30:
            return dict(present=False, removed=False, reliable=False, reason='Reference support insufficient.')
        gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32)
        flat = gray.ravel()
        gray_ring, gray_metal = flat[ring], flat[metal]
        shift = float(np.median(cache['gray_ring']-gray_ring))
        corrected_ring = np.clip(gray_ring+shift, 0, 255)
        corrected_metal = np.clip(gray_metal+shift, 0, 255)
        changed = float(np.mean(np.abs(cache['gray_metal']-corrected_metal) > 22))
        ring_changed = float(np.mean(np.abs(cache['gray_ring']-corrected_ring) > 20))
        background = float(np.median(corrected_ring))
        new_contrast = float(np.mean(np.abs(corrected_metal-background)))
        corrected = np.clip(gray+shift, 0, 255)
        current_edges = cv2.dilate(cv2.Canny(corrected.astype(np.uint8), 50, 120), np.ones((3, 3), np.uint8))
        retained = float(np.mean(current_edges.ravel()[edges] > 0)) if len(edges) else 1.
        reliable = ring_changed < .20 and abs(shift) < 35
        pixels = current.reshape(-1, 3)
        bed_colour = np.median(np.clip(pixels[ring].astype(np.float32)+shift, 0, 255), axis=0)
        support_colour = np.clip(pixels[cache['support']].astype(np.float32)+shift, 0, 255)
        bed_fraction = float(np.mean(np.max(np.abs(support_colour-bed_colour), axis=1) < cache['bed_tolerance']))
        return dict(verification='otsu_foreground_and_bed_colour', verification_pixels=len(metal),
                    changed=round(changed, 3), edge_retention=round(retained, 3), ring_changed=round(ring_changed, 3),
                    illumination_shift=round(shift, 2), background_contrast=round(new_contrast, 2),
                    bed_fraction=round(bed_fraction, 3), reliable=reliable,
                    present=reliable and len(edges) >= 8 and changed < .22 and retained > .65,
                    removed=reliable and cache['old_contrast'] > 18 and changed > .65 and retained < .45 and
                            new_contrast < cache['old_contrast']*.55 and bed_fraction >= .90)

    def collecting(self, jewel, kind, t):
        elapsed = max(0., t-jewel.vote_since)
        label = 'Empty home' if kind == 'removed' else 'Jewel still present'
        reason = f'{label}: {min(jewel.votes, 3)}/3 clear observations; {elapsed:.2f}/0.45 s. Keep the home visible.'
        jewel.verification_status = 'verifying_empty_home' if kind == 'removed' else 'verifying_jewel_present'
        if jewel.pickup_at is None:
            self.change(jewel, 'VERIFYING', t, reason)
        else:
            jewel.reason = reason

    @staticmethod
    def evidence_blocker(metrics, gold_at_home):
        if metrics.get('reason'):
            return 'Reference unsuitable: '+metrics['reason']
        if not metrics.get('reliable', False):
            return ('Lighting changed too much around home.' if abs(metrics.get('illumination_shift', 0)) >= 35 else
                    'Background around home changed; waiting for a clear, stable view.')
        if gold_at_home > .025:
            return f'Gold still detected at home ({gold_at_home:.0%} coverage); removal not verified.'
        if metrics.get('edge_retention', 1.) >= .45:
            return 'Original metal edges still visible or obstructed; removal not verified.'
        if metrics.get('bed_fraction', 0.) < .90:
            return f"Home does not fully match exposed bench ({metrics.get('bed_fraction', 0.):.0%}); keep it clear."
        return 'Metal/reference evidence is conflicting; waiting for a clear view of the original spot.'

    def appearance_departure(self, jewel, metrics, gold_mask, bag_mask, gap):
        """Appearance alone needs empty-bed evidence, not a brightness change.

        Reuse the independent metal/edge/bed-colour check. A stationary bangle
        whose reflections change must not start another jewel's pickup cycle.
        This only starts observation; temporal confirmation still needs votes.
        """
        return (not gap and not self.global_uncertain and
                metrics.get('removed', False) and not metrics.get('present', False) and
                coverage(gold_mask, jewel.box) <= .025 and
                coverage(bag_mask, jewel.crop_box) <= .02)

    def vote(self, jewel, kind, t):
        if jewel.vote_kind != kind:
            jewel.vote_kind, jewel.votes, jewel.vote_since = kind, 0, t
        jewel.votes += 1
        return jewel.votes >= 3 and t-jewel.vote_since >= .45

    def update_live(self, frame, hands, tracked, t):
        for jewel in self.jewels:
            observation = tracked.get(jewel.id, {})
            jewel.tracker_visibility = observation.get('visibility', 'NOT_SEEN')
            if observation.get('seen') == t:
                jewel.live_box = tuple(map(int, observation['box']))
                jewel.live_seen, jewel.live_source = t, 'gold'
            else:
                jewel.live_source = 'not seen'

    def update(self, frame, gold, hands, bags, roi_mask, t, tracked=None, legacy_ids=()):
        """Model-free hand cues start candidates; independent bed evidence resolves them.

        `hands` is retained in this local API for assignment compatibility; the
        application always passes an empty list and runs no hand model.
        """
        if self.last_t is not None and t <= self.last_t:
            return
        gap = self.last_t is not None and t-self.last_t > 1.1
        self.last_t = t
        if not self.jewels:
            self.assign(frame, gold, [], bags, roi_mask, t)
            return
        tracked = tracked or {}
        self.update_live(frame, [], tracked, t)
        self.global_uncertain = not self.scene_stable(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        all_gold = union_masks(gold, frame.shape)
        bag_mask = union_masks(bags, frame.shape)
        for jewel in self.jewels:
            if jewel.state == 'PICKUP_CONFIRMED':
                continue
            observation = tracked.get(jewel.id, {})
            fresh = observation.get('seen') == t
            diagonal = np.linalg.norm(np.asarray(jewel.box[2:])-jewel.box[:2])
            moved = fresh and np.linalg.norm(center(jewel.live_box)-jewel.position) > max(10., diagonal*.20)
            metrics = self.compare(jewel, frame)
            jewel.metrics = metrics
            present = metrics['present']
            if present:
                moved = False
            cue = jewel.tracker_visibility in ('IN_HAND', 'UNDER_HAND')
            appearance_departed = self.appearance_departure(jewel, metrics, all_gold, bag_mask, gap)
            if jewel.state in ('ON_BENCH', 'PICKUP_CANCELLED'):
                # A carried J1 may cover J2. Candidate is only a request to
                # observe J2's own anchor; it never transfers an ID or green tick.
                trigger = ('existing '+jewel.tracker_visibility if cue else
                           'existing packet trigger' if jewel.id in legacy_ids else
                           'jewel movement' if moved else 'empty original spot' if appearance_departed else '')
                if not trigger:
                    continue
                # A persistent contact on a stationary jewel must not open a
                # new clip on every frame after cancellation.
                if jewel.state == 'PICKUP_CANCELLED' and present and not moved:
                    continue
                jewel.contact_at, jewel.departed = t, False
                jewel.pickup_at = jewel.detected_at = jewel.verified_at = None
                jewel.motion_hits, jewel.motion_since, jewel.motion_previous = 0, None, None
                jewel.votes, jewel.vote_kind, jewel.vote_since = 0, '', None
                jewel.pickup_basis, jewel.verification_status = '', 'pending'
                jewel.trigger_source = trigger
                self.change(jewel, 'PICKUP_CANDIDATE', t, trigger+'; recording started, removal not confirmed.')
            if gap or self.global_uncertain or coverage(bag_mask, jewel.crop_box) > .02:
                jewel.votes, jewel.vote_kind = 0, ''
                jewel.motion_hits, jewel.motion_since, jewel.motion_previous = 0, None, None
                self.verification_pending(jewel, t, 'Frame gap over 1.1 s; verification restarted.' if gap else
                    'Camera/background unstable; verification paused.' if self.global_uncertain else
                    'Packet covers the original spot; verification paused.')
                continue
            if present:
                jewel.motion_hits, jewel.motion_since, jewel.motion_previous = 0, None, None
                if self.vote(jewel, 'present', t):
                    jewel.verified_at, jewel.verification_status = t, 'jewel_present'
                    self.change(jewel, 'PICKUP_CANCELLED', t, 'Reference confirms the original jewel remains. Ready for a new change.')
                else:
                    self.collecting(jewel, 'present', t)
                continue
            # Without hand association a merged carrier/J2 mask can move the
            # wrong ID. Require an independently empty original anchor for the
            # early tick too, not just a moving detection or an occluded spot.
            other_anchor = fresh and any(other.id != jewel.id and
                np.linalg.norm(center(jewel.live_box)-other.position) < max(12., diagonal*.25)
                for other in self.jewels)
            previous = jewel.motion_previous
            coherent = previous is None or (.08 <= t-previous[0] <= .65 and
                np.linalg.norm(center(jewel.live_box)-previous[1]) <= max(45., diagonal*1.5))
            if moved and metrics['removed'] and not other_anchor and coherent:
                if jewel.motion_since is None:
                    jewel.motion_since = t
                jewel.motion_hits += 1
                jewel.motion_previous = (t, center(jewel.live_box).copy())
                if jewel.motion_hits >= 2 and jewel.pickup_at is None:
                    jewel.pickup_at, jewel.detected_at = jewel.motion_since, t
                    jewel.pickup_basis = 'tracked_gold_and_empty_anchor'
                    self.change(jewel, 'PICKUP_DETECTED', t, 'Gold moved AND its original spot independently matched empty bed in two observations; final temporal verification pending.')
            else:
                jewel.motion_hits, jewel.motion_since, jewel.motion_previous = 0, None, None
            detected_at_anchor = coverage(all_gold, jewel.box) > .025
            if not detected_at_anchor and metrics['removed']:
                jewel.departed = True
                if self.vote(jewel, 'removed', t):
                    # When gold is hidden, the first verified-empty observation
                    # is the earliest available evidence; never invent touch time.
                    if jewel.pickup_at is None:
                        jewel.pickup_at, jewel.detected_at = jewel.vote_since, t
                        jewel.pickup_basis = 'verified_empty_original_position'
                    jewel.verified_at, jewel.verification_status = t, 'removal_verified'
                    self.change(jewel, 'PICKUP_CONFIRMED', t, 'Gold absent AND cached metal/edges replaced by visible bench background across three observations.')
                else:
                    self.collecting(jewel, 'removed', t)
            else:
                jewel.votes, jewel.vote_kind = 0, ''
                self.verification_pending(jewel, t, self.evidence_blocker(metrics, coverage(all_gold, jewel.box)))
        self.message = ('Verification paused: background/camera uncertain.' if self.global_uncertain else
                        'No hand model | IN_HAND / UNDER_HAND + gold movement + original-position verification.')



class HomePositionEngine(PickupEngine):
    """Verify empty homes, allowing partial exposure with visible jewel displacement."""
    def __init__(self, emit, save_reference=None):
        super().__init__(emit, save_reference)
        self.hand_tracks = {}
        self.next_hand_id = 1
        self.assignment_probe = None
        self.assignment_since = None
        self.assignment_frames = 0

    def adopt_tracks(self, frame, tracks, bags, roi_mask, t, hands=()):
        if self.jewels:
            return
        boxes = [(tr.id, tuple(tr.det_box)) for tr in tracks]
        hand_mask = union_masks(hands, frame.shape)
        bag_mask = union_masks(bags, frame.shape)
        clear = bool(boxes) and all(tr.det_seen == t for tr in tracks) and all(coverage(hand_mask, padded(box, frame.shape, 8)) == 0 and
                                         coverage(bag_mask, padded(box, frame.shape, 8)) == 0
                                         for _, box in boxes)
        stable = clear and self.assignment_probe is not None and len(boxes) == len(self.assignment_probe)
        if stable:
            stable = all(a[0] == b[0] and np.linalg.norm(center(a[1])-center(b[1])) < 9
                         for a, b in zip(boxes, self.assignment_probe))
        if not stable:
            self.assignment_probe = boxes if clear else None
            self.assignment_since = t if clear else None
            self.assignment_frames = 1 if clear else 0
            self.message = 'Home mode: keep jewels stable and remove hands before saving homes.'
            return
        self.assignment_frames += 1
        if t-self.assignment_since < 1.5 or self.assignment_frames < 4:
            self.message = f'Home mode: uncovered home capture {t-self.assignment_since:.1f}/1.5 s'
            return
        super().adopt_tracks(frame, tracks, bags, roi_mask, t)

    def associate_hands(self, hands, t):
        # One-to-one matching. Merges/crossings get no inherited hand identity.
        previous = {key: value for key, value in self.hand_tracks.items() if 0 < t-value['time'] <= .8}
        candidates = []
        for hand in hands:
            cog = center(hand.box)
            matches = []
            for key, old in previous.items():
                intersection = np.count_nonzero((hand.mask > 0) & (old['mask'] > 0))
                union = np.count_nonzero((hand.mask > 0) | (old['mask'] > 0))
                distance = np.linalg.norm(cog-old['cog'])
                gate = max(25., np.linalg.norm(np.array(hand.box[2:])-hand.box[:2])*.6)
                if (union and intersection/union > .1) or distance < gate:
                    matches.append(key)
            candidates.append((hand, cog, matches))
        usage = {key: sum(key in matches for _, _, matches in candidates) for key in previous}
        current = {}
        for hand, cog, matches in candidates:
            if len(matches) == 1 and usage[matches[0]] == 1:
                key = matches[0]
            else:
                key = self.next_hand_id
                self.next_hand_id += 1
            current[key] = dict(mask=hand.mask, box=hand.box, cog=cog, time=t)
        self.hand_tracks = current
        return current

    def partial_home(self, jewel, frame, hand_mask, bag_mask, gold_mask):
        """Inspect only exposed original support; covered pixels never vote empty."""
        blocked = region(hand_mask | bag_mask, jewel.crop_box)
        # Exclude mask boundaries, where segmentation and skin shadows are uncertain.
        clear = (cv2.dilate(blocked, np.ones((7, 7), np.uint8)) == 0).ravel()
        cache = jewel.verification_cache
        support = cache['support'][clear[cache['support']]]
        metal = cache['metal'][clear[cache['metal']]]
        ring = cache['ring'][clear[cache['ring']]]
        fraction = len(support)/max(1, len(cache['support']))
        result = dict(partial_visible_fraction=round(fraction, 3), partial_empty=False)
        if (not jewel.verification_info['valid'] or fraction < .5 or
                len(metal) < max(15, .5*len(cache['metal'])) or len(ring) < 30):
            return result
        current = region(frame, jewel.crop_box)
        gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
        reference = cv2.cvtColor(jewel.reference, cv2.COLOR_BGR2GRAY).astype(np.float32).ravel()
        shift = float(np.median(reference[ring]-gray[ring]))
        corrected = np.clip(gray+shift, 0, 255)
        stable = abs(shift) < 35 and np.mean(np.abs(reference[ring]-corrected[ring]) > 20) < .20
        bed_gray = float(np.median(corrected[ring]))
        old_contrast = float(np.mean(np.abs(reference[metal]-np.median(reference[ring]))))
        changed = float(np.mean(np.abs(reference[metal]-corrected[metal]) > 22))
        contrast = float(np.mean(np.abs(corrected[metal]-bed_gray)))
        pixels = np.clip(current.reshape(-1, 3).astype(np.float32)+shift, 0, 255)
        bed = np.median(pixels[ring], axis=0)
        bed_fraction = float(np.mean(np.max(np.abs(pixels[support]-bed), axis=1) < cache['bed_tolerance']))
        gold_fraction = float(np.mean(region(gold_mask, jewel.crop_box).ravel()[support] > 0))
        result.update(partial_bed_fraction=round(bed_fraction, 3),
                      partial_gold_fraction=round(gold_fraction, 3),
                      partial_empty=bool(stable and old_contrast > 18 and changed > .65 and
                                         contrast < old_contrast*.55 and bed_fraction >= .90 and gold_fraction <= .025))
        return result

    def visible_cog(self, jewel, gold, hand_mask, tracked, t, allow_held=False):
        jewel.current_cog = jewel.displacement = jewel.velocity = None
        observation = tracked.get(jewel.id, {})
        # Third mode uses fresh segmentation evidence rather than the original
        # tracker's hand labels. Neither label alone proves the gold is hidden.
        blocked_visibility = () if allow_held else ('IN_HAND', 'UNDER_HAND')
        if observation.get('seen') != t or observation.get('visibility') in blocked_visibility:
            jewel.cog_previous = None
            return
        box = tuple(observation.get('detected_box', observation['box']))
        candidates = []
        for detection in gold:
            a, b = box, detection.box
            intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
            union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1])-intersection
            if union > 0 and intersection/union > .5:
                candidates.append(detection)
        if len(candidates) != 1:
            jewel.cog_previous = None
            return
        detection = candidates[0]
        if allow_held:
            # Hand status is an old tracker heuristic, not proof the fresh mask is
            # hidden. Require substantial original gold support before using it.
            ratio = np.count_nonzero(detection.mask)/max(1, np.count_nonzero(jewel.gold_mask))
            if not .65 <= ratio <= 1.5:
                jewel.cog_previous = None
                return
        # Reject merged detections that plausibly belong to another live ID.
        for other_id, other in tracked.items():
            if other_id != jewel.id and other.get('seen') == t:
                other_center = center(other['box'])
                if (detection.box[0] <= other_center[0] <= detection.box[2] and
                        detection.box[1] <= other_center[1] <= detection.box[3]):
                    jewel.cog_previous = None
                    return
        local_mask = region(detection.mask, detection.box)
        support = local_mask > 0
        if not np.any(support) or np.mean(region(hand_mask, detection.box)[support] > 0) > .03:
            jewel.cog_previous = None
            return
        moments = cv2.moments(local_mask, binaryImage=True)
        cog = np.array([moments['m10']/moments['m00']+detection.box[0],
                        moments['m01']/moments['m00']+detection.box[1]])
        # Never treat a detection at another jewel's home as a reliable J1 match.
        if any(other.id != jewel.id and np.linalg.norm(cog-other.home_cog) < max(12., np.linalg.norm(np.array(other.box[2:])-other.box[:2])*.25)
               for other in self.jewels):
            jewel.cog_previous = None
            return
        jewel.current_cog = cog
        jewel.displacement = float(np.linalg.norm(cog-jewel.home_cog))
        previous = jewel.cog_previous
        if previous is not None and 0 < t-previous[0] <= .8:
            jewel.velocity = float(np.linalg.norm(cog-previous[1])/(t-previous[0]))
        jewel.cog_previous = (t, cog.copy())

    def update(self, frame, gold, hands, bags, roi_mask, t, tracked=None, legacy_ids=()):
        if self.last_t is not None and t <= self.last_t:
            return
        gap = self.last_t is not None and t-self.last_t > 1.1
        self.last_t = t
        if not self.jewels:
            self.assign(frame, gold, hands, bags, roi_mask, t)
            return
        tracked = tracked or {}
        hand_tracks = self.associate_hands(hands, t)
        hand_mask = union_masks(hands, frame.shape)
        bag_mask = union_masks(bags, frame.shape)
        all_gold = union_masks(gold, frame.shape)
        self.update_live(frame, hands, tracked, t)
        self.global_uncertain = not self.scene_stable(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        for jewel in self.jewels:
            self.visible_cog(jewel, gold, hand_mask, tracked, t)
            if gap:
                jewel.velocity = None
                jewel.cog_previous = None
            support = cv2.dilate(jewel.gold_mask, np.ones((5, 5), np.uint8)) > 0
            contacts = [key for key, hand in hand_tracks.items()
                        if np.any(region(hand['mask'], jewel.crop_box)[support] > 0)]
            touching = bool(contacts)
            new_contact = touching and not jewel.contact_active
            jewel.contact_active = touching
            # Each home owns its association. Shared/merged hands never transfer jewel IDs.
            if touching:
                jewel.associated_hand = contacts[0] if len(contacts) == 1 else None
            carrier = hand_tracks.get(jewel.associated_hand)
            jewel.hand_position = carrier['cog'].tolist() if carrier is not None else None
            if carrier is None:
                jewel.associated_hand = None
            metrics = self.compare(jewel, frame)
            jewel.metrics = dict(metrics, home_cog=jewel.home_cog.tolist(),
                current_cog=jewel.current_cog.tolist() if jewel.current_cog is not None else None,
                displacement_px=jewel.displacement, velocity_px_s=jewel.velocity,
                associated_hand=jewel.associated_hand, hand_position=jewel.hand_position,
                home_covered=touching)
            # Destination observations remain supporting evidence and never prove insertion.
            if jewel.current_cog is not None and jewel.contact_at is not None and coverage(bag_mask, jewel.live_box) > .05:
                jewel.destination_hint = 'Tracked gold overlaps packet region; insertion unverified'
            if jewel.state == 'PICKUP_CONFIRMED':
                continue
            diagonal = np.linalg.norm(np.array(jewel.box[2:])-jewel.box[:2])
            moved = (jewel.displacement is not None and
                     jewel.displacement > max(10., diagonal*.20) and not metrics['present'])
            appearance_departed = (not touching and
                self.appearance_departure(jewel, metrics, all_gold, bag_mask, gap))
            if jewel.state in ('ON_BENCH', 'PICKUP_CANCELLED'):
                if jewel.state == 'PICKUP_CANCELLED' and not (new_contact or moved or appearance_departed):
                    continue
                trigger = ('hand contacts fixed home mask' if touching else
                           'visible COG departure' if moved else
                           'empty original spot' if appearance_departed else '')
                if not trigger:
                    continue
                jewel.contact_at, jewel.departed = t, False
                jewel.pickup_at = jewel.detected_at = jewel.verified_at = None
                jewel.votes, jewel.vote_kind, jewel.vote_since = 0, '', None
                jewel.motion_hits, jewel.motion_since, jewel.motion_previous = 0, None, None
                jewel.pickup_basis, jewel.verification_status = '', 'pending'
                jewel.trigger_source = trigger
                self.change(jewel, 'PICKUP_CANDIDATE', t, trigger+'; provisional recording started.')
            if touching and not (gap or self.global_uncertain or coverage(bag_mask, jewel.crop_box) > .02):
                partial = self.partial_home(jewel, frame, hand_mask, bag_mask, all_gold)
                jewel.metrics.update(partial)
                # A fresh, unoccluded detection of this ID must remain coherently
                # displaced on every partial-home vote. Hand movement is not evidence.
                displaced = (jewel.displacement is not None and
                             jewel.displacement > max(10., diagonal*.20))
                previous = jewel.motion_previous
                coherent = displaced and (previous is None or (.08 <= t-previous[0] <= .8 and
                    np.linalg.norm(jewel.current_cog-previous[1]) <= max(45., diagonal*1.5)))
                if partial['partial_empty'] and displaced and coherent:
                    jewel.motion_previous = (t, jewel.current_cog.copy())
                    if self.vote(jewel, 'partial_removed', t):
                        jewel.pickup_at, jewel.detected_at = jewel.vote_since, t
                        jewel.verified_at, jewel.verification_status = t, 'removal_verified'
                        jewel.pickup_basis = 'visible_cog_and_partial_empty_home'
                        jewel.departed = True
                        self.change(jewel, 'PICKUP_CONFIRMED', t,
                            'At least half the home is exposed and matches empty bench; same jewel displaced across three observations; destination unverified.')
                    else:
                        jewel.verification_status = 'verifying_partial_home'
                        self.change(jewel, 'VERIFYING', t,
                            f'Partial home empty + same jewel displaced: {jewel.votes}/3 observations; '
                            f'{t-jewel.vote_since:.2f}/0.45 s.')
                    continue
            if gap or self.global_uncertain or touching or coverage(bag_mask, jewel.crop_box) > .02:
                jewel.votes, jewel.vote_kind = 0, ''
                jewel.motion_hits, jewel.motion_since, jewel.motion_previous = 0, None, None
                self.verification_pending(jewel, t, 'Frame gap over 1.1 s; verification restarted.' if gap else
                    'Camera/background unstable; verification paused.' if self.global_uncertain else
                    'Hand covers the home; waiting for the original spot to become visible.' if touching else
                    'Packet covers the home; verification paused.')
                continue
            if metrics['present']:
                jewel.motion_hits, jewel.motion_since, jewel.motion_previous = 0, None, None
                if self.vote(jewel, 'present', t):
                    jewel.verified_at, jewel.verification_status = t, 'jewel_present'
                    self.change(jewel, 'PICKUP_CANCELLED', t, 'Uncovered home still contains the original jewel.')
                else:
                    self.collecting(jewel, 'present', t)
                continue
            if metrics['removed'] and coverage(all_gold, jewel.box) <= .025:
                previous = jewel.motion_previous
                coherent = moved and (previous is None or (.08 <= t-previous[0] <= .8 and
                    np.linalg.norm(jewel.current_cog-previous[1]) <= max(45., diagonal*1.5)))
                if moved and coherent:
                    if jewel.motion_since is None:
                        jewel.motion_since = t
                    jewel.motion_hits += 1
                    jewel.motion_previous = (t, jewel.current_cog.copy())
                    if jewel.motion_hits >= 2 and jewel.pickup_at is None:
                        jewel.pickup_at, jewel.detected_at = jewel.motion_since, t
                        jewel.pickup_basis = 'visible_cog_and_empty_home'
                        self.change(jewel, 'PICKUP_DETECTED', t, 'Visible jewel COG moved and uncovered home is empty; collecting final votes.')
                else:
                    jewel.motion_hits, jewel.motion_since, jewel.motion_previous = 0, None, None
                if self.vote(jewel, 'removed', t):
                    jewel.departed = True
                    if jewel.pickup_at is None:
                        jewel.pickup_at, jewel.detected_at = jewel.vote_since, t
                        jewel.pickup_basis = 'verified_empty_home_after_occlusion'
                    jewel.verified_at, jewel.verification_status = t, 'removal_verified'
                    self.change(jewel, 'PICKUP_CONFIRMED', t, 'Uncovered home matches empty bench across three observations; departure verified, destination unverified.')
                else:
                    self.collecting(jewel, 'removed', t)
            else:
                jewel.votes, jewel.vote_kind = 0, ''
                jewel.motion_hits, jewel.motion_since, jewel.motion_previous = 0, None, None
                self.verification_pending(jewel, t, self.evidence_blocker(metrics, coverage(all_gold, jewel.box)))
        self.message = 'Home-position mode | contact records provisionally; confirm clear empty home, or >=50% exposed empty home + same jewel displaced across frames.'


class COGMovementEngine(HomePositionEngine):
    """Third mode: fingertip contact followed by coherent visible jewel movement."""
    def __init__(self, emit, save_reference=None, threshold=20.):
        super().__init__(emit, save_reference)
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError('COG threshold must be a positive finite pixel distance')
        self.threshold = float(threshold)
        self.links = {}
        self.wait_for_exit = set()

    def update(self, frame, gold, hands, bags, roi_mask, t, tracked=None, legacy_ids=()):
        from standalone.hand_keypoints import TIP_IDS
        if self.last_t is not None and t <= self.last_t:
            return
        gap = self.last_t is not None and t-self.last_t > .8
        self.last_t = t
        if not self.jewels:
            self.assign(frame, gold, hands, [], roi_mask, t)
            return
        tracked = tracked or {}
        self.update_live(frame, [], tracked, t)
        tracks = self.associate_hands(hands, t)
        hand_by_key = {key: next(h for h in hands if h.mask is entry['mask'])
                       for key, entry in tracks.items()}
        self.global_uncertain = not self.scene_stable(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        clear_mask = np.zeros(frame.shape[:2], np.uint8)
        # Fresh visible gold only: landmarks never supply or extrapolate its COG.
        for jewel in self.jewels:
            self.visible_cog(jewel, gold, clear_mask, tracked, t, allow_held=True)
            jewel.hand_position = None
            if gap:
                jewel.velocity = None
        # Each hand may nominate just one home; near-ties are ambiguous.
        proposals = {}
        for key, hand in hand_by_key.items():
            ranked = []
            gate = max(8., hand.scale*.18)
            for jewel in self.jewels:
                if jewel.current_cog is None or jewel.displacement > self.threshold:
                    continue
                yy, xx = np.nonzero(jewel.gold_mask)
                if not len(xx):
                    continue
                pixels = np.column_stack((xx+jewel.crop_box[0], yy+jewel.crop_box[1]))
                distances = [float(np.sqrt(np.min(np.sum((pixels-hand.points[i])**2, axis=1))))
                             for i in TIP_IDS]
                nearest = int(np.argmin(distances))
                ranked.append((distances[nearest], jewel.id, TIP_IDS[nearest]))
            ranked.sort()
            if ranked and ranked[0][0] <= gate and (len(ranked) == 1 or ranked[1][0]-ranked[0][0] > gate*.35):
                distance, jewel_id, tip = ranked[0]
                proposals.setdefault(jewel_id, []).append((key, tip, distance))
        reserved = {link['hand'] for link in self.links.values()}
        self.wait_for_exit.intersection_update(proposals)
        for jewel in self.jewels:
            link = self.links.get(jewel.id)
            if jewel.state == 'MOVEMENT_TRIGGERED':
                if jewel.displacement is not None and jewel.displacement <= self.threshold*.5:
                    self.change(jewel, 'MOVEMENT_ENDED', t, 'Jewel returned home; motion recording ended.')
                    self.links.pop(jewel.id, None)
                continue
            if link is None:
                if jewel.id in self.wait_for_exit:
                    continue
                options = [item for item in proposals.get(jewel.id, []) if item[0] not in reserved]
                if len(options) != 1 or self.global_uncertain:
                    continue
                key, tip, distance = options[0]
                link = dict(hand=key, tip=tip, previous=None,
                            offset=jewel.current_cog-hand_by_key[key].points[tip])
                self.links[jewel.id] = link
                reserved.add(key)
                jewel.associated_hand = key
                jewel.contact_at = t
                jewel.pickup_at = jewel.detected_at = jewel.verified_at = None
                jewel.votes, jewel.vote_kind, jewel.vote_since = 0, '', None
                jewel.trigger_source = 'fingertip_near_jewel_mask'
                jewel.verification_status = 'pending_fingertip_motion'
                self.change(jewel, 'PICKUP_CANDIDATE', t, 'Fingertip near this jewel; provisional recording started.')
            hand = hand_by_key.get(link['hand'])
            cog = jewel.current_cog
            if gap or self.global_uncertain or hand is None or cog is None:
                link['previous'] = None
                jewel.votes, jewel.vote_kind = 0, ''
                self.verification_pending(jewel, t, 'Waiting for stable scene, same hand and visible jewel; hidden gold is not inferred.')
                if hand is None:
                    self.links.pop(jewel.id, None)
                    jewel.associated_hand = None
                continue
            tip = hand.points[link['tip']]
            jewel.hand_position = tip.tolist()
            previous = link['previous']
            link['previous'] = (t, cog.copy(), tip.copy())
            coherent = False
            cosine, drift = None, None
            if previous is not None and .08 <= t-previous[0] <= .8:
                gold_delta, tip_delta = cog-previous[1], tip-previous[2]
                gold_step, tip_step = np.linalg.norm(gold_delta), np.linalg.norm(tip_delta)
                if gold_step >= 2. and tip_step >= 2.:
                    cosine = float(np.dot(gold_delta, tip_delta)/(gold_step*tip_step))
                    drift = float(np.linalg.norm(gold_delta-tip_delta))
                    # Limit jumps and relative-distance drift, not just direction.
                    coherent = (cosine >= .7 and drift <= max(8., hand.scale*.15) and
                                np.linalg.norm((cog-tip)-link['offset']) <= max(12., hand.scale*.25) and
                                max(gold_step, tip_step) <= max(45., hand.scale))
            jewel.metrics = dict(verification='fingertip_and_visible_gold_motion',
                associated_hand=link['hand'], fingertip_id=link['tip'],
                displacement_px=jewel.displacement, threshold_px=self.threshold,
                motion_cosine=cosine, relative_drift_px=drift)
            if coherent:
                if self.vote(jewel, 'fingertip_motion', t) and jewel.displacement > self.threshold:
                    jewel.pickup_at, jewel.detected_at = jewel.vote_since, t
                    jewel.verification_status = 'coherent_motion_verified'
                    jewel.pickup_basis = 'same_fingertip_and_jewel_motion'
                    self.change(jewel, 'MOVEMENT_TRIGGERED', t,
                        'Same jewel and fingertip moved together across 3 observations / 0.45 s; lifting and insertion unverified.')
                else:
                    self.change(jewel, 'VERIFYING', t, f'Jewel follows fingertip: {jewel.votes}/3 motion observations; waiting for departure.')
            else:
                jewel.votes, jewel.vote_kind = 0, ''
                self.verification_pending(jewel, t, 'Pending pickup: need matching visible jewel and fingertip movement.')
                if t-jewel.contact_at >= 1.5 and jewel.displacement <= self.threshold*.5:
                    self.change(jewel, 'PICKUP_CANCELLED', t, 'Jewel stayed at home; candidate cancelled.')
                    self.links.pop(jewel.id, None)
                    reserved.discard(link['hand'])
                    self.wait_for_exit.add(jewel.id)
        self.message = 'Fingertip + jewel motion | proximity records; same hand/jewel motion across 3 observations / 0.45 s verifies movement; hidden gold stays pending.'


def make_engine(mode, emit, save_reference=None, cog_threshold=20.):
    if mode == 'cog':
        return COGMovementEngine(emit, save_reference, cog_threshold)
    if mode not in ('previous', 'home'):
        raise ValueError('Unknown trigger mode: '+str(mode))
    return (HomePositionEngine if mode == 'home' else PickupEngine)(emit, save_reference)

class Models:
    def __init__(self, original, log=None, load_bag=True):
        from standalone.tracking import GoldPredictor
        from standalone.models import PacketBagDetector
        self.gold = GoldPredictor()
        self.bag = PacketBagDetector(ROOT/'model onnx') if load_bag else None
        self.original = original
        self.hand = None
        self.keypoints = None
        self.log = log or (lambda message: None)
        self.identity_error = ''
        self.reset_tracking()

    def enable_bag(self):
        if self.bag is None:
            from standalone.models import PacketBagDetector
            self.bag = PacketBagDetector(ROOT/'model onnx')
        return self.bag

    def enable_hands(self):
        if self.hand is None:
            from standalone.models import Segmenter
            self.hand = Segmenter(ROOT/'model onnx'/'handseg model.onnx', .3)
        return self.hand

    def detect_hands(self, frame, roi_mask):
        # Inference always sees the full frame. Filter by actual mask overlap
        # afterward, before both home verification and overlay rendering.
        detections = self.enable_hands().detect(frame)
        return [hand for hand in detections
                if np.any((hand.mask > 0) & (roi_mask > 0))]

    def enable_keypoints(self):
        if self.keypoints is None:
            from standalone.hand_keypoints import HandKeypoints
            self.keypoints = HandKeypoints(ROOT/'model onnx'/'hand_landmarker.task')
        return self.keypoints

    def detect_keypoints(self, frame, roi_mask):
        return [hand for hand in self.enable_keypoints().detect(frame)
                if np.any((hand.mask > 0) & (roi_mask > 0))]

    def close_keypoints(self):
        if self.keypoints is not None:
            self.keypoints.close()
            self.keypoints = None

    def close_tracking(self):
        tracker = getattr(self, 'tracker', None)
        if tracker is None:
            return
        # Invalidate any background gallery publication for this old assignment.
        with tracker.lock:
            tracker.running = False
            tracker.flow_revision += 1
            tracker.identity_training_token = None
        thread = tracker.id_training_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.)
        # A subsequent reset uses the same serial build lock, so two gallery
        # preparations cannot compete even if the old one is still finishing.

    def reset_tracking(self):
        old = getattr(self, 'tracker', None)
        self.close_tracking()
        original = self.original

        class PickupTracker(original.Worker):
            # Preserve assignment, learn_ids, update_tracks, matching, recovery,
            # evaluate and hand/spot/motion logic from the original Worker.
            # This experiment has no touchstone or material-testing workflow.
            def stone_present(self, frame):
                return True
            def update_rubbing(self, t):
                pass
            def update_acid(self, t):
                pass
            def check_done(self, *args):
                pass
            def save_capture(self, frame, jewels):
                pass

        self.tracker = PickupTracker(0, [], continuous=True)
        if old is not None:
            self.tracker.identity_training_lock = old.identity_training_lock
        self.identity_error = ''
        if not self.tracker.identity_slic or self.tracker.id_model is None:
            raise RuntimeError('Original SLIC identity tracking is unavailable. Run with the project .venv-slic Python.')

        def event(message, level):
            if 'gallery unavailable' in message:
                self.identity_error = message
            if message.startswith('Initial IDs/crops ready after'):
                message = 'Initial jewel IDs/crops ready; original tracker assignment locked.'
            self.log('Tracker: '+message)
        self.tracker.event.connect(event, original.QtCore.Qt.DirectConnection)

    def identity_status(self):
        if self.identity_error:
            return 'identity unavailable (see log)'
        if self.tracker.mode != 'LOCKED':
            return 'original assignment preparing'
        return ('SLIC identity ready' if self.tracker.id_model.ready else
                'SLIC identity gallery preparing')

    def process(self, frame, roi, t):
        """One original gold pass, followed by the original lock/update path."""
        from continuous_testing import visibility
        tracker = self.tracker
        tracker.frame_wh = (frame.shape[1], frame.shape[0])
        tracker.roi = np.asarray(roi, np.int32)
        tracker.recent.append(frame)
        raw, rejected = tracker.detect(self.gold, frame, tracker.roi, captured_at=t)
        # Mirrors Worker's run loop, without stone/rubbing/acid processing.
        with tracker.lock:
            tracker.rejected = rejected
            if tracker.mode == 'ARMING':
                tracker.try_lock(raw, t, frame)
            elif tracker.mode == 'LOCKED':
                tracker.update_tracks(raw, t, frame)
            snapshot, ids = {}, set()
            for tr in tracker.tracks:
                row = dict(id=tr.id, box=tuple(map(int, tr.box)), detected_box=tuple(map(int, tr.det_box)), seen=tr.det_seen, det_seen=tr.det_seen,
                           state=tr.state, spot=tr.spot, in_hand=getattr(tr, 'in_hand', False))
                row['visibility'] = visibility(row, t, bool(getattr(tracker, 'moving_now', False)))
                snapshot[tr.id] = row
                if row['visibility'] in ('IN_HAND', 'UNDER_HAND'):
                    ids.add(tr.id)
        detections = []
        for detection in raw:
            full = np.zeros(frame.shape[:2], np.uint8)
            x, y = map(int, detection.origin)
            h, w = detection.mask.shape
            full[y:y+h, x:x+w] = detection.mask
            detections.append(SimpleNamespace(mask=full, box=tuple(map(int, detection.box)), confidence=detection.conf, raw=detection))
        return detections, snapshot, ids


class CandidateClips:
    """Debug clips start on each candidate, including its eventual resolution.

    JSONL preserves source timestamps. Frame repetition only fills short sampling
    gaps; longer gaps and 60s durations start a separate segment, never an invented
    continuous transfer. Debug videos are at most 960px wide; source frames,
    verification and clean trigger-start photographs retain capture resolution.
    """
    def __init__(self, folder, fps):
        self.folder, self.fps = folder, fps
        self.writers = {}
        self.counts = {}
        self.buffer = deque()

    def write_frame(self, entry, frame, t, state):
        slot = round((t-entry['start'])*self.fps)
        if slot <= entry['slot']:
            return
        duplicates = max(0, slot-entry['slot']-1)
        for _ in range(duplicates):
            entry['writer'].write(entry['previous'])
        entry['writer'].write(frame)
        with (self.folder/'candidate_frames.jsonl').open('a') as handle:
            handle.write(json.dumps(dict(video=entry['name'], slot=slot, source_time=t,
                                        state=state, repeated_previous_frames=duplicates,
                                        video_size=[frame.shape[1], frame.shape[0]]))+'\n')
        entry.update(slot=slot, last_t=t, previous=frame)

    def update(self, frame, jewels, t):
        original_frame = frame
        frame = small_frame(frame)  # Resize once for all clips and the shared pre-roll.

        while self.buffer and t-self.buffer[0][0] > 1.:
            self.buffer.popleft()
        for jewel in jewels:
            active = jewel.state in ('PICKUP_CANDIDATE', 'PICKUP_DETECTED', 'UNCERTAIN', 'VERIFYING', 'MOVEMENT_TRIGGERED') and jewel.contact_at is not None
            entry = self.writers.get(jewel.id)
            if entry and (t-entry['last_t'] > 2. or t-entry['start'] > 60.):
                entry['writer'].release()
                del self.writers[jewel.id]
                entry = None
            if entry is None and active:
                self.counts[jewel.id] = self.counts.get(jewel.id, 0)+1
                name = f'J{jewel.id}_candidate_{self.counts[jewel.id]:03d}.avi'
                writer = cv2.VideoWriter(str(self.folder/name), cv2.VideoWriter_fourcc(*'MJPG'), self.fps,
                                         (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    writer.release()
                    raise OSError('Could not create candidate debug video.')
                entry = dict(writer=writer, name=name, start=self.buffer[0][0] if self.buffer else t, last_t=t, slot=-1, previous=frame)
                self.writers[jewel.id] = entry
                for when, previous in self.buffer:
                    self.write_frame(entry, previous, when, 'PRE_ROLL')
                if not cv2.imwrite(str(self.folder/name.replace('.avi', '_start.jpg')), original_frame):
                    raise OSError('Could not save clean candidate start frame.')
            if entry is None:
                continue
            self.write_frame(entry, frame, t, jewel.state)
            if not active:
                entry['writer'].release()
                del self.writers[jewel.id]
        self.buffer.append((t, frame))

    def close(self):
        for entry in self.writers.values():
            entry['writer'].release()
        self.writers.clear()
        self.buffer.clear()


def dependencies():
    if not (TRACKER/'jewel_tracker_5.py').is_file():
        raise RuntimeError('Set JEWEL_TRACKER_DIR to the directory containing jewel_tracker_5.py.')
    import jewel_tracker_5 as original  # preserves torch-before-Qt loading order
    import torch
    import ultralytics.utils.torch_utils as torch_utils
    torch.set_num_threads(2)
    torch_utils.NUM_THREADS = 2
    cv2.setNumThreads(2)
    from PyQt5 import QtCore, QtGui, QtWidgets
    for name in ('QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_FONTDIR'):
        if '/cv2/' in os.environ.get(name, '').replace('\\', '/'):
            os.environ.pop(name, None)
    os.environ.setdefault('QT_QPA_PLATFORM_PLUGIN_PATH', QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath))
    return original, QtCore, QtGui, QtWidgets


def window_class(original, QtCore, QtGui, QtWidgets):
    class Worker(QtCore.QThread):
        result = QtCore.pyqtSignal(object)
        log = QtCore.pyqtSignal(str)
        failure = QtCore.pyqtSignal(str)
        notice = QtCore.pyqtSignal(str)

        def __init__(self, source, output, target_fps, mode='previous', cog_threshold=20.):
            super().__init__()
            self.source, self.output, self.target_fps = source, output, target_fps
            self.cog_threshold = cog_threshold
            self.requested_mode = mode
            self.mode = mode
            self.stop_event = threading.Event()
            self.reset_event = threading.Event()
            self.grabber = None
            self.engine = None

        def stop(self):
            self.stop_event.set()
            if self.grabber:
                self.grabber.stop()

        def run(self):
            from standalone.settings import tracker_camera, tracker_roi, read_json
            from standalone.camera import make_grabber
            cap = None
            clips = None
            models = None
            try:
                self.output.mkdir(parents=True, exist_ok=False)
                self.log.emit(('Loading gold INT8 and hand landmarks on CPU; trigger mode: ' if self.mode == 'cog' else
                               'Loading gold INT8 and bag ONNX/filter on CPU; trigger mode: ')+self.mode)
                self.notice.emit('Loading existing CPU models; camera will open next…')
                models = Models(original, self.log.emit, load_bag=self.mode != 'cog')
                if self.mode == 'home':
                    models.enable_hands()
                elif self.mode == 'cog':
                    models.enable_keypoints()
                video = isinstance(self.source, str) and Path(self.source).is_file()
                camera = tracker_camera(0 if video else self.source)
                reference_size = tuple(map(int, camera.get('resolution', '2560x1440').split('x')))
                packet_settings = read_json(ROOT/'packet_ui_settings.json')
                saved_roi = packet_settings.get('roi', []) if packet_settings.get('roi_override') else tracker_roi()
                if len(saved_roi) < 3:
                        raise ValueError('Save a valid bench ROI in the main application/Jewel Tracker first. No full-frame fallback is used.')
                if video:
                    cap = cv2.VideoCapture(self.source)
                    if not cap.isOpened():
                        raise ValueError('Cannot open video.')
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    if not 0 < fps <= 240:
                        raise ValueError('Video has no usable FPS timestamps.')
                    stride = max(1, math.ceil(fps/self.target_fps))
                else:
                    self.grabber = make_grabber(self.source, camera)
                    self.grabber.start()
                session = 0
                folder = self.output

                def display_time(value):
                    if value is None:
                        return 'not observed'
                    return (f'{value:.3f}s' if video else
                            datetime.fromtimestamp(value, TIMEZONE).strftime('%H:%M:%S.%f')[:-3]+' IST')

                def event(data):
                    data = dict(data, trigger_mode=self.mode)
                    with (folder/'events.jsonl').open('a') as handle:
                        handle.write(json.dumps(data)+'\n')
                    stamp = f"video {data['time']:.3f}s" if video else datetime.fromtimestamp(data['time'], TIMEZONE).strftime('%H:%M:%S.%f')[:-3]+' IST'
                    timing = (f" | observed pickup: {display_time(data.get('pickup_at'))}; verified: {display_time(data.get('verified_at'))}"
                              if data['state'] in ('PICKUP_DETECTED', 'PICKUP_CONFIRMED', 'PICKUP_CANCELLED') else '')
                    self.log.emit(f"{stamp} | {data.get('jewel', '')} {data['state']} | {data.get('reason', '')}"+timing)

                def save_reference(jewel):
                    if not cv2.imwrite(str(folder/f'J{jewel.id}_reference.jpg'), jewel.reference):
                        raise OSError('Could not save reference crop.')
                    if not cv2.imwrite(str(folder/f'J{jewel.id}_reference_mask.png'), jewel.gold_mask):
                        raise OSError('Could not save reference segmentation mask.')
                    if not cv2.imwrite(str(folder/f'J{jewel.id}_verification_mask.png'), jewel.verification_mask):
                        raise OSError('Could not save Otsu verification mask.')
                    if not cv2.imwrite(str(folder/f'J{jewel.id}_verification_edges.png'), jewel.verification_edges):
                        raise OSError('Could not save verification edge reference.')
                    (folder/f'J{jewel.id}_reference.json').write_text(json.dumps(dict(
                        id=jewel.id, position=jewel.position.tolist(), box=list(jewel.box), crop_box=list(jewel.crop_box),
                        coordinates='capture_frame', assigned_at=jewel.at, home_cog=jewel.home_cog.tolist(), trigger_mode=self.mode,
                        verification=jewel.verification_info), indent=2))

                self.engine = make_engine(self.mode, event, save_reference, self.cog_threshold)
                clips = CandidateClips(folder, self.target_fps)
                index, previous_t, shape = -1, None, None
                last_wall, started_wall = 0., time.perf_counter()
                bags, bag_t = [], -math.inf
                connection_message = None
                last_connection_notice = -math.inf
                while not self.stop_event.is_set():
                    if self.reset_event.is_set():
                        self.reset_event.clear()
                        session += 1
                        folder = self.output/f'reset_{session:03d}'
                        folder.mkdir()
                        clips.close()
                        clips = CandidateClips(folder, self.target_fps)
                        self.mode = self.requested_mode
                        if self.mode == 'home':
                            models.enable_hands()
                        else:
                            models.hand = None
                        if self.mode == 'cog':
                            models.enable_keypoints()
                        else:
                            models.close_keypoints()
                        if self.mode == 'cog':
                            models.bag = None
                        self.engine = make_engine(self.mode, event, save_reference, self.cog_threshold)
                        models.reset_tracking()
                        bags, bag_t = [], -math.inf
                        if video:
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            index, previous_t, started_wall = -1, None, time.perf_counter()
                        self.log.emit('RESET: old references/events preserved; clear hands and lay out jewels again.')
                    if video:
                        ok = cap.grab()
                        index += 1
                        if not ok:
                            self.log.emit('Video ended. Results retained; use Start for another source.')
                            break
                        if index % stride:
                            continue
                        ok, source_frame = cap.retrieve()
                        t = index/fps
                        if not ok:
                            raise OSError('Video frame decode failed.')
                        delay = started_wall+t-time.perf_counter()
                        if delay > 0 and self.stop_event.wait(delay):
                            break
                    else:
                        delay = 1/self.target_fps-(time.perf_counter()-last_wall)
                        if delay > 0 and self.stop_event.wait(delay):
                            break
                        source_frame, t = self.grabber.latest(previous_t)
                        if source_frame is None:
                            now = time.monotonic()
                            if now-last_connection_notice >= 1.:
                                applying = time.time() < getattr(self.grabber, 'af_busy_until', 0.)
                                message = (self.grabber.problem or
                                           ('Applying saved camera controls; waiting for the first frame…' if applying else
                                            f'Waiting for camera {self.source}. Check the device number and whether another application is using it.'))
                                self.notice.emit(message)
                                if message != connection_message:
                                    self.log.emit(message)
                                    connection_message = message
                                last_connection_notice = now
                            self.stop_event.wait(.03)
                            continue
                        if time.time()-t > .8:
                            self.log.emit('Stale camera frame rejected.')
                            continue
                    sample_gap = t-previous_t if previous_t is not None else None
                    previous_t, last_wall = t, time.perf_counter()
                    frame = source_frame  # Original Worker thresholds/identity crops use capture pixels.
                    if shape is not None and shape != frame.shape:
                        raise ValueError('Resolution changed. Stop and restart; original positions cannot be reused.')
                    shape = frame.shape
                    if not (folder/'session.json').exists():
                        (folder/'session.json').write_text(json.dumps(dict(
                            source_kind='video' if video else 'camera',
                            camera_index=self.source if isinstance(self.source, int) else None,
                            saved_resolution=list(reference_size), capture_size=[source_frame.shape[1], source_frame.shape[0]],
                            working_size=[frame.shape[1], frame.shape[0]], saved_roi=saved_roi,
                            timestamp_kind='video_seconds' if video else 'unix_seconds',
                            max_inference_fps=self.target_fps, hand_inference='onnx_handseg' if self.mode == 'home' else 'mediapipe_hand_landmarker' if self.mode == 'cog' else 'disabled_no_hand_model',
                            trigger_mode=self.mode, trigger='fingertip_proximity_and_coherent_jewel_motion' if self.mode == 'cog' else 'hand_contact_cog_empty_home' if self.mode == 'home' else 'original_tracker_cues_plus_independent_otsu',
                            cog_threshold_px=self.cog_threshold if self.mode == 'cog' else None,
                            tracker='original_Worker_try_lock_update_tracks_SLIC_identity',
                            verification='fingertip_and_visible_gold_motion' if self.mode == 'cog' else 'fixed_reference_observer_only', debug_video_max_width=960), indent=2))
                    scale = np.array([frame.shape[1]/reference_size[0], frame.shape[0]/reference_size[1]])
                    roi = np.rint(np.asarray(saved_roi)*scale).astype(np.int32)
                    roi_mask = np.zeros(frame.shape[:2], np.uint8)
                    cv2.fillPoly(roi_mask, [roi], 255)
                    inference_start = time.perf_counter()
                    # Serial, bounded CPU execution avoids several models oversubscribing six cores.
                    timings = {}
                    stage_start = time.perf_counter()
                    gold, tracked, legacy_ids = models.process(frame, roi, t)
                    timings['tracking_ms'] = round((time.perf_counter()-stage_start)*1000, 2)
                    stage_start = time.perf_counter()
                    if self.mode != 'cog' and t-bag_t >= 1.:
                        bags, bag_t = models.enable_bag().detect(frame), t
                    timings['packet_ms'] = round((time.perf_counter()-stage_start)*1000, 2)
                    stage_start = time.perf_counter()
                    hands = (models.detect_hands(frame, roi_mask) if self.mode == 'home' else
                             models.detect_keypoints(frame, roi_mask) if self.mode == 'cog' else [])
                    timings['hand_ms'] = round((time.perf_counter()-stage_start)*1000, 2)
                    stage_start = time.perf_counter()
                    if models.tracker.mode == 'LOCKED':
                        if self.mode in ('home', 'cog'):
                            self.engine.adopt_tracks(frame, models.tracker.tracks, bags, roi_mask, t, hands)
                        else:
                            self.engine.adopt_tracks(frame, models.tracker.tracks, bags, roi_mask, t)
                        self.engine.update(frame, gold, hands, bags, roi_mask, t, tracked, legacy_ids)
                    else:
                        self.engine.message = (f'Original assignment: {models.tracker.arm_count} jewels, '
                                               f'{models.tracker.stable:.0%}; keep jewels still for initial capture.')
                    timings['verification_ms'] = round((time.perf_counter()-stage_start)*1000, 2)
                    stage_start = time.perf_counter()
                    clips.update(frame, self.engine.jewels, t)
                    timings['recording_ms'] = round((time.perf_counter()-stage_start)*1000, 2)
                    with (folder/'observations.jsonl').open('a') as handle:
                        handle.write(json.dumps(dict(time=t, trigger_mode=self.mode, timings_ms=timings, sample_gap_s=sample_gap, existing_trigger_ids=sorted(legacy_ids),
                            reference_scene_uncertain=self.engine.global_uncertain,
                            tracker_mode=models.tracker.mode, identity_status=models.identity_status(),
                            original_tracks=list(tracked.values()),
                            jewels=[dict(id=j.id, state=j.state, tracker_visibility=j.tracker_visibility,
                                         original_box=list(j.box), live_box=list(j.live_box) if j.live_box else None,
                                         live_source=j.live_source, live_seen=j.live_seen,
                                         trigger=j.trigger_source, verification_reference=j.verification_info,
                                         contact_at=j.contact_at, pickup_at=j.pickup_at, detected_at=j.detected_at,
                                         verified_at=j.verified_at, verification_status=j.verification_status,
                                         pickup_basis=j.pickup_basis, destination_hint=j.destination_hint,
                                         reason=j.reason, verification_votes=j.votes, vote_kind=j.vote_kind,
                                         metrics=j.metrics) for j in self.engine.jewels]))+'\n')
                    elapsed = time.perf_counter()-inference_start
                    display = frame.copy()
                    overlay_scale = max(1., frame.shape[1]/960.)
                    overlay_line = max(1, round(overlay_scale))
                    cv2.polylines(display, [roi], True, (255, 160, 0), overlay_line)
                    for bag in bags:
                        cv2.rectangle(display, bag.box[:2], bag.box[2:], (255, 160, 0), overlay_line)
                    for hand in hands:
                        if hasattr(hand, 'points'):
                            from standalone.hand_keypoints import draw_hand
                            draw_hand(display, hand, overlay_line)
                        else:
                            contours, _ = cv2.findContours(hand.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                            cv2.drawContours(display, contours, -1, (220, 80, 220), overlay_line)
                    for detection in gold:
                        contours, _ = cv2.findContours(detection.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(display, contours, -1, (0, 220, 220), overlay_line)
                    rows = []
                    for jewel in self.engine.jewels:
                        colors = {'PICKUP_CONFIRMED': (60, 220, 60), 'PICKUP_CANCELLED': (180, 220, 80),
                                  'PICKUP_CANDIDATE': (0, 220, 255), 'PICKUP_DETECTED': (255, 220, 30),
                                  'UNCERTAIN': (0, 140, 255), 'VERIFYING': (230, 180, 60)}
                        color = colors.get(jewel.state, (255, 200, 60))
                        if self.mode in ('home', 'cog'):
                            home_point = tuple(np.rint(jewel.home_cog).astype(int))
                            cv2.drawMarker(display, home_point, (180, 180, 180), cv2.MARKER_CROSS, round(10*overlay_scale), overlay_line)
                            if jewel.current_cog is not None:
                                live_point = tuple(np.rint(jewel.current_cog).astype(int))
                                cv2.circle(display, live_point, round(3*overlay_scale), color, -1)
                                cv2.line(display, home_point, live_point, color, overlay_line)
                                if self.mode == 'cog' and jewel.hand_position is not None:
                                    tip_point = tuple(np.rint(jewel.hand_position).astype(int))
                                    cv2.line(display, tip_point, live_point, (0, 240, 240), overlay_line)
                        original.dashed_rect(display, jewel.box[:2], jewel.box[2:], (135, 135, 135), dash=max(5, round(5*overlay_scale)))
                        cv2.putText(display, f'J{jewel.id} bench reference', (jewel.box[0], min(display.shape[0]-5, jewel.box[3]+round(13*overlay_scale))), cv2.FONT_HERSHEY_SIMPLEX, .36*overlay_scale, (180, 180, 180), overlay_line)
                        if jewel.live_box:
                            live = jewel.live_box
                            if jewel.live_source in ('gold', 'gold at reference') and jewel.live_seen == t:
                                cv2.rectangle(display, live[:2], live[2:], color, 2*overlay_line)
                            else:
                                original.dashed_rect(display, live[:2], live[2:], color)
                            cv2.putText(display, f'J{jewel.id} {jewel.tracker_visibility} | {jewel.state} | {jewel.live_source}',
                                        (live[0], max(round(16*overlay_scale), live[1]-round(6*overlay_scale))), cv2.FONT_HERSHEY_SIMPLEX, .42*overlay_scale, color, overlay_line)
                        stamp = f'{jewel.at:.3f}s' if video else datetime.fromtimestamp(jewel.at, TIMEZONE).strftime('%H:%M:%S.%f')[:-3]+' IST'
                        rows.append([f'J{jewel.id}', jewel.state, jewel.tracker_visibility, stamp,
                                     display_time(jewel.pickup_at), display_time(jewel.verified_at),
                                     jewel.trigger_source+' | '+jewel.verification_status+' | '+jewel.reason+' '+(json.dumps(jewel.metrics) if jewel.metrics else '')])
                    observed_fps = f'{1/sample_gap:.1f}' if sample_gap and sample_gap > 0 else '-'
                    status = (f'{self.engine.message} | {observed_fps} observed FPS | {elapsed*1000:.0f} ms processing | '
                              f'capture {source_frame.shape[1]}×{source_frame.shape[0]} / work {frame.shape[1]}×{frame.shape[0]} | {models.identity_status()} | mode: {self.mode} | t={t:.3f}')
                    self.result.emit(dict(frame=small_frame(display), rows=rows, status=status))
                self.log.emit('Stopped. Reference images and events: '+str(self.output))
            except Exception as exc:
                self.failure.emit(str(exc))
            finally:
                if models:
                    models.close_tracking()
                    models.close_keypoints()
                if clips:
                    clips.close()
                if cap:
                    cap.release()
                if self.grabber:
                    self.grabber.stop()
                    self.grabber.join(timeout=4)

    class Window(QtWidgets.QWidget):
        def __init__(self, args):
            super().__init__()
            from standalone.settings import tracker_source
            self.args, self.worker, self.last_image = args, None, None
            self.selected_mode = getattr(args, 'trigger_mode', 'previous')
            self.setWindowTitle('Standalone jewellery pickup trigger test')
            self.resize(1250, 900)
            layout = QtWidgets.QVBoxLayout(self)
            bar = QtWidgets.QHBoxLayout(); layout.addLayout(bar)
            bar.addWidget(QtWidgets.QLabel('Camera:'))
            self.camera = QtWidgets.QSpinBox(); self.camera.setRange(0, 15); bar.addWidget(self.camera)
            saved = tracker_source()
            try:
                packet_source = json.loads((ROOT/'packet_ui_settings.json').read_text()).get('source')
                if packet_source is not None and str(packet_source).isdigit():
                    saved = str(packet_source)
            except (OSError, ValueError, TypeError):
                pass
            self.available_cameras = capture_devices()
            automatic_selection = None
            indices = [index for index, _ in self.available_cameras]
            if args.source is None and saved.isdigit() and self.available_cameras and int(saved) not in indices:
                automatic_selection = f'Saved camera {saved} is unavailable; selected camera {indices[0]} for this test only.'
                saved = str(indices[0])
            if saved.isdigit() and int(saved) <= 15:
                self.camera.setValue(int(saved))
            self.source = QtWidgets.QLineEdit(args.source if args.source is not None else saved)
            bar.addWidget(self.source, 1)
            for text, callback in [('Open camera', self.use_camera), ('Open video', self.open_video),
                                   ('Start', self.start), ('Stop', self.stop), ('Reset', self.reset)]:
                button = QtWidgets.QPushButton(text); button.clicked.connect(callback); bar.addWidget(button)
            mode_bar = QtWidgets.QHBoxLayout(); layout.addLayout(mode_bar)
            self.mode_buttons = {}
            self.mode_group = QtWidgets.QButtonGroup(self)
            self.mode_group.setExclusive(True)
            for mode, label in (('previous', 'Previous logic'), ('home', 'Home-position logic'), ('cog', 'Fingertip + jewel movement')):
                button = QtWidgets.QPushButton(label)
                button.setCheckable(True)
                button.setChecked(mode == self.selected_mode)
                button.setStyleSheet('QPushButton:checked {background: #2563eb; color: white;}')
                self.mode_group.addButton(button)
                button.clicked.connect(lambda checked, selected=mode: self.select_mode(selected))
                self.mode_buttons[mode] = button
                mode_bar.addWidget(button)
            self.video = QtWidgets.QLabel('Start with all jewels on the bed and hands clear.')
            self.video.setMinimumSize(640, 360); self.video.setAlignment(QtCore.Qt.AlignCenter)
            self.video.setStyleSheet('background:#15202b;color:#ddd;'); layout.addWidget(self.video, 1)
            self.status = QtWidgets.QLabel('Original positions never move. Missing/covered evidence stays UNCERTAIN.')
            self.status.setWordWrap(True); layout.addWidget(self.status)
            help_text = QtWidgets.QLabel('Original Worker: persistent assignment, SLIC identity, matching, recovery and IN_HAND / UNDER_HAND. '
                                       'Green tick requires independent empty-spot evidence; IN_HAND / UNDER_HAND alone cannot confirm pickup. Fully covered/ambiguous spots stay uncertain. '
                                       'Home mode also allows >=50% exposed empty home plus the same visible jewel displaced across 3 observations / 0.45 s; hand motion never substitutes for jewel motion. '
                                       'Fingertip mode records on proximity and verifies matching fingertip/jewel movement across frames; it does not verify lifting or insertion. '
                                       'Switching modes resets references and preserves recordings. Gray boxes are fixed bench references. '
                                       'Removal from the original position does not prove placement inside the packet.')
            help_text.setWordWrap(True); layout.addWidget(help_text)
            self.table = QtWidgets.QTableWidget(0, 7)
            self.table.setHorizontalHeaderLabels(['ID', 'State', 'Tracker cue', 'Event time', 'Observed pickup', 'Verified at', 'Reason / metrics'])
            self.table.horizontalHeader().setSectionResizeMode(6, QtWidgets.QHeaderView.Stretch)
            for column, width in enumerate((70, 180, 110, 120, 130, 130)):
                self.table.setColumnWidth(column, width)
            self.table.setMaximumHeight(180); layout.addWidget(self.table)
            self.logs = QtWidgets.QPlainTextEdit(); self.logs.setReadOnly(True)
            self.logs.setMaximumBlockCount(350); self.logs.setMaximumHeight(130); layout.addWidget(self.logs)
            if self.available_cameras:
                self.logs.appendPlainText('Available capture cameras: '+', '.join(f'{index}: {name}' for index, name in self.available_cameras))
            if automatic_selection:
                self.logs.appendPlainText(automatic_selection)
                self.status.setText(automatic_selection+' Click Open camera or Start.')
            if args.autostart:
                QtCore.QTimer.singleShot(0, self.start)

        def select_mode(self, mode):
            if mode == self.selected_mode:
                return
            self.selected_mode = mode
            self.logs.appendPlainText('Trigger mode: '+mode+'. Fresh home references required; previous recordings preserved.')
            if self.worker and self.worker.isRunning():
                self.worker.requested_mode = mode
                self.worker.reset_event.set()
                self.status.setText('Switching trigger mode; resetting references. Keep jewels stable and hands clear.')
            else:
                self.status.setText('Selected trigger mode: '+mode+'. Click Start.')

        def use_camera(self):
            self.source.setText(str(self.camera.value()))
            self.start()

        def open_video(self):
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Video', str(ROOT), 'Videos (*.mp4 *.avi *.mkv *.mov)')
            if path:
                self.source.setText(path)

        def start(self):
            if self.worker and self.worker.isRunning():
                self.logs.appendPlainText('Stop the current source before changing it. Reset keeps the same source.')
                return
            value = self.source.text().strip()
            if not value:
                return
            source = int(value) if value.isdigit() else value
            if isinstance(source, int) and sys.platform.startswith('linux'):
                devices = capture_devices()
                if source not in [index for index, _ in devices]:
                    options = ', '.join(str(index) for index, _ in devices) or 'none (check USB connection and access permissions)'
                    message = f'Camera {source} is not an available video-capture device. Available camera indices: {options}.'
                    self.logs.appendPlainText('ERROR: '+message)
                    self.status.setText(message)
                    return
            folder = self.args.output/datetime.now(TIMEZONE).strftime('%Y%m%d_%H%M%S_%f')
            previous = self.worker
            self.worker = Worker(source, folder, self.args.fps, self.selected_mode, getattr(self.args, 'cog_threshold', 20.))
            if previous:
                previous.deleteLater()
            self.worker.result.connect(self.show_result)
            self.worker.log.connect(self.logs.appendPlainText)
            self.worker.notice.connect(self.status.setText)
            self.worker.failure.connect(self.show_error)
            self.table.setRowCount(0)
            self.worker.start()

        def show_error(self, message):
            self.logs.appendPlainText('ERROR: '+message)
            self.status.setText('ERROR: '+message)

        def stop(self):
            if self.worker:
                self.worker.stop()

        def reset(self):
            if self.worker and self.worker.isRunning():
                self.worker.reset_event.set()
            else:
                self.logs.appendPlainText('Start a source first. Reset archives references and assigns new IDs.')

        def show_result(self, result):
            rgb = cv2.cvtColor(result['frame'], cv2.COLOR_BGR2RGB)
            h, w = rgb.shape[:2]
            self.last_image = QtGui.QImage(rgb.data, w, h, rgb.strides[0], QtGui.QImage.Format_RGB888).copy()
            self.paint_frame()
            self.status.setText(result['status'])
            self.table.setRowCount(len(result['rows']))
            for row, values in enumerate(result['rows']):
                state = values[1]
                styles = {
                    'MOVEMENT_TRIGGERED': ('▶ ', '#dbeafe', '#1e40af', 'Same fingertip and visible jewel moved together; lifting and insertion remain unverified.'),
                    'MOVEMENT_ENDED': ('', '#f3f4f6', '#374151', 'COG returned close to its initial position.'),
                    'PICKUP_DETECTED': ('✓ ', '#dcfce7', '#166534', 'Pickup detected. Original-spot verification is still separate.'),
                    'PICKUP_CONFIRMED': ('✓ ', '#bbf7d0', '#14532d', 'Pickup removal verified at the original position.'),
                    'PICKUP_CANCELLED': ('× ', '#fee2e2', '#991b1b', 'Candidate cancelled: the jewel was found at its original position.'),
                    'PICKUP_CANDIDATE': ('', '#fef3c7', '#92400e', 'Candidate recorded; pickup is not yet detected.'),
                    'UNCERTAIN': ('', '#ffedd5', '#9a3412', 'Evidence is insufficient. Recording continues.'),
                    'VERIFYING': ('', '#dbeafe', '#1e40af', 'Collecting consistent observations; check the reason for progress.'),
                }
                for col, value in enumerate(values):
                    item = QtWidgets.QTableWidgetItem(value)
                    if col in (0, 1) and state in styles:
                        prefix, background, foreground, tooltip = styles[state]
                        if col == 0:
                            item.setText(prefix+value)
                        item.setBackground(QtGui.QColor(background))
                        item.setForeground(QtGui.QColor(foreground))
                        font = item.font(); font.setBold(True); item.setFont(font)
                        item.setToolTip(tooltip)
                    self.table.setItem(row, col, item)

        def paint_frame(self):
            if self.last_image is not None:
                self.video.setPixmap(QtGui.QPixmap.fromImage(self.last_image).scaled(self.video.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

        def resizeEvent(self, event):
            self.paint_frame()
            super().resizeEvent(event)

        def closeEvent(self, event):
            self.stop()
            if self.worker and self.worker.isRunning():
                self.logs.appendPlainText('Stopping capture/inference. Close again after it stops.')
                event.ignore()
                QtCore.QTimer.singleShot(400, self.close)
                return
            super().closeEvent(event)

    return Window


def self_test():
    """No weights/camera: original visibility cues, movement and independent verification."""
    shape = (240, 400, 3)
    frame = np.full(shape, 215, np.uint8)
    box = (80, 95, 130, 135)
    cv2.rectangle(frame, box[:2], box[2:], (20, 120, 180), -1)
    mask = np.zeros(shape[:2], np.uint8)
    mask[95:136, 80:131] = 255
    gold = SimpleNamespace(box=box, mask=mask, confidence=.95)
    roi = np.full(shape[:2], 255, np.uint8)
    empty = np.full_like(frame, 215)

    def observation(t, cue='VISIBLE', live=box, seen=None):
        return {1: dict(box=live, seen=t if seen is None else seen, visibility=cue)}

    def engine():
        events = []
        e = PickupEngine(events.append)
        for t in (0., .5, 1., 1.6):
            e.update(frame, [gold], [], [], roi, t)
        assert len(e.jewels) == 1 and e.jewels[0].verification_info['valid']
        e.scene_stable = lambda gray: True
        return e, events

    # Verification must adopt the original assignment, never renumber/reseed
    # tracks, snap a current box to its reference or stop a confirmed track.
    e, events = engine()
    live = (150, 95, 200, 135)
    snapshot = observation(1.8, 'VISIBLE', live)
    e.update(frame, [gold], [], [], roi, 1.8, snapshot)
    assert e.jewels[0].live_box == live, 'Reference checks must not snap the live tracker box'
    assert snapshot[1]['box'] == live
    e.jewels[0].state = 'PICKUP_CONFIRMED'
    later = (170, 95, 220, 135)
    e.update(frame, [gold], [], [], roi, 2., observation(2., 'VISIBLE', later))
    assert e.jewels[0].live_box == later, 'Confirmed jewels must still follow original tracking'
    e = PickupEngine(lambda event: None)
    local_mask = region(mask, box).copy()
    original_track = SimpleNamespace(id=7, det_box=box, det_mask=local_mask, det_origin=box[:2])
    saved_mask = local_mask.copy()
    e.adopt_tracks(frame, [original_track], [], roi, 0.)
    assert len(e.jewels) == 1 and e.jewels[0].id == 7
    np.testing.assert_array_equal(original_track.det_mask, saved_mask)
    assert original_track.det_box == box

    # Regression: moving J2 changes J1's reflections, but J1's metal edges
    # and position remain. Neither mode may start a J1 candidate from that
    # brightness change. Actual J2 departure must still verify normally.
    second_box, moved_box = (200, 95, 250, 135), (300, 95, 350, 135)
    initial = frame.copy()
    cv2.rectangle(initial, second_box[:2], second_box[2:], (20, 120, 180), -1)
    second_mask = np.zeros(shape[:2], np.uint8)
    second_mask[95:136, 200:251] = 255
    second = SimpleNamespace(box=second_box, mask=second_mask)
    moving = empty.copy()
    cv2.rectangle(moving, box[:2], box[2:], (0, 70, 130), -1)
    cv2.rectangle(moving, moved_box[:2], moved_box[2:], (20, 120, 180), -1)
    moved_mask = np.zeros(shape[:2], np.uint8)
    moved_mask[95:136, 300:351] = 255
    moved_gold = SimpleNamespace(box=moved_box, mask=moved_mask)
    for engine_type in (PickupEngine, HomePositionEngine):
        events = []
        e = engine_type(events.append)
        e.set_references(initial, [(1, gold), (2, second)], [], roi, 0.)
        e.scene_stable = lambda gray: True
        reflected = e.compare(e.jewels[0], moving)
        assert reflected['changed'] > .30 and reflected['edge_retention'] > .65
        assert not reflected['present'] and not reflected['removed']
        for t in (1., 1.25, 1.6):
            tracked = {1: dict(box=box, seen=t, visibility='VISIBLE'),
                       2: dict(box=moved_box, seen=t, visibility='VISIBLE')}
            e.update(moving, [gold, moved_gold], [], [], roi, t, tracked)
            assert e.jewels[0].state == 'ON_BENCH', 'Reflections on J1 must not start its pickup check'
        assert e.jewels[1].state == 'PICKUP_CONFIRMED'
        assert not any(event.get('jewel') == 'J1' and event['state'] == 'PICKUP_CANDIDATE'
                       for event in events)
        if engine_type is HomePositionEngine:
            e.update(moving, [gold, moved_gold], [gold], [], roi, 1.8, tracked)
            assert e.jewels[0].contact_at == 1.8, 'Real home contact must still start immediately'

    # A detector dropout with an unchanged reference is not even a new candidate.
    e, events = engine()
    for t in (1.8, 2., 2.3):
        e.update(frame, [], [], [], roi, t, observation(t, 'NOT_SEEN', seen=1.6))
    assert e.jewels[0].state == 'ON_BENCH'

    # Reused IN_HAND starts recording immediately; a stationary jewel cancels.
    e, events = engine()
    e.update(frame, [gold], [], [], roi, 1.8, observation(1.8, 'IN_HAND'))
    assert any(event['state'] == 'PICKUP_CANDIDATE' for event in events)
    assert e.jewels[0].state == 'VERIFYING' and e.jewels[0].votes == 1
    for t in (2., 2.3):
        e.update(frame, [gold], [], [], roi, t, observation(t, 'IN_HAND'))
    assert e.jewels[0].state == 'PICKUP_CANCELLED' and e.jewels[0].pickup_at is None
    count = len(events)
    e.update(frame, [gold], [], [], roi, 2.5, observation(2.5, 'IN_HAND'))
    assert len(events) == count, 'Persistent contact must not restart cancelled clips'

    # UNDER_HAND means covered, not picked up. Test a similar-luminance skin
    # occluder: grayscale comparison alone is inadequate, colour must reject it.
    e, events = engine()
    covered = frame.copy()
    covered[95:136, 80:131] = (175, 205, 240)
    assert not e.compare(e.jewels[0], covered)['removed']
    for t in (1.8, 2., 2.3):
        e.update(covered, [], [], [], roi, t, observation(t, 'UNDER_HAND', seen=1.6))
    assert any(x['state']=='PICKUP_CANDIDATE' for x in events)
    assert e.jewels[0].state == 'UNCERTAIN' and e.jewels[0].pickup_at is None
    # Clearing the original spot independently resolves a fully hidden pickup.
    for t in (2.5, 2.8, 3.1):
        e.update(empty, [], [], [], roi, t, observation(t, 'NOT_SEEN', seen=1.6))
    j = e.jewels[0]
    assert j.state == 'PICKUP_CONFIRMED' and j.pickup_at == 2.5 and j.verified_at == 3.1

    # No hand cues at all: a genuine reference change starts and resolves removal.
    e, events = engine()
    for t in (1.8, 2.1, 2.4):
        e.update(empty, [], [], [], roi, t)
    assert e.jewels[0].state == 'PICKUP_CONFIRMED'

    # Movement while the original spot is covered cannot give an early tick:
    # a merged carrier/J2 mask can falsely move an otherwise stationary ID.
    e, events = engine()
    for t, shift in ((1.8, 65), (2., 85), (2.2, 95), (2.4, 105)):
        moving = covered.copy() if t < 2.2 else empty.copy()
        live = (80+shift, 95, 130+shift, 135)
        cv2.rectangle(moving, live[:2], live[2:], (20, 120, 180), -1)
        moving_mask = np.zeros(shape[:2], np.uint8)
        moving_mask[95:136, 80+shift:131+shift] = 255
        e.update(moving, [SimpleNamespace(mask=moving_mask)], [], [], roi, t,
                 observation(t, 'IN_HAND', live))
        if t <= 2.:
            assert e.jewels[0].pickup_at is None, 'Covered original anchor cannot produce a green tick'
    j = e.jewels[0]
    assert j.state == 'PICKUP_DETECTED' and j.pickup_at == 2.2 and j.detected_at == 2.4
    assert j.verified_at is None
    e.update(covered, [], [], [], roi, 2.6, observation(2.6, 'UNDER_HAND', seen=2.4))
    assert j.state == 'PICKUP_DETECTED' and j.verification_status == 'uncertain'
    for t in (2.8, 3.1, 3.4):
        e.update(empty, [], [], [], roi, t)
    assert j.state == 'PICKUP_CONFIRMED' and j.pickup_at == 2.2 and j.verified_at == 3.4

    # A source gap / stale gold cannot create fast consensus.
    e, events = engine()
    for t in (1.8, 3.2):
        e.update(covered, [], [], [], roi, t, observation(t, 'IN_HAND', (160, 95, 210, 135)))
    assert e.jewels[0].pickup_at is None
    e.update(covered, [], [], [], roi, 3.4, observation(3.4, 'UNDER_HAND', (190, 95, 240, 135), seen=3.2))
    assert e.jewels[0].pickup_at is None

    # A J1 carrier passing J2 cannot give J2 a tick while its reference remains.
    e, events = engine()
    second_box = (250, 95, 300, 135)
    second_mask = np.zeros(shape[:2], np.uint8)
    second_mask[95:136, 250:301] = 255
    second_frame = frame.copy()
    cv2.rectangle(second_frame, second_box[:2], second_box[2:], (20, 120, 180), -1)
    crop = padded(second_box, shape, 15)
    e.jewels.append(Jewel(2, second_box, center(second_box), crop,
                         region(second_frame, crop).copy(), region(second_mask, crop).copy()))
    for t in (1.8, 2.1, 2.4):
        e.update(second_frame, [gold, SimpleNamespace(mask=second_mask)], [], [], roi, t,
                 {2:dict(box=(290, 95, 340, 135), seen=t, visibility='IN_HAND')})
    assert e.jewels[1].pickup_at is None and e.jewels[1].state == 'PICKUP_CANCELLED'

    # Camera motion, bag coverage, illumination change and invalid references
    # must not turn an empty detector result into a confirmed pickup.
    for guard in ('camera', 'bag', 'lighting', 'invalid'):
        e, events = engine()
        image = empty.copy()
        bags = []
        if guard == 'camera':
            e.scene_stable = lambda gray: False
        elif guard == 'bag':
            bags = [SimpleNamespace(mask=roi)]
        elif guard == 'lighting':
            image[:] = 150
        elif guard == 'invalid':
            e.jewels[0].verification_info['valid'] = False
        for t in (1.8, 2.1, 2.4, 2.7):
            e.update(image, [], [], bags, roi, t, observation(t, 'UNDER_HAND', seen=1.6))
        assert e.jewels[0].state != 'PICKUP_CONFIRMED', guard

    # Regression: Otsu mask preserves a bangle's hole despite a filled model mask.
    bangle = np.full((140, 140, 3), 220, np.uint8)
    cv2.circle(bangle, (70, 70), 45, (20, 120, 180), 3)
    disc = np.zeros((140, 140), np.uint8)
    cv2.circle(disc, (70, 70), 47, 255, -1)
    original_disc = disc.copy()
    j = Jewel(1, (23, 23, 118, 118), np.array([70., 70.]), (0, 0, 140, 140), bangle.copy(), disc)
    assert j.verification_info['valid'] and j.verification_mask[70, 70] == 0
    assert np.count_nonzero(j.verification_mask) < .4*np.count_nonzero(disc)
    np.testing.assert_array_equal(disc, original_disc)
    comparison = PickupEngine(lambda event: None)
    assert comparison.compare(j, bangle)['present']
    assert comparison.compare(j, np.full_like(bangle, 220))['removed']
    dark_bed = np.full_like(bangle, 30)
    cv2.circle(dark_bed, (70, 70), 45, (80, 160, 210), 3)
    bright_ring = Jewel(3, j.box, j.position.copy(), j.crop_box, dark_bed, original_disc)
    assert bright_ring.verification_info['valid'] and comparison.compare(bright_ring, np.full_like(bangle, 30))['removed']

    # Candidate recording retains source timing and pre-roll, including uncertainty.
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        clips = CandidateClips(Path(directory), 5.)
        clips.update(frame, [], 0.)
        e, events = engine()
        j = e.jewels[0]
        j.state, j.contact_at = 'PICKUP_CANDIDATE', .2
        clips.update(frame, [j], .2)
        j.state = 'UNCERTAIN'
        clips.update(frame, [j], .4)
        j.state = 'PICKUP_CONFIRMED'
        clips.update(frame, [j], .6)
        clips.close()
        manifest = [json.loads(line) for line in (Path(directory)/'candidate_frames.jsonl').read_text().splitlines()]
        assert manifest[0]['state'] == 'PRE_ROLL' and manifest[0]['source_time'] == 0.
        assert any(row['state']=='UNCERTAIN' for row in manifest)
    # Progress cannot retain an obsolete UNCERTAIN label while empty-home votes
    # accumulate. Occlusion resets votes and shows the actual blocker.
    e = HomePositionEngine(lambda event: None)
    e.set_references(frame, [(1, gold)], [], roi, 0.)
    e.scene_stable = lambda gray: True
    e.update(frame, [], [gold], [], roi, 1.)
    for t, expected in ((1.2, 1), (1.45, 2)):
        e.update(empty, [], [], [], roi, t)
        j = e.jewels[0]
        assert j.state == 'VERIFYING' and j.votes == expected
        assert f'{expected}/3' in j.reason and j.verified_at is None
    e.update(empty, [], [gold], [], roi, 1.6)
    assert j.state == 'UNCERTAIN' and j.votes == 0 and 'Hand covers' in j.reason
    # Visible motion can turn into hidden gold between observations without
    # subtracting None from the previous COG or inventing a new jewel position.
    moved_box = (180, 95, 230, 135)
    moved_mask = np.zeros(shape[:2], np.uint8)
    moved_mask[95:135, 180:230] = 255
    moved_gold = SimpleNamespace(box=moved_box, mask=moved_mask)
    for t in (1.8, 2.):
        e.update(empty, [moved_gold], [], [], roi, t, observation(t, live=moved_box))
    assert j.state == 'PICKUP_DETECTED'
    e.update(empty, [], [], [], roi, 2.2)
    assert j.verified_at is None and j.current_cog is None
    e.update(empty, [], [], [], roi, 2.4)
    assert j.state == 'PICKUP_CONFIRMED'

    # VERIFYING keeps recording. Video scaling never changes source dimensions
    # used for verification or the clean trigger-start photograph.
    with tempfile.TemporaryDirectory() as directory:
        large = np.full((1080, 1920, 3), 215, np.uint8)
        clips = CandidateClips(Path(directory), 5.)
        j.state, j.contact_at = 'VERIFYING', 0.
        clips.update(large, [j], 0.)
        assert j.id in clips.writers
        clips.close()
        movie = next(Path(directory).glob('*.avi'))
        cap = cv2.VideoCapture(str(movie))
        ok, recorded = cap.read()
        cap.release()
        assert ok and recorded.shape[:2] == (540, 960)
        assert cv2.imread(str(movie.with_name(movie.stem+'_start.jpg'))).shape[:2] == large.shape[:2]
        assert large.shape[:2] == (1080, 1920)
    print('PASS: no-hand-model assignment, IN_HAND/UNDER_HAND candidates, dropout rejection, '
          'stationary cancellation, colour occlusion rejection, removal verification, early gold movement, '
          'source gaps, J1-over-J2, camera/lighting/bag guards, Otsu bangle masks, candidate recording, '
          'original-ID adoption and verification that never changes live tracking.')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', help='Camera index or video path; defaults to saved source')
    parser.add_argument('--fps', type=float, default=5., help='Maximum inference sampling rate (default 5; camera drops old frames)')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs'/'pickup_trigger_test')
    parser.add_argument('--trigger-mode', choices=('previous', 'home', 'cog'), default='previous')
    parser.add_argument('--cog-threshold', type=float, default=20., help='Fingertip-mode jewel displacement threshold in capture pixels (default 20)')
    parser.add_argument('--autostart', action='store_true')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--smoke-ui', action='store_true')
    parser.add_argument('--check-models', action='store_true')
    parser.add_argument('--list-cameras', action='store_true', help='List Linux capture devices without opening a stream')
    args = parser.parse_args()
    if not math.isfinite(args.cog_threshold) or args.cog_threshold <= 0:
        parser.error('--cog-threshold must be a positive finite pixel distance')
    if not 1 <= args.fps <= 10:
        parser.error('--fps must be between 1 and 10')
    if args.self_test:
        self_test()
        return 0
    if args.list_cameras:
        devices = capture_devices()
        for index, name in devices:
            print(f'Camera {index}: {name} (/dev/video{index})')
        if not devices:
            print('No readable Linux video-capture devices found.')
        return 0
    original, QtCore, QtGui, QtWidgets = dependencies()
    if args.check_models:
        models = Models(original, load_bag=args.trigger_mode != 'cog')
        frame = np.full((540, 960, 3), 220, np.uint8)
        roi = np.array([[0, 0], [959, 0], [959, 539], [0, 539]], np.int32)
        models.process(frame, roi, 0.)
        if args.trigger_mode != 'cog':
            models.enable_bag().detect(frame)
        if args.trigger_mode == 'home':
            hands = models.enable_hands().detect(frame)
            assert all(hand.mask.shape == frame.shape[:2] for hand in hands)
        assert models.tracker.identity_slic and models.tracker.id_model is not None
        models.close_tracking()
        print('PASS: original Worker + SLIC identity and gold INT8 execute on CPU; trigger mode: '+args.trigger_mode)
        if args.trigger_mode == 'cog':
            hands = models.enable_keypoints().detect(frame)
            assert all(hand.points.shape == (21, 2) for hand in hands)
            assert models.hand is None and models.bag is None
            models.close_keypoints()
            print('PASS: landmark model executes on full frame; no hand segmentation or packet model loaded.')
        if args.trigger_mode == 'home':
            print('PASS: home-mode hand segmentation model executes on CPU at capture coordinates.')
        return 0
    app = QtWidgets.QApplication(sys.argv[:1])
    app.setStyle('Fusion')
    Window = window_class(original, QtCore, QtGui, QtWidgets)
    if args.smoke_ui:
        args.autostart = False
    window = Window(args); window.show()
    if args.smoke_ui:
        app.processEvents()
        window.mode_buttons['cog'].click()
        assert window.selected_mode == 'cog' and window.mode_buttons['cog'].isChecked()
        window.mode_buttons['home'].click()
        assert window.selected_mode == 'home' and window.mode_buttons['home'].isChecked()
        window.mode_buttons['previous'].click()
        assert window.selected_mode == 'previous' and window.mode_buttons['previous'].isChecked()
        assert not window.mode_buttons['home'].isChecked()
        worker = SimpleNamespace(isRunning=lambda: True, requested_mode='previous', reset_event=threading.Event())
        window.worker = worker
        window.mode_buttons['home'].click()
        assert worker.requested_mode == 'home' and worker.reset_event.is_set()
        window.worker = None
        window.close()
        print('PASS: standalone pickup UI and exclusive trigger-mode buttons work without starting capture.')
        return 0
    return app.exec_()


if __name__ == '__main__':
    raise SystemExit(main())

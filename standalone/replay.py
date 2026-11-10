"""Deterministic model replay using video timestamps, without camera frame drops."""
import json
from pathlib import Path

import cv2
import numpy as np

from .app import PacketWorker, ROOT
from .flow import PackingFlow, INSIDE
from .models import PacketModels
from .packet_region import PacketRegion
from .settings import tracker_camera, tracker_roi
from .tracking import TrackingBridge


def replay(path, report, step=6):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("Video has no usable frame rate")
    width, height = int(capture.get(3)), int(capture.get(4))
    reference = tracker_camera(0).get('resolution', '2560x1440')
    rw, rh = map(int, reference.split('x'))
    roi = np.rint(np.array(tracker_roi()) * [width/rw, height/rh]).astype(int)
    if len(roi) < 3:
        raise ValueError("Save a bench ROI in Jewel Tracker before replay")
    tracker = TrackingBridge(roi.tolist())
    worker = PacketWorker(str(path), roi.tolist())
    worker.tracker = tracker
    models, region = PacketModels(ROOT / 'model onnx'), PacketRegion()
    roi_mask = np.zeros((height, width), np.uint8)
    cv2.fillPoly(roi_mask, [roi], 255)
    events, frames, previous = [], 0, {}
    try:
        for index in range(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                break
            t = 1000. + index / fps
            tracker.inactive_ids = {j.id for j in worker.flow.jewels.values() if j.started is not None} if worker.flow else set()
            tracker.taken_ids = set(tracker.inactive_ids)
            tracker.transfer_active = bool(worker.flow and any(
                j.started is not None and j.status != INSIDE for j in worker.flow.jewels.values()))
            tracker.process(frame, t)
            bag, strip, clear, _, hands = models.inspect(frame, roi_mask)
            pickup = tracker.pickup_event(t, hand_mask=hands)
            if tracker.mode == 'LOCKED' and worker.flow is None:
                worker.flow = PackingFlow([tr.id for tr in tracker.tracks])
            packet_ok, changed = region.update(bag.mask if bag is not None else None, t, clear)
            if worker.flow:
                if changed:
                    worker.flow.pause_packet()
                obs = worker._transfer_observations(frame, t, bag, hands, pickup)
                worker.flow.update(t, obs, packet_ok=packet_ok, hands_clear=clear, strip=strip,
                                   pickup=pickup, packet_stationary=region.stationary)
                current = {j.id: j.status for j in worker.flow.jewels.values()}
                for jid, state in current.items():
                    if previous.get(jid) != state:
                        event = dict(seconds=round(index/fps, 3), jewel=jid, status=state)
                        events.append(event)
                        print(json.dumps(event), flush=True)
                previous = current
            frames += 1
        flow = worker.flow
        result = dict(video=str(Path(path).resolve()), fps=fps, frame_step=step, sampled_frames=frames,
                      events=events, all_placements_inferred=bool(flow and flow.all_packed),
                      closure='not_validated_by_replay', sealed=False,
                      jewels=[dict(id=j.id, status=j.status, placement_method=j.placement_method,
                                   pickup_seconds=j.started-1000 if j.started is not None else None,
                                   packet_seconds=j.entry-1000 if j.entry is not None else None,
                                   handoff_seconds=j.clip_ended-1000 if j.clip_ended is not None else None)
                              for j in flow.jewels.values()] if flow else [])
        report = Path(report)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(result, indent=2))
        print(f"Replay report: {report}")
        return result
    finally:
        tracker.close()
        capture.release()

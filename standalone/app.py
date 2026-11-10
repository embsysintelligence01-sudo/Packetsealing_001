"""PyQt packet sealing UI and the camera/inference coordinator."""
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue, Empty
import json
import time
import uuid
import os

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import jewel_tracker_5 as original

from .evidence import Recorder, export_session, ffmpeg_path, image_save, TIMEZONE
from .flow import PackingFlow, Observation, INSIDE, PICKED, AT_PACKET
from .models import PacketModels
from .tracking import TrackingBridge, draw_gold, packet_overlap
from .packet_region import PacketRegion
from .packet_motion import PacketMotion
from .seal_context import SealContext
from .transfer import TransferFollower
from .gold_transfer import GoldTransfer, packet_position, interaction_region
from .camera import make_grabber
from .settings import tracker_camera, tracker_roi, tracker_source
from .preview import DetectionPreview
from .packet_ocr import PacketOCR

ROOT = Path(__file__).resolve().parents[1]


class PacketWorker(QtCore.QThread):
    state = QtCore.pyqtSignal(object)
    notice = QtCore.pyqtSignal(str)
    exported = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, source, roi, parent=None):
        super().__init__(parent)
        self.source, self.roi_points = source, roi
        self.commands = Queue()
        self.running = True
        self.directory = None
        self.flow = None
        self.latest_observations = {}
        self.capture = True
        self.tracker = None
        self.grabber = None
        self.last_frame = None
        self.last_t = None
        self.sealing_start = None
        self.first_all_packed = None
        self.last_status = "Session stopped before sealing verification."
        self.live_overlay = DetectionPreview()
        self.gold_overlay = DetectionPreview()
        self.gold_transfers = {}
        self.packet_gold_previous = []
        self.packet_gold_history = []
        self.transfer = TransferFollower()
        self.packet_motion = PacketMotion()
        self.seal_context = SealContext()
        self.identities_announced = False
        self.ocr = None
        self.sealed_state = None

    def _restart_assignment(self):
        self.tracker.reset_assignment()
        self.flow = None
        self.latest_observations = {}
        self.capture = True
        self.gold_transfers = {}
        self.packet_gold_previous = []
        self.packet_gold_history = []
        self.transfer = TransferFollower()
        self.packet_motion = PacketMotion()
        self.seal_context = SealContext()
        self.identities_announced = False
        self.gold_overlay.clear()
        self.live_overlay.clear()

    def command(self, name, value=None):
        self.commands.put((name, value))

    def stop(self):
        self.running = False

    def _commands(self):
        while True:
            try:
                name, value = self.commands.get_nowait()
            except Empty:
                return
            try:
                if name == "zones":
                    if self.flow is not None and self.flow.packet_started:
                        raise ValueError("A pickup has started recording. Finish this session before changing the bench ROI.")
                    roi_changed = not np.array_equal(self.roi_points, value)
                    self.roi_points = value
                    if self.tracker:
                        self.tracker.roi = original.to_poly(self.roi_points)
                        self.tracker.roi_mask_key = None
                        if roi_changed:
                            self._restart_assignment()
                else:
                    raise ValueError("Pickup, inside verification and sealing run automatically; no process command is required.")
            except ValueError as exc:
                self.notice.emit(str(exc))

    def run(self):
        recorder = None
        packet_pool = None
        exe = None
        error = None
        try:
            self.notice.emit("Loading packet models and the SLIC jewel tracker…")
            exe = ffmpeg_path()
            models = PacketModels(ROOT / "model onnx")
            self.tracker = TrackingBridge(self.roi_points)
            self.directory = ROOT / "outputs" / (datetime.now(TIMEZONE).strftime("%Y%m%d_%H%M%S_") + uuid.uuid4().hex[:6])
            work = self.directory / ".working"
            work.mkdir(parents=True)
            recorder = Recorder(work)
            camera = tracker_camera(self.source)
            self.grabber = make_grabber(self.source, camera)
            self.grabber.on_frame = recorder.offer
            self.grabber.start()
            self.notice.emit("INT8 gold model ready. All jewels in the ROI will receive IDs automatically when stable.")
            packet_region = PacketRegion()
            last_camera_ts = None
            stale_notified = False
            packet_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="packet-models")
            while self.running:
                self._commands()
                frame, t = self.grabber.latest(last_camera_ts)
                if frame is None:
                    if getattr(self.grabber, "ended", False):
                        self.last_status = ("Video source finished; packet sealed, number capture incomplete."
                                            if self.flow and self.flow.sealed else
                                            "Video source finished before sealing checks passed.")
                        break
                    if last_camera_ts and time.time() - last_camera_ts > 2.5:
                        if self.flow and not stale_notified:
                            self.flow.invalidate_packet("Camera stream interrupted. Pickup and placement results retained; evidence needs review.")
                        stale_notified = True
                        self.notice.emit("Camera stream interrupted. Packing and sealing are suspended.")
                    elif not last_camera_ts:
                        self.notice.emit(self.grabber.problem or "Connecting to camera…")
                    self.msleep(100)
                    continue
                stale_notified = False
                last_camera_ts = t
                self.last_frame, self.last_t = frame, t
                if time.time() - t > 1.5:
                    self.notice.emit("Camera image is stale. Waiting for a fresh frame.")
                    continue
                if self.flow and self.flow.sealed:
                    self._read_packet_number(frame, t, work)
                    if recorder.error:
                        raise OSError(recorder.error)
                    continue
                self.tracker.inactive_ids = ({jid for jid, j in self.flow.jewels.items() if j.started is not None}
                                             if self.flow else set())
                self.tracker.taken_ids = ({jid for jid, j in self.flow.jewels.items() if j.started is not None}
                                          if self.flow else set())
                self.tracker.transfer_active = bool(self.flow and any(
                    j.started is not None and j.status != INSIDE for j in self.flow.jewels.values()))
                self.tracker.frame_wh = (frame.shape[1], frame.shape[0])
                self.tracker.ensure_roi()
                roi_mask = np.zeros(frame.shape[:2], np.uint8)
                cv2.fillPoly(roi_mask, [self.tracker.roi], 255)
                # Gold tracking and packet inspection have separate model owners.
                # Run them on the same timestamped image concurrently, preserving
                # verification alignment without making gold wait for all three models.
                seal_bag = self.seal_context.locate(frame)
                if seal_bag is not None:
                    packet_result = packet_pool.submit(models.inspect, frame, roi_mask, context_mask=seal_bag.mask)
                else:
                    packet_result = packet_pool.submit(models.inspect, frame, roi_mask)
                detections = self.tracker.process(frame, t, capture=self.capture)
                pickup = self.tracker.pickup_event(t)
                # Preserve the original recorder trigger and pre-roll even
                # before hand-model validation assigns recording ownership.
                if pickup.active:
                    recorder.enabled.set()
                # Gold markings publish as soon as gold tracking completes. They
                # do not wait behind bag, strip, hand, or per-jewel verification.
                self._publish_gold(frame, t)
                bag, strip, clear, strip_mask, hand_mask = packet_result.result()
                if hasattr(self.tracker, "bench_anchors"):
                    pickup = self.tracker.pickup_event(t, hand_mask=hand_mask)
                if pickup.active:
                    recorder.enabled.set()
                if recorder.error:
                    raise OSError(recorder.error)
                # Finish inference against the source timestamp, never a reused result.
                if time.time() - t > 2.5:
                    if self.flow:
                        self.flow.invalidate_packet("Inference exceeded the observation age limit. Pickup and placement results retained; evidence needs review.")
                    self.notice.emit("Inference is too slow for reliable transfers at this camera resolution.")
                    self.live_overlay.clear()
                    continue
                if self.tracker.mode == "LOCKED" and self.flow is None:
                    self.capture = False
                    self.flow = PackingFlow([tr.id for tr in self.tracker.tracks])
                    for tr in self.tracker.tracks:
                        j = self.flow.jewels[tr.id]
                        j.last_seen = t
                        thumb = getattr(tr, "thumb", None)
                        jewel_photo = thumb if thumb is not None and thumb.size else original.crop_box(frame, tr.box, pad=.12)
                        image_save(work / f"J{tr.id}_initial.jpg", jewel_photo)
                    image_save(work / "initial_scene.jpg", frame)
                    self.notice.emit("Assigned " + ", ".join(f"J{tr.id}" for tr in self.tracker.tracks)
                                     + ". Preparing SLIC/ResNet identities; keep the jewels visible.")
                packet_ok, packet_changed = packet_region.update(bag.mask if bag is not None else None, t, clear)
                if packet_changed and self.flow:
                    self.flow.pause_packet()
                self.latest_observations = self._transfer_observations(frame, t, bag, hand_mask, pickup)
                before = {jid: j.status for jid, j in self.flow.jewels.items()} if self.flow else {}
                if self.flow:
                    self.flow.update(t, self.latest_observations, packet_ok=packet_ok,
                                     hands_clear=clear, strip=strip,
                                     identities_ready=self.tracker.id_model.ready, pickup=pickup,
                                     packet_stationary=packet_region.stationary,
                                     packet_context_ok=bag is not None or seal_bag is not None,
                                     strip_view_ok=bag is not None or self.seal_context.strip_view_ok)
                    if self.flow.strip_removal_requested and not self.seal_context.ready:
                        self.seal_context.arm(frame, bag, strip_mask)
                    for jid, j in self.flow.jewels.items():
                        if j.started == t:
                            image_save(work / f"J{jid}_pickup.jpg", self._draw(frame, bag, strip_mask, hand_mask))
                        if j.status != before.get(jid):
                            if j.status == PICKED:
                                image_save(work / f"J{jid}_pickup.jpg", self._draw(frame, bag, strip_mask, hand_mask))
                            elif j.status == AT_PACKET:
                                image_save(work / f"J{jid}_entry.jpg", self._draw(frame, bag, strip_mask, hand_mask))
                            elif j.status == INSIDE:
                                image_save(work / f"J{jid}_entry.jpg", self._draw(frame, bag, strip_mask, hand_mask))
                                image_save(work / f"J{jid}_after.jpg", self._draw(frame, bag, strip_mask, hand_mask))
                    if self.flow.all_packed:
                        self.first_all_packed = self.first_all_packed or t
                        self.sealing_start = self.sealing_start or t
                    if self.tracker.identity_error:
                        self.flow.message = self.tracker.identity_error
                    elif self.tracker.id_model.ready and not self.identities_announced:
                        self.identities_announced = True
                        self.notice.emit("Jewel identities ready. Pickup starts recording; the bag model supplies the full packet ROI automatically.")
                    if self.flow.sealed:
                        image_save(work / "sealed.jpg", self._draw(frame, bag, strip_mask, hand_mask))
                        for jewel in self.flow.jewels.values():
                            if jewel.clip_ended is None:
                                jewel.clip_ended = t
                        self.flow.packet_number.update(status="pending")
                        self.flow.message = "Packet sealed. Rotate the packet to show its number for OCR."
                        self.last_status = "Packet sealed; packet-number capture pending."
                        self.gold_overlay.clear()
                        self.live_overlay.clear()
                    # Refresh labels after this frame's pickup/inside decision,
                    # retaining the source timestamp rather than extending age.
                    self._publish_gold(frame, t)
                # Publish geometry only. The GUI renders it over the newest camera
                # image on its own timer, independently of model completion.
                self._publish_packet(frame, t, bag, strip_mask)
                state = dict(
                    phase=self.flow.message if self.flow else f"Auto assigning jewels: {self.tracker.arm_count} detected ({self.tracker.stable:.0%}). Keep all jewels separated in the ROI.",
                    packet=packet_ok, packet_source=self.packet_motion.source, strip=strip, clear=clear,
                    packet_stationary=packet_region.stationary, packet_hand_scope=bag is not None or seal_bag is not None, closure=self.flow.closure_status if self.flow else "pending_strip_removal",
                    closure_message="Sealing is inferred after verified strip removal; adhesive closure is not directly measured.", strip_enabled=bool(self.flow and self.flow.all_packed),
                    strip_removed=bool(self.flow and self.flow.strip_removed),
                    strip_armed=bool(self.flow and self.flow.strip_removal_requested),
                    seal_context=seal_bag is not None,
                    identities=self.tracker.id_model.ready,
                    jewels=[dict(id=tr.id, thumb=getattr(tr, "thumb", None),
                                 status=self.flow.jewels[tr.id].status, note=self.flow.jewels[tr.id].note)
                            for tr in self.tracker.tracks] if self.flow else [],
                    can_draw=not self.flow or not self.flow.packet_started,
                    directory=str(self.directory))
                self.state.emit(state)
                if self.flow and self.flow.sealed:
                    self.sealed_state = state
        except Exception as exc:
            error = str(exc)
            self.last_status = "Session interrupted: " + error
            if self.flow:
                self.flow.invalidate_packet(self.last_status)
            self.failed.emit(error)
        finally:
            if self.ocr:
                self.ocr.close()
            if packet_pool:
                packet_pool.shutdown(wait=True, cancel_futures=True)
            if self.grabber:
                self.grabber.on_frame = None
                self.grabber.stop()
                self.grabber.join(timeout=4)
            if self.tracker:
                self.tracker.close()
            if recorder:
                try:
                    recorder.close()
                except Exception as exc:
                    error = str(exc)
                    self.failed.emit(error)
                if self.flow:
                    try:
                        self.notice.emit("Exporting and checking PDF/video evidence…")
                        if self.last_frame is not None:
                            image_save(work / "final.jpg", self.last_frame)
                        result = export_session(self.directory, work, recorder, self.flow,
                                                self.last_t or time.time(), self.last_status, exe,
                                                self.sealing_start)
                        self.exported.emit(dict(directory=str(self.directory), **result))
                    except Exception as exc:
                        self.failed.emit(f"Export failed: {exc}. Source evidence retained in {work}")
                else:
                    import shutil
                    shutil.rmtree(self.directory)

    def _read_packet_number(self, frame, t, work):
        if self.ocr is None:
            self.ocr = PacketOCR(ROOT)
        result, accepted = self.ocr.update(frame, self.roi_points, t)
        self.flow.packet_number = dict(result)
        self.flow.message = result['message']
        self.last_status = ("Packet sealed; packet number confirmed." if result['status'] == 'confirmed'
                            else "Packet sealed; packet-number capture incomplete.")
        if accepted is not None:
            image_save(work / "packet_number_crop.jpg", accepted['crop'])
            image_save(work / "packet_number_scene.jpg", accepted['scene'])
            self.flow.packet_number.update(crop="packet_number_crop.jpg", scene="packet_number_scene.jpg")
            self.running = False
        self.notice.emit(result['message'])
        if self.sealed_state:
            state = dict(self.sealed_state, phase=result['message'])
            self.state.emit(state)

    def _publish_gold(self, frame, t):
        statuses = {jid: j.status for jid, j in self.flow.jewels.items()} if self.flow else None
        image = draw_gold(np.zeros_like(frame), self.tracker, t, statuses)
        # A real pickup must immediately remove its old bench box. Hold empty
        # detections only before assignment, never after an ownership change.
        self.gold_overlay.publish(image, t, hold_empty=self.tracker.mode == "ARMING")

    def _publish_packet(self, frame, t, bag, strip_mask):
        motion = bag is None and self.packet_motion.source in ('motion', 'recent')
        if motion:
            bag = self.packet_motion.current
        for key, mask, color in (("bag", bag.mask if bag is not None else None, (0, 170, 255) if motion else (60, 200, 60)),
                                 ("strip", strip_mask, (0, 170, 255))):
            image = np.zeros_like(frame)
            if mask is not None:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, color, 2)
            if key == "bag" and motion:
                cv2.putText(image, 'Packet ROI - '+self.packet_motion.source, (12, 52), cv2.FONT_HERSHEY_SIMPLEX, .65, color, 2)
            self.live_overlay.publish(image, t, key=key, hold_empty=True)

    def _transfer_observations(self, frame, t, bag, hand_mask, pickup):
        transfer_bag = self.packet_motion.update(frame, bag, t)
        observations = self._observations(frame, t, transfer_bag)
        if not self.flow:
            return observations
        raw_gold = (getattr(self.tracker, "raw_detections", [])
                    if abs(getattr(self.tracker, "raw_timestamp", t)-t) < .0001 else [])
        gold_bag = interaction_region(transfer_bag)
        anchors = getattr(self.tracker, "bench_anchors", {})
        for jid, jewel in self.flow.jewels.items():
            if jid not in self.gold_transfers and jid in anchors:
                self.gold_transfers[jid] = GoldTransfer(jid, anchors[jid][0], t)
        new_pickup = (pickup.active and not pickup.ambiguous and pickup.jewel_id in self.flow.jewels
                      and self.flow.jewels[pickup.jewel_id].started is None)
        self.packet_gold_history = [(stamp, relative, pixels) for stamp, relative, pixels in self.packet_gold_history
                                    if 0 <= t-stamp <= 2.5]
        prior_packet_gold = tuple(point for _, relative, _ in self.packet_gold_history for point in relative)
        prior_packet_pixels = tuple(point for _, _, pixels in self.packet_gold_history for point in pixels)
        for jid, path in self.gold_transfers.items():
            if self.flow.jewels[jid].started is None:
                path.blocked_packet_positions = prior_packet_gold
                path.blocked_packet_points = prior_packet_pixels
        gold_observations = {}
        priority = sorted(self.gold_transfers, key=lambda jid: (
            jid != self.flow.active_jewel_id, jid != pickup.jewel_id))
        claimed = set()
        for jid in priority:
            path = self.gold_transfers[jid]
            jewel = self.flow.jewels[jid]
            if jewel.status == INSIDE or jewel.clip_ended is not None:
                continue  # Never move an old path onto the next carried jewel.
            other_bench = [np.asarray(anchor[0]) for other, anchor in anchors.items()
                           if other != jid and self.flow.jewels[other].started is None]
            available = [det for det in raw_gold if id(det) not in claimed]
            result = path.observe(t, available, gold_bag, frame.shape, other_bench)
            if path.last_detection is not None:
                claimed.add(id(path.last_detection))
            if (result is None and new_pickup and jid == pickup.jewel_id
                    and path.last_interaction_t is not None and 0 <= t-path.last_interaction_t <= .8):
                result = path.last_interaction  # Transfer seen just before the pickup trigger completed.
            if result is not None:
                gold_observations[jid] = result
        packet_gold = [(position, tuple(det.c)) for det in raw_gold
                       if (position := packet_position(det.c, gold_bag, allow_box=True)) is not None]
        self.packet_gold_previous = [relative for relative, _ in packet_gold]
        self.packet_gold_history.append((t, self.packet_gold_previous, [pixels for _, pixels in packet_gold]))
        # Process the outgoing courier before starting the next one. Recording
        # ownership and transfer outcome are independent, even on the same frame.
        outgoing = self.flow.jewels.get(self.transfer.owner)
        if outgoing is not None and outgoing.status != INSIDE:
            interaction = self.transfer.observe(t, hand_mask, transfer_bag, frame.shape)
            if interaction is not None:
                observations[self.transfer.owner] = interaction
        if pickup.active and not pickup.ambiguous and pickup.jewel_id in self.flow.jewels:
            if self.flow.jewels[pickup.jewel_id].started is None:
                tr = next(tr for tr in self.tracker.tracks if tr.id == pickup.jewel_id)
                anchor = getattr(self.tracker, "bench_anchors", {}).get(tr.id, (tr.pos, None))[0]
                seed = getattr(self.tracker, "pickup_seeds", {}).get(tr.id, anchor)
                self.transfer.start(tr.id, seed, t)
                interaction = self.transfer.observe(t, hand_mask, transfer_bag, frame.shape)
                if interaction is not None:
                    observations[self.transfer.owner] = interaction
        # OR: a hand interaction wins when observed; otherwise use the moving
        # gold path owned by this pickup. Bench/old packet gold cannot claim it.
        for jid, result in gold_observations.items():
            jewel = self.flow.jewels[jid]
            if jewel.started is not None or (new_pickup and jid == pickup.jewel_id):
                existing = observations.get(jid)
                if existing is None or not (existing.near_packet or existing.packet_overlap >= .1):
                    observations[jid] = result
        return observations

    def _observations(self, frame, t, bag):
        observations = {}
        if not self.flow:
            return observations
        expanded = None
        if bag is not None:
            x1, y1, x2, y2 = bag.box
            margin = max(12, round(max(x2 - x1, y2 - y1) * .04))
            expanded = cv2.dilate(bag.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1)))
        for tr in self.tracker.tracks:
            if self.flow.jewels[tr.id].status == INSIDE:
                continue  # Completed placement is not queried again.
            if self.flow.jewels[tr.id].started is not None and getattr(self.tracker, "bench_anchors", None):
                continue
            det = self.tracker.returners.get(tr.id)
            if det is not None:
                point, box = det.c, det.box
                mask, origin = det.mask, det.origin
            elif abs(tr.det_seen - t) < .0001:
                point, box = tr.pos, tr.det_box
                mask, origin = tr.det_mask, tr.det_origin
            else:
                continue
            identity_ok = self.tracker.continuous_identity.get(tr.id, False)
            if self.tracker.id_model.ready and self.flow.jewels[tr.id].started is None:
                verdict, jid, _, _ = self.tracker.id_model.decide(original.crop_box(frame, box))
                if verdict != "ambiguous":
                    identity_ok = verdict == "jewel" and jid == tr.id
            observations[tr.id] = Observation(tuple(map(float, point)),
                packet_overlap=packet_overlap(mask, origin, bag.mask if bag is not None else None), identity_ok=identity_ok,
                near_packet=packet_overlap(mask, origin, expanded) >= .1,
                stationary_tolerance=max(3., frame.shape[1] * .005))
        return observations

    def _draw(self, frame, bag, strip_mask, hand_mask, *, include_gold=True):
        out = frame.copy()
        for mask, color in ((bag.mask if bag is not None else None, (60, 200, 60)),
                            (strip_mask, (0, 170, 255))):
            if mask is not None:
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(out, contours, -1, color, 2)
        if self.tracker and self.tracker.roi is not None:
            cv2.polylines(out, [self.tracker.roi], True, (255, 190, 0), 2)
        if self.tracker and include_gold:
            draw_gold(out, self.tracker, self.last_t or time.time(),
                      {jid: j.status for jid, j in self.flow.jewels.items()} if self.flow else None)
        cv2.putText(out, datetime.fromtimestamp(self.last_t or time.time(), TIMEZONE).strftime("%Y-%m-%d %H:%M:%S IST"),
                    (12, out.shape[0] - 16), cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 2)
        return out


class MainWindow(QtWidgets.QWidget):
    def __init__(self, *, auto_connect=False):
        super().__init__()
        self.setWindowTitle("Packet Sealing — Jewel Guard")
        self.worker = None
        self.restart_pending = False
        self.regions_locked = False
        self.roi = []
        self.draft, self.drawing = [], None
        self.last_image = None
        self.output_directory = ROOT / "outputs"
        self.settings_path = ROOT / "packet_ui_settings.json"
        settings = {}
        if self.settings_path.is_file():
            try:
                settings = json.loads(self.settings_path.read_text())
            except (ValueError, OSError):
                pass
        # Older packet settings used a separate ROI at a different resolution.
        # Start from the tracker's actual zones unless explicitly redrawn here.
        self.roi_override = bool(settings.get("roi_override", False))
        self.roi = settings.get("roi", []) if self.roi_override else tracker_roi()
        self.source = QtWidgets.QLineEdit(settings.get("source") or tracker_source())
        self.source.setPlaceholderText("USB camera number, RTSP URL, or video path")
        self.connect_button = QtWidgets.QPushButton("Connect")
        self.connect_button.clicked.connect(self.connect_camera)
        self.finish_button = QtWidgets.QPushButton("Finish / Save")
        self.finish_button.clicked.connect(self.finish_session)
        self.finish_button.setEnabled(False)
        self.reset_button = QtWidgets.QPushButton("Reset / Start Over")
        self.reset_button.setToolTip("Save the current session, then restart jewel assignment and all packet checks from the beginning.")
        self.reset_button.clicked.connect(self.reset_process)
        top = QtWidgets.QHBoxLayout()
        top.addWidget(QtWidgets.QLabel("Camera / source:"))
        top.addWidget(self.source, 1)
        top.addWidget(self.connect_button)
        top.addWidget(self.finish_button)
        top.addWidget(self.reset_button)
        self.video = original.VideoLabel()
        self.video.setMinimumSize(640, 360)
        self.video.clicked.connect(self.click_video)
        self.roi_button = QtWidgets.QPushButton("Draw bench ROI")
        self.shared_roi_button = QtWidgets.QPushButton("Use Jewel Tracker ROI")
        self.shared_roi_button.clicked.connect(self.use_tracker_roi)
        self.roi_button.clicked.connect(lambda: self.begin_draw("roi"))
        self.folder_button = QtWidgets.QPushButton("Open saved PDFs / videos")
        self.folder_button.clicked.connect(lambda: QtGui.QDesktopServices.openUrl(QtCore.QUrl.fromLocalFile(str(self.output_directory))))
        self.phase = QtWidgets.QLabel("Camera starts automatically. Lay all jewels separately in the ROI; J1, J2… assign when stable.")
        self.phase.setWordWrap(True)
        self.phase.setStyleSheet("font-weight:bold;font-size:16px;color:#075985;padding:8px")
        self.checks = QtWidgets.QLabel("Gold: INT8 | SLIC identities | Jewel Tracker ROI loaded" if self.roi else "Gold: INT8 | SLIC identities | Draw bench ROI")
        self.checks.setWordWrap(True)
        self.instruction = QtWidgets.QLabel("Pickup fixes the jewel ID and starts recording. Hand OR moving gold from the current pickup reaching the packet confirms transfer. No identity check inside, disappearance or clear-hands wait. Packet movement and brief detection gaps retain progress. The next pickup switches recording without failing the previous transfer. After all placements: put the packet on the test bed with its strip visible. Once verified, remove the strip. The strip area must be visible; the whole bag is not required. Confirmed strip removal completes sealing and stays retained. Then rotate the packet to show its number inside the bench ROI. Three matching OCR reads complete the session and save the PDF/videos.")
        self.instruction.setWordWrap(True)
        self.notice = QtWidgets.QLabel("")
        self.notice.setWordWrap(True)
        self.notice.setStyleSheet("color:#9a3412;padding:6px")
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Jewel", "Initial photo", "Packing", "Evidence"])
        self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        self.table.setWordWrap(True)
        self.table.verticalHeader().hide()
        side = QtWidgets.QVBoxLayout()
        for widget in (self.shared_roi_button, self.roi_button, self.phase, self.checks,
                       self.instruction, self.table, self.notice, self.folder_button):
            side.addWidget(widget)
        panel = QtWidgets.QWidget()
        panel.setLayout(side)
        panel.setMinimumWidth(460)
        layout = QtWidgets.QHBoxLayout()
        layout.addWidget(self.video, 3)
        layout.addWidget(panel, 2)
        outer = QtWidgets.QVBoxLayout(self)
        outer.addLayout(top)
        outer.addLayout(layout, 1)
        self.resize(1400, 850)
        self.preview_source_size = None
        self.preview_timestamp = None
        self.preview_timer = QtCore.QTimer(self)
        self.preview_timer.timeout.connect(self.refresh_preview)
        self.preview_timer.start(original.PREVIEW_INTERVAL_MS)
        if auto_connect:
            QtCore.QTimer.singleShot(0, self.connect_camera)

    def use_tracker_roi(self):
        if self.regions_locked:
            return
        roi = tracker_roi()
        if not roi:
            self.notice.setText("Jewel Tracker has no saved bench ROI.")
            return
        self.roi, self.roi_override = roi, False
        try:
            saved = json.loads(self.settings_path.read_text()) if self.settings_path.is_file() else {}
            saved.update(roi=self.roi, roi_override=False)
            self.settings_path.write_text(json.dumps(saved, indent=2))
        except (OSError, ValueError):
            pass
        self.send("zones", self.roi)
        self.notice.setText("Loaded the bench ROI from Jewel Tracker zones.json.")

    def refresh_preview(self):
        worker = self.worker
        grabber = worker.grabber if worker else None
        if grabber is None:
            return
        frame, timestamp = grabber.latest(self.preview_timestamp)
        if frame is None:
            return
        self.preview_timestamp = timestamp
        source_size = (frame.shape[1], frame.shape[0])
        worker.live_overlay.render(frame, timestamp)
        if hasattr(worker, "gold_overlay"):
            worker.gold_overlay.render(frame, timestamp)
        if self.roi:
            cv2.polylines(frame, [np.asarray(self.roi, np.int32)], True, (255, 190, 0), 2)
        if frame.shape[1] > 1280:
            frame = cv2.resize(frame, (1280, round(frame.shape[0] * 1280 / frame.shape[1])), interpolation=cv2.INTER_AREA)
        self.preview_source_size = source_size
        self.show_frame(original.to_qimage(frame).copy())

    def connect_camera(self):
        if self.worker is not None and self.worker.isRunning():
            return
        value = self.source.text().strip()
        if not value:
            self.notice.setText("Enter a camera number, RTSP URL, or video path.")
            return
        source = int(value) if value.isdigit() else value
        if not self.roi_override:
            self.roi = tracker_roi()
        self.preview_timestamp = None
        self.regions_locked = False
        previous = self.worker
        self.worker = PacketWorker(source, self.roi, self)
        if previous is not None and hasattr(previous, "deleteLater"):
            previous.deleteLater()
        self.worker.state.connect(self.show_state)
        self.worker.notice.connect(self.notice.setText)
        self.worker.failed.connect(self.show_error)
        self.worker.exported.connect(self.show_export)
        self.worker.finished.connect(self.worker_finished)
        self.worker.start()
        self.connect_button.setEnabled(False)
        self.source.setEnabled(False)
        self.finish_button.setEnabled(True)
        self.roi_button.setEnabled(True)
        self.shared_roi_button.setEnabled(True)
        self.table.setRowCount(0)
        self.notice.clear()

    def reset_process(self):
        if self.restart_pending:
            return
        self.restart_pending = True
        self.reset_button.setEnabled(False)
        self.drawing, self.draft = None, []
        if self.worker is not None and self.worker.isRunning():
            if self.worker.flow is None or not self.worker.flow.sealed:
                self.worker.last_status = "Process reset by operator; previous session saved incomplete."
            self.worker.stop()
            self.finish_button.setEnabled(False)
            self.roi_button.setEnabled(False)
            self.shared_roi_button.setEnabled(False)
            self.phase.setText("Resetting process — saving current session…")
            self.notice.setText("The new process starts automatically after evidence export finishes. Lay all jewels separately in the ROI.")
        else:
            self._restart_after_save()

    def _restart_after_save(self):
        if not self.restart_pending:
            return
        self.restart_pending = False
        previous, self.worker = self.worker, None
        if previous is not None and hasattr(previous, "deleteLater"):
            previous.deleteLater()
        self.regions_locked = False
        self.drawing, self.draft = None, []
        self.table.clearContents()
        self.table.setRowCount(0)
        self.preview_timestamp = self.preview_source_size = self.last_image = None
        self.video.clear()
        self.video.frame_size = None
        self.video.setText("Starting new process…")
        self.phase.setText("Starting from the beginning. Lay all jewels separately for fresh J1, J2… assignment.")
        self.checks.setText("Gold: INT8 | Packet: waiting | Strip: waiting | Sealing: waiting")
        self.checks.setToolTip("")
        self.reset_button.setEnabled(True)
        self.connect_camera()

    def send(self, name, value=None):
        if self.worker and self.worker.isRunning():
            self.worker.command(name, value)
        else:
            self.notice.setText("Connect the camera first.")

    def begin_draw(self, kind):
        if self.regions_locked:
            return
        self.drawing, self.draft = kind, []
        self.notice.setText(f"Draw {kind}: left-click corners; right-click to finish (at least 3 points).")

    def click_video(self, x, y, left):
        if not self.drawing or self.regions_locked:
            return
        if left:
            self.draft.append((x, y))
        elif len(self.draft) >= 3:
            polygon = np.asarray(self.draft, np.int32)
            if cv2.contourArea(polygon) < 100:
                self.notice.setText("Draw a larger polygon with corners in order.")
                return
            self.roi = list(self.draft)
            self.roi_override = True
            self.drawing, self.draft = None, []
            # Keep credentials out of the settings file.
            value = self.source.text().strip()
            self.settings_path.write_text(json.dumps(dict(roi=self.roi,
                roi_override=self.roi_override, source=value if value.isdigit() else None), indent=2))
            self.send("zones", self.roi)
            self.notice.setText("Bench ROI saved. The bag model defines the packet ROI automatically.")
        if self.last_image is not None:
            self.show_frame(self.last_image)

    def show_frame(self, image):
        self.last_image = image
        image = image.copy()
        painter = QtGui.QPainter(image)
        if self.preview_source_size:
            painter.scale(image.width() / self.preview_source_size[0], image.height() / self.preview_source_size[1])
        painter.setPen(QtGui.QPen(QtGui.QColor("yellow"), 3))
        if self.draft:
            points = [QtCore.QPoint(x, y) for x, y in self.draft]
            painter.drawPolyline(QtGui.QPolygon(points))
            for point in points:
                painter.drawEllipse(point, 5, 5)
        painter.end()
        self.video.set_frame(image, self.preview_source_size)

    def show_state(self, state):
        self.phase.setText(state["phase"])
        strip = ("removed (retained)" if state.get("strip_removed") else
                 state["strip"] if state["strip_enabled"] else "waiting for jewel transfers")
        stationary = "test-bed baseline retained" if state.get("strip_armed") and not state["packet_stationary"] else str(state["packet_stationary"])
        hand_status = ('clear' if state['clear'] else 'present') if state.get('packet_hand_scope', True) else 'unverified'
        self.checks.setText(f"Gold: INT8 | Packet: {'verified' if state['packet'] else 'motion tracking' if state.get('packet_source') == 'motion' else 'saved body context' if state.get('seal_context') else 'visibility pending'} | Strip: {strip} | Packet hands: {hand_status} | Stationary: {stationary} | Sealing: {'sealed (inferred)' if state['closure'] == 'inferred_from_strip_removal' else 'waiting for strip removal'} | SLIC IDs: {'ready' if state['identities'] else 'preparing'}")
        self.checks.setToolTip(state["closure_message"])
        self.roi_button.setEnabled(state["can_draw"])
        self.shared_roi_button.setEnabled(state["can_draw"])
        self.regions_locked = not state["can_draw"]
        if self.regions_locked:
            self.drawing, self.draft = None, []
        self.output_directory = Path(state["directory"])
        selected = self.table.currentRow()
        self.table.setRowCount(len(state["jewels"]))
        for row, item in enumerate(state["jewels"]):
            self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(f"J{item['id']}"))
            label = QtWidgets.QLabel()
            if item["thumb"] is not None:
                label.setPixmap(QtGui.QPixmap.fromImage(original.to_qimage(item["thumb"])).scaled(90, 65, QtCore.Qt.KeepAspectRatio))
            self.table.setCellWidget(row, 1, label)
            for column, key in ((2, "status"), (3, "note")):
                cell = QtWidgets.QTableWidgetItem(item[key])
                if item["status"] == INSIDE:
                    cell.setBackground(QtGui.QColor("#dcfce7"))
                elif item["status"] == "Uncertain":
                    cell.setBackground(QtGui.QColor("#ffedd5"))
                self.table.setItem(row, column, cell)
            self.table.setRowHeight(row, 85)
        if 0 <= selected < self.table.rowCount():
            self.table.selectRow(selected)

    def finish_session(self):
        if self.worker:
            self.worker.stop()
            self.finish_button.setEnabled(False)
            self.roi_button.setEnabled(False)
            self.shared_roi_button.setEnabled(False)
            self.notice.setText("Finishing recording and exporting evidence. Please keep this window open.")

    def show_error(self, message):
        self.notice.setText(message)
        self.phase.setText("Attention required")

    def show_export(self, result):
        self.output_directory = Path(result["directory"])
        number = result.get('packet_number', {})
        phase = (f"Sealed — packet number: {number['text']}" if number.get('status') == 'confirmed' else
                 "Sealed — packet number not confirmed") if result['sealed'] else "Session saved — incomplete / review required"
        self.phase.setText(phase)
        size = "All evidence groups met the 2 MB target." if result["all_budgets_met"] else "Some evidence exceeds 2 MB; source recording retained."
        self.notice.setText(f"{size}\nSaved to: {self.output_directory}")
        final = {j["id"]: j for j in result["jewels"]}
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item is None:
                continue
            jewel = final.get(int(item.text()[1:]))
            if jewel:
                for column, key in ((2, "status"), (3, "note")):
                    cell = QtWidgets.QTableWidgetItem(jewel[key])
                    cell.setBackground(QtGui.QColor("#dcfce7" if jewel["status"] == INSIDE else "#ffedd5"))
                    self.table.setItem(row, column, cell)

    def worker_finished(self):
        self.connect_button.setEnabled(True)
        self.connect_button.setText("New session / Connect")
        self.source.setEnabled(True)
        self.finish_button.setEnabled(False)
        if self.restart_pending:
            QtCore.QTimer.singleShot(0, self._restart_after_save)
            return
        self.reset_button.setEnabled(True)
        if self.worker and self.worker.flow is None:
            self.phase.setText("Session ended. No jewels were captured.")

    def closeEvent(self, event):
        self.restart_pending = False  # Closing cancels a queued automatic restart.
        if self.worker and self.worker.isRunning():
            self.finish_session()
            self.notice.setText("Saving evidence. Close the window after export finishes.")
            event.ignore()
        else:
            self.preview_timer.stop()
            event.accept()

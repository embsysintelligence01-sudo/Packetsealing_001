"""Packet sealing workflow service."""

from __future__ import annotations

import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from HandRemover.handremover import get_hand_model, infer_hand_mask
from hailo_model_runner import (
    DEFAULT_HAILO_BATCH_SIZE,
    DEFAULT_HAILO_INFERENCE_TIMEOUT_MS,
    HailoRuntime,
)
from jewel_tracka_rpi import striping_process_hef as support

from services.camera_service import CameraService, save_jpeg
from services.roi_service import RoiService


HAND_CHECK_FRAMES = 5
HAND_REQUIRED_FRAMES = 2
HAND_MIN_AREA_FRACTION = 0.02
HAND_MIN_HEIGHT_FRACTION = 0.20
PACKET_RECHECK_MAX_RESULTS = 60


class _HandWorker(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self._input: queue.Queue[np.ndarray | None] = queue.Queue(maxsize=1)
        self._output: queue.Queue[tuple[Any, str]] = queue.Queue(maxsize=1)

    def submit(self, frame: np.ndarray) -> None:
        try:
            self._input.get_nowait()
        except queue.Empty:
            pass
        self._input.put(frame.copy())

    def result(self):
        try:
            return self._output.get_nowait()
        except queue.Empty:
            return None

    def stop(self) -> None:
        try:
            self._input.get_nowait()
        except queue.Empty:
            pass
        self._input.put(None)

    def run(self) -> None:
        while True:
            frame = self._input.get()
            if frame is None:
                return
            error = ""
            try:
                mask = infer_hand_mask(frame, confidence=0.45, mask_threshold=0.65)
                result = (support.get_centroid(mask), mask)
            except Exception as exc:
                result = None
                error = str(exc)
            try:
                self._output.get_nowait()
            except queue.Empty:
                pass
            self._output.put((result, error))


class PacketService:
    def __init__(
        self,
        base_dir: Path,
        camera: CameraService,
        roi: RoiService,
        speak: Callable[[str], None],
    ) -> None:
        self.base_dir = Path(base_dir)
        self.camera = camera
        self.roi = roi
        self.speak = speak
        self._lock = threading.RLock()
        self._models: dict[str, Any] = {}
        self._worker: Any = None
        self._worker_model: Any = None
        self._worker_kind = ""
        self._record_thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._pledge_dir: Path | None = None
        self._video_path: Path | None = None
        self._evidence_path: Path | None = None
        self._final_image_path: Path | None = None
        self._display_frame: np.ndarray | None = None
        self._status = "idle"
        self._reason = ""
        self._error = ""
        self._active = False
        self._completed = False
        self._sealed: bool | None = None
        self._cover_mask: np.ndarray | None = None
        self._cover_confidence: float | None = None
        self._cover_confirmations = 0
        self._packet_recheck_frames = 0
        self._packet_recheck_candidate: np.ndarray | None = None
        self._strip_mask: np.ndarray | None = None
        self._strip_confidence: float | None = None
        self._strip_search_misses = 0
        self._live_strip_misses = 0
        self._strip_hits = 0
        self._strip_misses = 0
        self._hand_frames = 0
        self._hand_hits = 0
        self._hand_mask: np.ndarray | None = None

    def preload(self, runtime: HailoRuntime) -> dict[str, Any]:
        with self._lock:
            if self._models:
                return self.status()
            bag = runtime.create_model(
                str(self.base_dir / "models" / "bagnewmodel.hef"),
                "PacketStripBag",
                timeout_ms=DEFAULT_HAILO_INFERENCE_TIMEOUT_MS,
                batch_size=DEFAULT_HAILO_BATCH_SIZE,
            )
            strip = runtime.create_model(
                str(self.base_dir / "models" / "packetstrip.hef"),
                "PacketStrip",
                timeout_ms=DEFAULT_HAILO_INFERENCE_TIMEOUT_MS,
                batch_size=DEFAULT_HAILO_BATCH_SIZE,
            )
            if bag is None or strip is None:
                raise RuntimeError(runtime.last_model_error or "Packet HEFs could not be loaded.")
            hand = get_hand_model(runtime)
            for model in (bag, strip, hand):
                model.validate_runtime_contract(probe_runs=1)
            self._models = {"bag": bag, "strip": strip, "hand": hand}
            return self.status()

    def start(self, pledge_dir: Path) -> dict[str, Any]:
        with self._lock:
            if self._active:
                raise RuntimeError("Packet closing is already active.")
            if not self._models:
                raise RuntimeError("Packet models are not ready.")
            self._pledge_dir = Path(pledge_dir)
            packet_dir = self._pledge_dir / "packet"
            packet_dir.mkdir(parents=True, exist_ok=True)
            self._video_path = packet_dir / "closing_video.mp4"
            self._evidence_path = None
            self._final_image_path = None
            self._display_frame = None
            self._reset_cycle()
            self._active = True
            self._status = "tracking"
            self._reason = "Place the packet flat on the test bed with its strip visible."
            self._stop.clear()
            self._start_model("cover")
            self._record_thread = threading.Thread(
                target=self._record_loop,
                name="packet-closing",
                daemon=True,
            )
            self._record_thread.start()
            self.speak("Packet closing started. Place the packet and strip on the test bed.")
            return self.snapshot()

    def request_hand_clear(self) -> dict[str, Any]:
        with self._lock:
            if self._status != "strip_detected":
                raise ValueError("Wait until the packet strip is detected.")
            self._hand_frames = 0
            self._hand_hits = 0
            self._hand_mask = None
            self._status = "hand_check"
            self._reason = "Remove hands while the system checks the test bed."
            self._start_model("hand")
            return self.snapshot()

    def retry(self) -> dict[str, Any]:
        with self._lock:
            if not self._active:
                raise ValueError("Packet closing is not active.")
            self._reset_cycle(keep_active=True)
            self._status = "tracking"
            self._reason = "Place the packet flat on the test bed with its strip visible."
            self._start_model("cover")
            return self.snapshot()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if not self._active:
                return self.snapshot()
            self._stop.set()
            self._reason = "Saving packet photos."
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "ready": bool(self._models),
                "active": self._active,
                "completed": self._completed,
                "sealed": self._sealed,
                "status": self._status,
                "reason": self._reason,
                "error": self._error,
                "hand_clear_enabled": self._status == "strip_detected",
                "video": str(self._video_path) if self._completed and self._video_path else None,
                "evidence": str(self._evidence_path) if self._evidence_path else None,
                "final_image": str(self._final_image_path) if self._final_image_path else None,
                "bag_confidence": self._cover_confidence,
                "strip_confidence": self._strip_confidence,
            }

    def status(self) -> dict[str, Any]:
        return {"ready": bool(self._models), "error": self._error}

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        with self._lock:
            output = frame.copy()
            for mask, color, label in (
                (self._cover_mask, (50, 220, 50), "PACKET"),
                (self._strip_mask, (0, 140, 255), "STRIP"),
                (self._hand_mask, (220, 70, 220), "HAND - WAIT"),
            ):
                if mask is None or mask.shape[:2] != output.shape[:2]:
                    continue
                selected = mask > 0
                if not np.any(selected):
                    continue
                output[selected] = (
                    output[selected].astype(np.float32) * 0.68
                    + np.asarray(color, dtype=np.float32) * 0.32
                ).astype(np.uint8)
                contour = support.get_largest_contour(mask)
                if contour is not None:
                    cv2.drawContours(output, [contour], -1, color, 3)
                    x, y, _width, _height = cv2.boundingRect(contour)
                    cv2.putText(
                        output,
                        label,
                        (x, max(28, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        color,
                        2,
                    )
            return output

    def display_frame(self) -> np.ndarray | None:
        with self._lock:
            return None if self._display_frame is None else self._display_frame.copy()

    def shutdown(self) -> None:
        self._stop.set()
        thread = self._record_thread
        if thread and thread.is_alive():
            thread.join(timeout=3.0)
        with self._lock:
            self._stop_worker()

    def _record_loop(self) -> None:
        writer: cv2.VideoWriter | None = None
        try:
            while not self._stop.is_set():
                frame = self.camera.frame()
                if frame is None:
                    time.sleep(0.05)
                    continue
                self._process_frame(frame)
                record = self.annotate(frame)
                with self._lock:
                    self._display_frame = record.copy()
                if record.shape[1] > 1280:
                    height = max(2, round(record.shape[0] * 1280 / record.shape[1]))
                    record = cv2.resize(record, (1280, height), interpolation=cv2.INTER_AREA)
                if writer is None:
                    writer = cv2.VideoWriter(
                        str(self._video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        6.0,
                        (record.shape[1], record.shape[0]),
                    )
                    if not writer.isOpened():
                        raise RuntimeError("Could not open packet evidence video writer.")
                writer.write(record)
                time.sleep(1.0 / 6.0)
        except Exception as exc:
            with self._lock:
                self._error = str(exc)
        finally:
            if writer is not None:
                writer.release()
            with self._lock:
                self._stop_worker()
                self._active = False
                self._completed = bool(self._sealed is True and not self._error)
                self._status = "completed" if self._completed else "not_sealed"
                if self._completed:
                    self._reason = "Packet sealing confirmed and photos saved."
                elif not self._reason or self._reason == "Saving packet photos.":
                    self._reason = "Packet sealing was not verified. Retry packet closing."
                self._record_thread = None

    def _process_frame(self, frame: np.ndarray) -> None:
        with self._lock:
            worker = self._worker
            kind = self._worker_kind
            if worker is None:
                return
            processing_roi = self.roi.processing_roi_for_frame(frame)
            inference_frame = self.roi.crop_processing(frame)
            worker.submit(inference_frame)
            if kind == "hand":
                payload = worker.result()
                if payload is None:
                    return
                result, error = payload
                if error:
                    self._error = error
                    return
                result = self._offset_result(result, processing_roi, frame.shape[:2])
                self._process_hand(result, frame)
                return
            result = worker.get_result()
            if result is None:
                return
            result = self._offset_result(result, processing_roi, frame.shape[:2])
            confidence = worker.last_confidence
            if kind == "cover":
                self._process_cover(result, confidence, frame)
            else:
                self._process_strip(result, confidence, frame)

    @staticmethod
    def _offset_result(
        result,
        processing_roi: dict[str, int] | None,
        full_shape: tuple[int, int],
    ):
        if result is None or not processing_roi:
            return result
        centroid, mask = result
        x = int(processing_roi["x"])
        y = int(processing_roi["y"])
        width = int(processing_roi["w"])
        height = int(processing_roi["h"])
        roi = (x, y, x + width, y + height)
        if centroid is not None:
            centroid = (int(centroid[0]) + x, int(centroid[1]) + y)
        return centroid, support.offset_mask(mask, roi, full_shape)

    def _process_cover(self, result, confidence: float | None, frame: np.ndarray) -> None:
        _centroid, mask = result
        if self._status in {"packet_check", "packet_final_check"}:
            self._process_packet_recheck(mask, confidence, frame)
            return
        if mask is None:
            self._cover_confirmations = 0
            self._cover_mask = None
            self._cover_confidence = None
            return
        stable = support.mask_matches_reference(
            self._cover_mask,
            mask,
            support.TARGET_BAG_MASK_IOU,
            support.TARGET_BAG_AREA_RATIO,
        )
        self._cover_mask = mask
        self._cover_confidence = confidence
        self._cover_confirmations = self._cover_confirmations + 1 if stable else 1
        self._reason = "Confirming the packet before looking for its strip."
        if self._cover_confirmations >= support.STRIP_CONFIRM_FRAMES:
            self._status = "strip_mode"
            self._reason = "Packet detected. Looking for the packet strip."
            self._start_model("strip")

    def _begin_packet_recheck(self, *, final: bool = False) -> None:
        self._strip_mask = None
        self._strip_confidence = None
        self._cover_confirmations = 0
        self._packet_recheck_frames = 0
        self._packet_recheck_candidate = None
        self._status = "packet_final_check" if final else "packet_check"
        self._reason = "Confirming that the packet remains on the test bed."
        self._start_model("cover")

    def _process_packet_recheck(
        self,
        mask: np.ndarray | None,
        confidence: float | None,
        frame: np.ndarray,
    ) -> None:
        self._packet_recheck_frames += 1
        if mask is None:
            self._cover_confirmations = 0
            self._packet_recheck_candidate = None
        else:
            stable = support.mask_matches_reference(
                self._packet_recheck_candidate,
                mask,
                support.TARGET_BAG_MASK_IOU,
                support.TARGET_BAG_AREA_RATIO,
            )
            self._packet_recheck_candidate = mask.copy()
            self._cover_confirmations = self._cover_confirmations + 1 if stable else 1
            if self._cover_confirmations >= support.STRIP_CONFIRM_FRAMES:
                self._cover_mask = self._packet_recheck_candidate.copy()
                self._cover_confidence = confidence
                self._packet_recheck_candidate = None
                if self._status == "packet_final_check":
                    self._set_terminal(
                        True,
                        "Hands clear, strip absent, and packet remains on the test bed.",
                        frame,
                    )
                else:
                    self._begin_strip_check()
                return
        if self._packet_recheck_frames >= PACKET_RECHECK_MAX_RESULTS:
            self._set_terminal(
                False,
                "Packet was not detected on the test bed after Hand Check-Seal.",
                frame,
            )

    def _process_strip(self, result, confidence: float | None, frame: np.ndarray) -> None:
        _centroid, mask = result
        mask = support.mask_inside_reference(mask, self._cover_mask)
        detected = mask is not None and support.get_centroid(mask) is not None
        if detected:
            self._strip_mask = mask.copy()
            self._strip_confidence = confidence
        if self._status == "strip_mode":
            if not detected:
                self._strip_search_misses += 1
                if self._strip_search_misses >= support.STRIP_CONFIRM_FRAMES * 4:
                    self._status = "tracking"
                    self._reason = "Looking again for the packet and its strip."
                    self._cover_mask = None
                    self._cover_confirmations = 0
                    self._start_model("cover")
                return
            self._strip_search_misses = 0
            self._live_strip_misses = 0
            self._status = "strip_detected"
            self._reason = "Remove the strip and hands, then click Hand Check-Seal."
            self.speak("Remove only the strip and hands, then click Hand Check Seal.")
            return
        if self._status == "strip_detected":
            if detected:
                self._live_strip_misses = 0
            else:
                self._live_strip_misses += 1
                if self._live_strip_misses >= support.STRIP_CONFIRM_FRAMES:
                    self._strip_mask = None
                    self._strip_confidence = None
                    self._reason = "Strip is absent in the live view. Click Hand Check-Seal."
            return
        if self._status != "strip_check":
            return
        if detected:
            self._strip_misses = 0
            self._strip_hits += 1
            if self._strip_hits >= support.STRIP_CONFIRM_FRAMES:
                self._set_terminal(False, "Hands clear, but the strip is still present.", frame)
        else:
            self._strip_hits = 0
            self._strip_misses += 1
            if self._strip_misses >= support.STRIP_DEBOUNCE_FRAMES:
                self._begin_packet_recheck(final=True)

    def _process_hand(self, result, frame: np.ndarray) -> None:
        _centroid, mask = result or (None, None)
        mask = self._validated_hand_mask(mask, frame)
        present = mask is not None
        self._hand_mask = mask.copy() if present else None
        self._hand_frames += 1
        self._hand_hits += int(present)
        self._reason = f"Checking hands ({self._hand_frames} of {HAND_CHECK_FRAMES})."
        if self._hand_frames < HAND_CHECK_FRAMES:
            return
        if self._hand_hits >= HAND_REQUIRED_FRAMES:
            self._status = "strip_detected"
            self._reason = "Hand detected. Remove hands and click Hand Check-Seal again."
            self._hand_mask = None
            self._start_model("strip")
            self.speak("Hand detected. Remove your hands and try again.")
            return
        self._hand_mask = None
        self._begin_packet_recheck()

    def _begin_strip_check(self) -> None:
        self._status = "strip_check"
        self._reason = "Hands clear and packet present. Checking whether the strip is absent."
        self._strip_hits = 0
        self._strip_misses = 0
        self._start_model("strip")

    def _validated_hand_mask(
        self, mask: np.ndarray | None, frame: np.ndarray
    ) -> np.ndarray | None:
        if mask is None:
            return None
        processing_roi = self.roi.processing_roi_for_frame(frame)
        if processing_roi:
            width = int(processing_roi["w"])
            height = int(processing_roi["h"])
        else:
            height, width = frame.shape[:2]
        minimum_area = max(1000, int(width * height * HAND_MIN_AREA_FRACTION))
        minimum_height = max(40, int(height * HAND_MIN_HEIGHT_FRACTION))
        accepted = np.zeros_like(mask, dtype=np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            _x, _y, box_width, box_height = cv2.boundingRect(contour)
            if area < minimum_area or box_height < minimum_height:
                continue
            if box_width / max(1, box_height) > 3.5 and box_height < height * 0.25:
                continue
            cv2.drawContours(accepted, [contour], -1, 255, cv2.FILLED)
        return accepted if np.any(accepted) else None

    def _set_terminal(self, sealed: bool, reason: str, frame: np.ndarray) -> None:
        self._sealed = sealed
        self._status = "sealed" if sealed else "not_sealed"
        self._reason = reason
        self._save_evidence(frame)
        self._strip_mask = None
        self._hand_mask = None
        self._packet_recheck_candidate = None
        self._stop_worker()
        self.speak("Packet sealed." if sealed else "Packet not sealed.")

    def _save_evidence(self, frame: np.ndarray) -> None:
        if self._pledge_dir is None:
            return
        packet_dir = self._pledge_dir / "packet"
        label = "SEALED" if self._sealed else "NOT SEALED"
        color = (50, 220, 50) if self._sealed else (0, 0, 255)
        evidence = frame.copy()
        cv2.rectangle(evidence, (0, 0), (evidence.shape[1], 90), (20, 20, 20), -1)
        cv2.putText(evidence, label, (24, 62), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        self._evidence_path = packet_dir / (
            "packet_sealed.jpg" if self._sealed else "packet_not_sealed.jpg"
        )
        save_jpeg(self._evidence_path, evidence)
        if self._sealed:
            self._final_image_path = packet_dir / "final_image.jpg"
            save_jpeg(self._final_image_path, frame)

    def _start_model(self, kind: str) -> None:
        self._stop_worker()
        if kind == "hand":
            worker = _HandWorker()
            worker.start()
            self._worker = worker
            self._worker_kind = kind
            return
        is_cover = kind == "cover"
        model = support.HailoSegModel(
            str(
                self.base_dir
                / "models"
                / ("bagnewmodel.hef" if is_cover else "packetstrip.hef")
            ),
            conf=support.COVER_CONF_THRESHOLD if is_cover else support.STRIP_CONF_THRESHOLD,
            rgb_input=True,
            label=f"packet-{kind}",
            hailo_model=self._models["bag" if is_cover else "strip"],
        )
        worker = support.SegWorker(model)
        worker.start()
        self._worker_model = model
        self._worker = worker
        self._worker_kind = kind

    def _stop_worker(self) -> None:
        worker = self._worker
        model = self._worker_model
        self._worker = None
        self._worker_model = None
        self._worker_kind = ""
        if worker is not None:
            worker.stop()
            if worker is not threading.current_thread():
                worker.join(timeout=3.0)
        if model is not None:
            model.close()

    def _reset_cycle(self, keep_active: bool = False) -> None:
        self._stop_worker()
        self._error = ""
        self._completed = False
        self._sealed = None
        self._cover_mask = None
        self._cover_confidence = None
        self._cover_confirmations = 0
        self._packet_recheck_frames = 0
        self._packet_recheck_candidate = None
        self._strip_mask = None
        self._strip_confidence = None
        self._strip_search_misses = 0
        self._live_strip_misses = 0
        self._strip_hits = 0
        self._strip_misses = 0
        self._hand_frames = 0
        self._hand_hits = 0
        self._hand_mask = None
        if not keep_active:
            self._active = False

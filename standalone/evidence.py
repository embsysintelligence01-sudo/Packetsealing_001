"""Camera-rate recording, compact per-jewel clips and PDF evidence."""
from collections import deque
from datetime import datetime
from pathlib import Path
from queue import Queue, Full
from zoneinfo import ZoneInfo
import json
import shutil
import subprocess
import threading

import cv2
from .flow import INSIDE

FPS = 10
BUDGET = 2_000_000
TIMEZONE = ZoneInfo("Asia/Kolkata")


def ffmpeg_path():
    import os
    candidate = os.environ.get("JEWELTRACKER_FFMPEG") or shutil.which("ffmpeg")
    local = Path.home() / ".local/share/jeweltracker/ffmpeg/ffmpeg"
    candidate = candidate or (str(local) if local.is_file() else None)
    if not candidate:
        raise RuntimeError("FFmpeg/MJPEG is required for compact evidence export.")
    result = subprocess.run([candidate, "-hide_banner", "-encoders"], capture_output=True, text=True, timeout=15)
    if result.returncode or "mjpeg" not in result.stdout:
        raise RuntimeError("The configured FFmpeg does not provide MJPEG.")
    return candidate


def image_save(path, frame):
    frame = frame.copy()
    if frame.shape[1] > 960:
        frame = cv2.resize(frame, (960, round(frame.shape[0] * 960 / frame.shape[1])))
    if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 78]):
        raise OSError(f"Could not save evidence image: {path}")


class Recorder:
    """Bounded camera queue and four-second pickup pre-roll, independent of inference."""
    def __init__(self, work):
        self.path = work / "session_raw.avi"
        self.queue = Queue(maxsize=35)
        self.enabled = threading.Event()
        self.start = None
        self.end = None
        self.gaps = []
        self.error = ""
        self.last_offered = -1e20
        self.closed = False
        self.thread = threading.Thread(target=self._run, name="packet-evidence", daemon=True)
        self.thread.start()

    def offer(self, frame, t):
        if self.closed or t - self.last_offered < 1 / FPS - .003:
            return
        self.last_offered = t
        try:
            self.queue.put_nowait((t, frame))
        except Full:
            self.gaps.append((t - 1 / FPS, t + 1 / FPS, "Recorder queue full"))

    def close(self):
        self.closed = True
        while self.thread.is_alive():
            try:
                self.queue.put(None, timeout=.2)
                break
            except Full:
                continue
        self.thread.join()
        if self.error:
            raise OSError(self.error)

    def _run(self):
        writer, buffer, previous = None, deque(), None
        count, last_t = 0, None
        try:
            while True:
                item = self.queue.get()
                if item is None:
                    break
                t, frame = item
                height, width = frame.shape[:2]
                scale = min(1., 960 / width)
                size = (max(2, int(width * scale) // 2 * 2), max(2, int(height * scale) // 2 * 2))
                small = cv2.resize(frame, size)
                if writer is None:
                    buffer.append((t, small))
                    while buffer and buffer[0][0] < t - 4:
                        buffer.popleft()
                    if not self.enabled.is_set():
                        continue
                    writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"MJPG"), FPS, size)
                    if not writer.isOpened():
                        raise OSError("Could not open packet recording writer.")
                    self.start = buffer[0][0]
                    pending, buffer = list(buffer), deque()
                else:
                    pending = [(t, small)]
                for timestamp, image in pending:
                    if previous is not None and image.shape != previous.shape:
                        raise OSError("Camera resolution changed during recording.")
                    slot = round((timestamp - self.start) * FPS)
                    if slot < count:
                        continue
                    if last_t is not None and timestamp - last_t > .5:
                        self.gaps.append((last_t, timestamp, "Camera frame gap"))
                    if slot - count > FPS * 5:
                        raise OSError("Camera gap exceeded five seconds; evidence is incomplete.")
                    while count < slot:
                        writer.write(previous if previous is not None else image)
                        count += 1
                    writer.write(image)
                    previous, last_t = image, timestamp
                    count += 1
                    self.end = self.start + count / FPS
        except Exception as exc:
            self.error = str(exc)
        finally:
            if writer is not None:
                writer.release()


def clock_text(t):
    return datetime.fromtimestamp(t, TIMEZONE).strftime("%Y-%m-%d %H:%M:%S %Z") if t is not None else "Not observed"


def validate_video(path, expected):
    cap = cv2.VideoCapture(str(path))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = 0
        while True:
            ok, _ = cap.read()
            if not ok:
                break
            frames += 1
        if not frames or fps <= 0 or abs(frames / fps - expected) > .4:
            raise OSError("Exported clip failed duration/decode verification.")
        return dict(frames=frames, fps=fps, seconds=frames / fps)
    finally:
        cap.release()


def compact_clip(exe, source, target, offset, duration, budget):
    """Keep the whole interval. Report budget failure rather than truncate it."""
    if budget < 100_000 or duration <= .05:
        raise ValueError("Insufficient clip budget or duration.")
    common = [exe, "-nostdin", "-v", "error", "-y", "-threads", "1", "-ss", str(max(0., offset)),
              "-i", str(source), "-t", str(duration), "-an", "-vf", "fps=10,scale=trunc(iw/2)*2:trunc(ih/2)*2",
              "-c:v", "mjpeg", "-vtag", "MJPG", "-threads", "1", "-filter_threads", "1",
              "-pix_fmt", "yuvj420p", "-f", "avi"]
    def run(args):
        result = subprocess.run(args, capture_output=True, text=True, timeout=900)
        if result.returncode:
            raise OSError("Video export failed: " + result.stderr[-800:])
    run([*common, "-q:v", "5", str(target)])
    if target.stat().st_size > budget:
        # Minimum bitrate preserves readable evidence; very long clips may exceed the target.
        rate = max(150_000, int((budget - 24_000) * 8 / duration * .92))
        run([*common, "-b:v", str(rate), "-qmin", "3", "-qmax", "20", str(target)])
    metadata = validate_video(target, duration)
    metadata.update(container="avi", codec="mjpeg", bytes=target.stat().st_size, budget_met=target.stat().st_size <= budget)
    return metadata


def export_session(directory, work, recorder, flow, finish_t, status, exe, sealing_start=None):
    """One shared report, individual jewel clips, and the packet sealing clip."""
    from .report import write_session_report
    summary = directory / "report.pdf"
    result = dict(status=status, sealed=flow.sealed, size_target_bytes=BUDGET,
                  size_scope="Each jewel clip: 2 MB; shared report plus sealing clip: 2 MB",
                  report=summary.name, report_updated=clock_text(datetime.now(TIMEZONE).timestamp()),
                  recorder_gaps=recorder.gaps, recorder_error=recorder.error,
                  unassigned_pickups=flow.unassigned_pickups, jewels=[], exports=[])
    result.update(sealing_method=flow.sealing_method, closure=dict(status=flow.closure_status, confidence=flow.closure_confidence),
                  packet_stationary=flow.packet_stationary, initial_strip_verified=flow.strip_seen,
                  bed_verified_at=flow.bed_verified_at, strip_removed=flow.strip_removed,
                  strip_removed_at=flow.strip_removed_at)
    result['packet_number'] = dict(flow.packet_number)
    result.update(sealing_started_at=sealing_start,
                  sealing_ended_at=flow.strip_removed_at if flow.strip_removed else None)
    if (work / 'sealed.jpg').is_file():
        shutil.copy2(work / 'sealed.jpg', directory / 'sealed_packet.jpg')
        result['sealed_packet_image'] = 'sealed_packet.jpg'
    for key in ('crop', 'scene'):
        name = result['packet_number'].get(key)
        if name and (work / name).is_file():
            shutil.copy2(work / name, directory / name)
    review_reasons = [recorder.error] if recorder.error else []
    keep_raw = bool(recorder.error)
    for jid, j in flow.jewels.items():
        starts = [v for v in [j.started, *(a.get("start") for a in j.attempts)] if v is not None]
        start = min(starts) - 1 if starts else None
        end = min(finish_t, j.clip_ended if j.clip_ended is not None else finish_t)
        gaps = [g for g in recorder.gaps if start is not None and g[0] <= end and g[1] >= start]
        issues = []
        if j.status == INSIDE and (recorder.start is None or recorder.end is None or
                j.started is None or j.ended is None or recorder.start > j.started + .15 or
                recorder.end < end - .15):
            issues.append("The complete transfer interval was not recorded; review required.")
        if gaps and j.status == INSIDE:
            issues.append("Recording gap overlaps transfer; review required.")
        review_reasons.extend(issues)
        keep_raw |= bool(issues)
        record = dict(id=jid, status=j.status, note=j.note, start=j.started, entry=j.entry,
                      end=j.ended, clip_end=j.clip_ended, attempts=j.attempts, pdf=summary.name,
                      placement_method=j.placement_method, evidence_status="review_required" if issues else "recorded",
                      evidence_issues=issues)
        if start is not None and recorder.start is not None and recorder.end is not None:
            lo, hi = max(start, recorder.start), min(end, recorder.end)
            if hi > lo:
                target = directory / f"J{jid}.avi"
                meta = compact_clip(exe, recorder.path, target, lo - recorder.start, hi - lo, BUDGET - 2048)
                record.update(video=target.name, media=meta, combined_bytes=meta["bytes"],
                              budget_met=meta["bytes"] <= BUDGET)
                result["exports"].append(record)
                keep_raw |= not record["budget_met"]
        result["jewels"].append(record)
    if not flow.all_packed or review_reasons or flow.sealing_method != "strip_removal_inferred" or not flow.strip_removed or not flow.strip_removal_requested:
        flow.sealed = False
    result.update(sealed=flow.sealed, evidence_review=review_reasons)
    lo = hi = None
    if recorder.start is not None and recorder.end is not None:
        lo = max(recorder.start, sealing_start - 2) if sealing_start is not None else recorder.start
        hi = min(finish_t, recorder.end)
        if hi > lo:
            result["sealing_video"] = "packet_sealing.avi"
    # Render before sizing the sealing clip; all jewel sections are in this PDF.
    write_session_report(summary, directory, work, flow, result, finish_t, clock_text)
    report_export = dict(pdf=summary.name, combined_bytes=summary.stat().st_size,
                         budget_met=summary.stat().st_size <= BUDGET)
    if result.get("sealing_video"):
        target = directory / result["sealing_video"]
        meta = compact_clip(exe, recorder.path, target, lo - recorder.start, hi - lo,
                            BUDGET - summary.stat().st_size - 2048)
        report_export.update(video=target.name, media=meta, combined_bytes=summary.stat().st_size + meta["bytes"])
        report_export["budget_met"] = report_export["combined_bytes"] <= BUDGET
    result["exports"].append(report_export)
    keep_raw |= not report_export["budget_met"]
    result.update(raw_retained=keep_raw, all_budgets_met=all(r["budget_met"] for r in result["exports"]))
    (directory / "evidence.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not keep_raw:
        shutil.rmtree(work)
    return result

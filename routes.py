from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from flask import Flask

from core.state_machine import (
    WorkflowState,
    now_iso,
    packet_prerequisites,
    transition,
)


def register_packet_sealing_routes(
    app: Flask,
    runtime: Any,
    *,
    ok: Callable[..., Any],
    fail: Callable[..., Any],
    require_state: Callable[..., dict[str, Any]],
    relative_artifact: Callable[[Path], dict[str, Any]],
) -> None:
    @app.post("/api/packet/start")
    def start_packet():
        try:
            with runtime.lock:
                session = require_state(WorkflowState.PACKET_CLOSING_READY)
                ready, reason = packet_prerequisites(session)
                if not ready:
                    raise ValueError(reason)
                packet = runtime.packet.start(runtime.sessions.session_dir())
                session["packet_closing"].update(packet)
                transition(session, WorkflowState.PACKET_CLOSING_ACTIVE, "Packet closing active.")
                runtime.sessions.save()
            return ok(session=runtime.sessions.snapshot(), packet=packet)
        except Exception as exc:
            return fail(str(exc))

    @app.post("/api/packet/hand-clear")
    def packet_hand_clear():
        try:
            with runtime.lock:
                require_state(WorkflowState.PACKET_CLOSING_ACTIVE)
                packet = runtime.packet.request_hand_clear()
            return ok(packet=packet)
        except Exception as exc:
            return fail(str(exc))

    @app.post("/api/packet/retry")
    def packet_retry():
        try:
            with runtime.lock:
                require_state(WorkflowState.PACKET_CLOSING_ACTIVE)
                packet = runtime.packet.retry()
            return ok(packet=packet)
        except Exception as exc:
            return fail(str(exc))

    @app.post("/api/packet/stop")
    def packet_stop():
        try:
            with runtime.lock:
                require_state(WorkflowState.PACKET_CLOSING_ACTIVE)
                packet = runtime.packet.stop()
            return ok(packet=packet)
        except Exception as exc:
            return fail(str(exc))

    @app.post("/api/packet-number/capture")
    def capture_packet_number():
        try:
            with runtime.lock:
                session = require_state(WorkflowState.PACKET_NUMBER_CAPTURE)
                frame = runtime.camera.require_frame()
                packet_number = session["packet_number"]
                try:
                    result = runtime.packet_number.read(
                        frame,
                        runtime.sessions.session_dir() / "packet_number",
                    )
                except Exception:
                    result = {
                        "status": "ocr_error",
                        "message": "The packet number could not be read. Reposition the packet and try again.",
                    }
                packet_number.update(
                    confirmed=False,
                    value=None,
                    pending_value=result.get("packet_number"),
                    status=result["status"],
                    message=result["message"],
                    capture_image=(
                        relative_artifact(Path(result["capture_path"]))
                        if result.get("capture_path")
                        else None
                    ),
                    crop_image=(
                        relative_artifact(Path(result["crop_path"]))
                        if result.get("crop_path")
                        else None
                    ),
                    detector_confidence=result.get("detector_confidence"),
                    ocr_confidence=result.get("ocr_confidence"),
                )
                session["status"] = result["message"]
                session["updated_at"] = now_iso()
                runtime.sessions.save()
            return ok(session=runtime.sessions.snapshot(), packet_number=packet_number)
        except Exception as exc:
            return fail(str(exc))

    @app.post("/api/packet-number/confirm")
    def confirm_packet_number():
        try:
            with runtime.lock:
                session = require_state(WorkflowState.PACKET_NUMBER_CAPTURE)
                packet_number = session["packet_number"]
                value = str(packet_number.get("pending_value") or "").strip()
                if packet_number.get("status") != "detected" or not value:
                    raise ValueError("Capture a readable packet number before confirming.")
                packet_number.update(
                    confirmed=True,
                    value=value,
                    pending_value=None,
                    status="confirmed",
                    message=f"Packet number {value} saved against pledge {session['pledge_id']}.",
                    confirmed_at=now_iso(),
                )
                transition(session, WorkflowState.PLEDGE_COMPLETE, "Pledge complete.")
                runtime._ensure_report(session)
                runtime.sessions.save()
            return ok(session=runtime.sessions.snapshot())
        except Exception as exc:
            return fail(str(exc))

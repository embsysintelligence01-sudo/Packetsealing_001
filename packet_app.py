#!/usr/bin/env python3
"""Launch the independent packet application with the existing SLIC environment."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
TRACKER = Path(os.environ.get("JEWEL_TRACKER_DIR", ROOT.parent / "jewel_tracker/jewel_tracker"))
if not (TRACKER / "jewel_tracker_5.py").is_file():
    raise SystemExit("Set JEWEL_TRACKER_DIR to the directory containing jewel_tracker_5.py.")
sys.path.insert(0, str(TRACKER))
os.environ.setdefault("YOLO_AUTOINSTALL", "False")
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")


def main():
    parser = argparse.ArgumentParser(description="Standalone SLIC jewel-to-packet sealing application")
    parser.add_argument("--check-models", action="store_true", help="Load all required models and run a CPU smoke check without a camera")
    parser.add_argument("--smoke-ui", action="store_true", help="Construct the UI and exit without opening a camera")
    parser.add_argument("--check-ocr", action="store_true", help="Test number detector and isolated PaddleOCR; downloads OCR models on first use")
    parser.add_argument("--replay", type=Path, help="Check a saved video with real models and deterministic source timestamps")
    parser.add_argument("--replay-report", type=Path, default=ROOT / "outputs/video_replay.json")
    parser.add_argument("--replay-step", type=int, default=6, help="Analyze every Nth video frame (default 6)")
    args = parser.parse_args()
    if args.replay_step < 1:
        parser.error("--replay-step must be positive")
    if args.check_ocr:
        from standalone.packet_ocr import check_ocr
        check_ocr(ROOT)
        return 0
    if args.replay:
        from standalone.replay import replay
        replay(args.replay, args.replay_report, args.replay_step)
        return 0
    # Load the tracker first, preserving its torch/OpenCV/Qt import ordering.
    from standalone.app import MainWindow
    from PyQt5 import QtWidgets
    if args.check_models:
        import cv2
        import numpy as np
        from standalone.models import PacketModels
        from standalone.tracking import TrackingBridge
        from standalone.evidence import ffmpeg_path
        if not hasattr(cv2, "ximgproc"):
            raise RuntimeError("SLIC is unavailable; use .venv-slic.")
        models = PacketModels(ROOT / "model onnx")
        frame = np.full((480, 640, 3), 230, np.uint8)
        roi = np.full(frame.shape[:2], 255, np.uint8)
        bag, strip, clear, _, _ = models.inspect(frame, roi)
        bridge = TrackingBridge([(0, 0), (639, 0), (639, 479), (0, 479)])
        bridge.process(frame, 1.)
        import jewel_tracker_5 as tracker
        extractor = tracker.JewelIdModel()
        features = extractor.features([frame[100:200, 100:200]])
        bridge.close()
        print(f"PASS: 5 packet models, INT8 gold detector, SLIC, ResNet features {features.shape}, FFmpeg {ffmpeg_path()}")
        print(f"Blank-frame smoke result: bag={bag is not None}, strip={strip}, hands_clear={clear}")
        print("Sealing rule: verified strip removal; no additional closure model required.")
        return 0
    app = QtWidgets.QApplication(sys.argv[:1])
    window = MainWindow(auto_connect=not args.smoke_ui)
    if args.smoke_ui:
        window.show()
        app.processEvents()
        window.close()
        print("PASS: standalone PyQt UI constructed without starting a camera.")
        return 0
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())

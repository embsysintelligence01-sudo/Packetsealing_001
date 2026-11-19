import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import packet_app
import jewel_tracker_5 as original
import numpy as np
from PyQt5 import QtWidgets
from live_pipeline import LatestPreview

from standalone.app import MainWindow, PacketWorker
from standalone.models import Detection, PacketModels
from standalone.settings import tracker_camera, tracker_roi
from standalone.tracking import GoldPredictor
from standalone.flow import PackingFlow, INSIDE


class PreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_jewel_table_reuses_widgets_and_retains_selection(self):
        window = MainWindow()
        self.addCleanup(window.close)
        item = dict(id=1, thumb=np.zeros((30, 30, 3), np.uint8), status='On bench', note='Waiting')
        state = dict(phase='Packing', strip='unknown', strip_enabled=True, packet_stationary=False,
                     clear=True, packet=True, closure='pending', closure_message='Waiting',
                     identities=True, can_draw=True, directory='/tmp', jewels=[item])
        window.show_state(state)
        widget = window.table.cellWidget(0, 1)
        cells = [window.table.item(0, col) for col in (0, 2, 3)]
        window.table.selectRow(0)
        item.update(status=INSIDE, note='Placement inferred')
        window.show_state(state)
        self.assertIs(window.table.cellWidget(0, 1), widget)
        for cell, column in zip(cells, (0, 2, 3)):
            self.assertIs(window.table.item(0, column), cell)
        self.assertEqual(window.table.item(0, 2).text(), INSIDE)
        self.assertEqual(window.table.currentRow(), 0)

    def test_preview_reads_new_camera_frames_without_inference_results(self):
        window = MainWindow()
        window.roi = []
        frames = [np.full((1440, 2560, 3), color, np.uint8) for color in ((10, 40, 120), (30, 80, 150))]
        grabber = Mock()
        grabber.latest.side_effect = [(frames[0], 100.), (frames[1], 100.04)]
        # No model completion, result signal or tracker is needed for either frame.
        window.worker = SimpleNamespace(grabber=grabber, live_overlay=LatestPreview(), isRunning=lambda: False)
        window.refresh_preview()
        self.assertEqual(window.last_image.pixelColor(200, 200).red(), 120)
        window.refresh_preview()
        self.assertEqual(window.last_image.pixelColor(200, 200).red(), 150)
        self.assertEqual(window.last_image.width(), 1280)
        self.assertEqual(window.video.frame_size, (2560, 1440))
        self.assertEqual(window.preview_timestamp, 100.04)
        window.close()

    def test_no_manual_process_buttons_and_automatic_camera_start(self):
        with patch.object(MainWindow, "connect_camera") as connect:
            window = MainWindow(auto_connect=True)
            self.app.processEvents()
            connect.assert_called_once()
            labels = [button.text() for button in window.findChildren(QtWidgets.QPushButton)]
            for forbidden in ("Mark packet opening", "Hand Check / Seal", "Recheck selected jewel on bench",
                              "Reset jewel IDs (before packing)"):
                self.assertNotIn(forbidden, labels)
            window.close()

    def test_hand_mask_does_not_tint_camera_pixels(self):
        worker = PacketWorker(0, [])
        frame = np.full((240, 320, 3), (15, 80, 150), np.uint8)
        out = worker._draw(frame, None, np.zeros((240, 320), np.uint8), np.full((240, 320), 255, np.uint8))
        np.testing.assert_array_equal(out[20:180], frame[20:180])

    def test_latched_placement_is_not_queried_again_for_observations(self):
        worker = PacketWorker(0, [])
        worker.flow = PackingFlow([1])
        worker.flow.jewels[1].status = INSIDE
        tracker = SimpleNamespace(tracks=[SimpleNamespace(id=1)], id_model=Mock())
        worker.tracker = tracker
        frame = np.zeros((100, 100, 3), np.uint8)
        self.assertEqual(worker._observations(frame, 100., None), {})
        tracker.id_model.decide.assert_not_called()

    def test_fresh_gold_overlay_is_visible_when_packet_overlay_has_expired(self):
        window = MainWindow()
        window.roi = []
        frame = np.full((240, 320, 3), (10, 40, 120), np.uint8)
        old_packet, fresh_gold = LatestPreview(max_age=.75), LatestPreview(max_age=.75)
        image = np.zeros_like(frame)
        image[50, 40] = (0, 255, 0)
        old_packet.publish(image, 100.)
        image = np.zeros_like(frame)
        image[70, 80] = (0, 255, 255)
        fresh_gold.publish(image, 100.9)
        grabber = Mock()
        grabber.latest.return_value = (frame, 101.)
        window.worker = SimpleNamespace(grabber=grabber, live_overlay=old_packet,
                                        gold_overlay=fresh_gold, isRunning=lambda: False)
        window.refresh_preview()
        self.assertEqual(window.last_image.pixelColor(80, 70).getRgb()[:3], (255, 255, 0))
        self.assertEqual(window.last_image.pixelColor(40, 50).getRgb()[:3], (120, 40, 10))
        window.close()

    def test_gold_predictor_loads_existing_int8_model(self):
        with patch("standalone.tracking.OpenVINOYolo") as model:
            GoldPredictor()
            model.assert_called_once_with(original.MODEL_PATH)
            self.assertEqual(Path(original.MODEL_PATH).name, "gold_int8.xml")

    def test_shared_tracker_roi_and_camera_resolution_are_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            zones, camera = Path(tmp) / "zones.json", Path(tmp) / "camera.json"
            points = [[262, 19], [2269, 8], [2266, 1420], [329, 1417]]
            zones.write_text(json.dumps(dict(roi=points)))
            camera.write_text(json.dumps(dict(usb=dict(resolution="2560x1440", antiflicker=dict(enabled=True, shutter_denom=50)))))
            with patch.object(original, "ZONES_FILE", str(zones)), patch.object(original, "CAMERA_FILE", str(camera)):
                self.assertEqual(tracker_roi(), points)
                config = tracker_camera(0)
                self.assertEqual(config["resolution"], "2560x1440")
                self.assertTrue(config["antiflicker"]["enabled"])
                self.assertEqual(config["antiflicker"]["shutter_denom"], 50)

    def test_packet_inspection_preserves_source_frame_masks_and_coordinates(self):
        models = PacketModels.__new__(PacketModels)
        frame = np.zeros((1440, 2560, 3), np.uint8)
        roi = np.full(frame.shape[:2], 255, np.uint8)
        context = roi.copy()
        zone = np.zeros_like(roi)
        zone[144:170, 256:1280] = 255
        bag = Detection(roi, (256, 144, 1280, 720), .9)
        result = (bag, "present", True, zone, roi)
        for context_mask, inspection_zone in ((None, None), (context, None),
                                               (None, zone), (context, zone)):
            with self.subTest(context=context_mask is not None, zone=inspection_zone is not None):
                models._inspect_native = Mock(return_value=result)
                actual = models.inspect(frame, roi, context_mask, inspection_zone)
                args = models._inspect_native.call_args.args
                self.assertEqual(models._inspect_native.call_count, 1)
                for passed, source in zip(args, (frame, roi, context_mask, inspection_zone)):
                    self.assertIs(passed, source)
                self.assertIs(actual, result)
                self.assertEqual(actual[0].box, (256, 144, 1280, 720))
                for mask in (actual[0].mask, actual[3], actual[4]):
                    self.assertEqual(mask.shape, (1440, 2560))


if __name__ == "__main__":
    unittest.main()

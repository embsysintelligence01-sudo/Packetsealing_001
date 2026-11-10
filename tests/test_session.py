"""Real coordinator/recorder/exporter with controlled gold and bag observations."""
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import packet_app
import jewel_tracker_5 as original
import numpy as np

from standalone.app import PacketWorker
from standalone.evidence import Recorder
from standalone.models import Detection
from standalone.flow import INSIDE, PICKED, AT_PACKET
from standalone.tracking import TrackingBridge


class SessionTests(unittest.TestCase):
    def test_automatic_pickup_before_bag_then_overlap_disappearance_and_seal(self):
        self.run_session()

    def test_strip_removal_and_closure_complete_with_body_context_after_bag_detector_fails(self):
        self.run_session(drop_bag_after_baseline=True)

    def run_session(self, drop_bag_after_baseline=False):
        frame = np.full((240, 400, 3), 225, np.uint8)
        mask = np.zeros(frame.shape[:2], np.uint8)
        mask[50:200, 200:380] = 255
        bag = Detection(mask, (200, 50, 380, 200), .95)
        started = time.time()
        clock = dict(index=-1)
        class OCR:
            def __init__(self, root):
                self.calls = 0
            def update(self, image, roi, t):
                self.calls += 1
                status = 'confirmed' if self.calls >= 2 else 'reading'
                result = dict(status=status, text='PC123456', confidence=.98,
                              confirmed_at=t if status == 'confirmed' else None,
                              message='Packet number confirmed.' if status == 'confirmed' else 'Rotate the packet.')
                accepted = dict(crop=image[:60, :180], scene=image) if status == 'confirmed' else None
                return result, accepted
            def close(self):
                pass
        capture_flags, recorders, gold_updates = [], [], []
        det = original.Det(np.array([10, 90, 30, 110]), np.array([20., 100.]), 400., .95,
                           mask=np.full((20, 20), 255, np.uint8), origin=(10, 90))
        tr = original.Track(1, det, started)
        tr.thumb = frame[90:110, 10:30].copy()

        class Tracker:
            pickup_event = TrackingBridge.pickup_event
            def __init__(self, roi):
                self.roi = np.asarray(roi, np.int32)
                self.mode = "LOCKED"
                self.tracks = [tr]
                self.id_model = Mock(ready=True)
                self.id_model.decide.return_value = ("jewel", 1, .95, {})
                self.identity_error = ""
                self.returners = {}
                self.recent_hand = {}
                self.continuous_identity = {1: True}
                self.moving_now = False

            def ensure_roi(self):
                pass

            def process(self, frame, t, capture):
                capture_flags.append(capture)
                i = clock["index"]
                tr.in_hand = 3 <= i <= 8
                if i >= 9:
                    tr.state, tr.spot = "OCCLUDED", "removed"
                    return []
                x = 20. if i < 4 else 250.
                tr.pos = np.array([x, 100.])
                tr.det_box = np.array([x - 10, 90, x + 10, 110])
                tr.det_origin = (int(x - 10), 90)
                tr.box = tr.det_box
                tr.det_seen = t
                tr.state, tr.spot = "OK", "jewel"
                return []

            def close(self):
                pass

        models = Mock()
        empty = np.zeros(frame.shape[:2], np.uint8)
        def inspect(frame, roi, context_mask=None):
            i = clock["index"]
            # Packet inference waits for the same frame's gold publication.
            deadline = time.monotonic() + 2
            timestamp = started + i * .3
            while (not gold_updates or gold_updates[-1] != timestamp) and time.monotonic() < deadline:
                time.sleep(.005)
            self.assertEqual(gold_updates[-1], timestamp)
            if i == 3:
                self.assertTrue(recorders[0].enabled.is_set(), "Pickup must enable recording before the bag exists")
            if drop_bag_after_baseline and i >= 12:
                self.assertIsNotNone(context_mask)
            observed_bag = bag if i >= 4 and not (drop_bag_after_baseline and i >= 12) else None
            return observed_bag, "present" if i < 12 else "absent", not (3 <= i <= 8), empty, empty
        models.inspect.side_effect = inspect

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            worker = PacketWorker("controlled-source", [(0, 0), (399, 0), (399, 239), (0, 239)])
            if drop_bag_after_baseline:
                class BodyContext:
                    ready = False
                    strip_view_ok = True
                    def arm(self, *args):
                        self.ready = True
                    def locate(self, frame):
                        return bag if self.ready else None
                worker.seal_context = BodyContext()
            exports, errors, states = [], [], []
            publish = worker.gold_overlay.publish
            def publish_gold(image, timestamp, **kwargs):
                publish(image, timestamp, **kwargs)
                gold_updates.append(timestamp)
            worker.gold_overlay.publish = publish_gold
            worker.exported.connect(exports.append)
            worker.failed.connect(errors.append)
            worker.state.connect(states.append)
            class Grabber:
                on_frame = None
                problem = ""
                def start(self):
                    pass
                def latest(self, after):
                    clock["index"] += 1
                    i = clock["index"]
                    if i > 25:
                        raise AssertionError("Automatic session did not reach sealing completion")
                    t = started + i * .3
                    self.on_frame(frame, t)
                    return frame.copy(), t
                def stop(self):
                    pass
                def join(self, timeout):
                    pass
            def recorder_factory(work):
                recorder = Recorder(work)
                recorders.append(recorder)
                return recorder
            with patch("standalone.app.ROOT", root), patch("standalone.app.PacketModels", return_value=models), \
                 patch("standalone.app.PacketOCR", OCR), \
                 patch("standalone.app.TrackingBridge", Tracker), patch("standalone.app.Recorder", side_effect=recorder_factory), \
                 patch("standalone.app.make_grabber", return_value=Grabber()):
                worker.run()
            self.assertEqual(errors, [])
            self.assertTrue(capture_flags[0])
            self.assertEqual(len(exports), 1)
            result = exports[0]
            self.assertTrue(result["sealed"], worker.flow.jewels[1].note)
            self.assertEqual(result['packet_number']['text'], 'PC123456')
            self.assertEqual(result['packet_number']['status'], 'confirmed')
            self.assertGreaterEqual(worker.ocr.calls, 2)
            self.assertTrue(result["all_budgets_met"])
            self.assertAlmostEqual(result["jewels"][0]["start"], started + .9)
            seen_states = [s["jewels"][0]["status"] for s in states]
            self.assertIn(PICKED, seen_states)
            self.assertNotIn(AT_PACKET, seen_states)  # Placement now latches on first interaction.
            self.assertIn(INSIDE, seen_states)
            self.assertFalse(states[3]["packet"])
            self.assertEqual(states[3]["jewels"][0]["status"], PICKED)
            directory = Path(result["directory"])
            for name in ("J1.avi", "report.pdf", "packet_sealing.avi", "evidence.json",
                         "packet_number_crop.jpg", "packet_number_scene.jpg"):
                self.assertTrue((directory / name).is_file(), name)
            self.assertFalse((directory / ".working").exists())
            if drop_bag_after_baseline:
                self.assertFalse(states[-1]['packet'])
                self.assertFalse(states[-1]['packet_stationary'])
                self.assertTrue(states[-1]['seal_context'])
                self.assertTrue(states[-1]['strip_removed'])
                self.assertEqual(result['sealing_method'], 'strip_removal_inferred')


if __name__ == "__main__":
    unittest.main()

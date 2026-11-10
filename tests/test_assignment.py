import unittest
from unittest.mock import Mock, patch

import packet_app
import jewel_tracker_5 as original
import numpy as np

from standalone.tracking import TrackingBridge, draw_gold, packet_overlap


class AssignmentTests(unittest.TestCase):
    def detection(self, x, y):
        return original.Det(np.array([x - 10, y - 10, x + 10, y + 10]),
                            np.array([float(x), float(y)]), 400., .9,
                            mask=np.full((20, 20), 255, np.uint8), origin=(x - 10, y - 10))

    def bridge(self):
        with patch("standalone.tracking.GoldPredictor"):
            tracker = TrackingBridge([(0, 0), (399, 0), (399, 299), (0, 299)])
        tracker.learn_ids = Mock()
        return tracker

    def lock(self, tracker):
        frame = np.full((300, 400, 3), 230, np.uint8)
        detections = [self.detection(300, 200), self.detection(200, 60), self.detection(80, 60)]
        tracker.detect = Mock(return_value=(detections, []))
        for i in range(7):
            tracker.process(frame, 100. + i * .35)
        return frame

    def test_automatic_count_assigns_reading_order_without_capture_command(self):
        tracker = self.bridge()
        self.lock(tracker)
        self.assertEqual(tracker.mode, "LOCKED")
        self.assertEqual([tr.id for tr in tracker.tracks], [1, 2, 3])
        self.assertEqual([tuple(tr.pos) for tr in tracker.tracks], [(80., 60.), (200., 60.), (300., 200.)])
        tracker.learn_ids.assert_called_once()
        tracker.close()

    def test_original_matching_keeps_ids_when_detector_order_changes(self):
        tracker = self.bridge()
        frame = self.lock(tracker)
        tracker.detect.return_value = ([self.detection(305, 202), self.detection(85, 62), self.detection(205, 62)], [])
        tracker.process(frame, 102.2)
        self.assertEqual([tr.id for tr in tracker.tracks], [1, 2, 3])
        self.assertEqual([tr.det_box[0] for tr in tracker.tracks], [75, 195, 295])
        tracker.close()

    def test_new_unclaimed_detection_does_not_add_or_renumber_ids_after_lock(self):
        tracker = self.bridge()
        frame = self.lock(tracker)
        tracker.detect.return_value = ([self.detection(80, 60), self.detection(200, 60),
                                       self.detection(300, 200), self.detection(50, 250)], [])
        tracker.process(frame, 102.2)
        self.assertEqual([tr.id for tr in tracker.tracks], [1, 2, 3])
        tracker.close()

    def test_reset_invalidates_old_gallery_and_restarts_count(self):
        tracker = self.bridge()
        self.lock(tracker)
        old_revision = tracker.flow_revision
        tracker.identity_training_token = object()
        tracker.id_model.ready = True
        tracker.reset_assignment()
        self.assertEqual(tracker.mode, "ARMING")
        self.assertEqual(tracker.tracks, [])
        self.assertFalse(tracker.id_model.ready)
        self.assertIsNone(tracker.identity_training_token)
        self.assertGreater(tracker.flow_revision, old_revision)
        tracker.close()

    def test_accepted_gold_mask_is_visible_before_any_id_or_flow_exists(self):
        tracker = self.bridge()
        tracker.detections = [self.detection(80, 60)]
        image = np.zeros((300, 400, 3), np.uint8)
        draw_gold(image, tracker, 100.)
        self.assertGreater(np.count_nonzero(image[45:75, 65:95]), 0)
        self.assertEqual(tracker.tracks, [])
        self.assertEqual(tracker.mode, "ARMING")
        tracker.close()

    def test_recording_trigger_uses_original_in_hand_state(self):
        tracker = self.bridge()
        self.lock(tracker)
        tracker.tracks[0].in_hand = True
        event = tracker.pickup_event(102.1)
        self.assertTrue(event.active)
        self.assertEqual(event.jewel_id, 1)
        self.assertFalse(event.ambiguous)
        tracker.close()

    def test_under_hand_or_absence_alone_does_not_start_a_recording(self):
        tracker = self.bridge()
        self.lock(tracker)
        tr = tracker.tracks[0]
        tr.in_hand, tr.state, tr.spot = False, "OCCLUDED", "hand"
        self.assertFalse(tracker.pickup_event(102.2).active)
        tr.spot = "removed"
        tracker.moving_now = False
        self.assertFalse(tracker.pickup_event(102.3).active)
        # The original visibility policy also accepts removal + bench motion.
        tracker.moving_now = True
        self.assertTrue(tracker.pickup_event(102.4).active)
        tracker.close()

    def test_multiple_in_hand_jewels_keep_recording_unassigned(self):
        tracker = self.bridge()
        self.lock(tracker)
        tracker.tracks[0].in_hand = tracker.tracks[1].in_hand = True
        event = tracker.pickup_event(102.1)
        self.assertTrue(event.active)
        self.assertTrue(event.ambiguous)
        self.assertIsNone(event.jewel_id)
        tracker.close()

    def test_packet_overlap_uses_gold_pixels_instead_of_box_center(self):
        jewel = np.full((20, 100), 255, np.uint8)
        packet = np.zeros((20, 100), np.uint8)
        packet[:, 40:60] = 255
        # Center is inside, but only 20% of the actual jewel overlaps the bag.
        self.assertAlmostEqual(packet_overlap(jewel, (0, 0), packet), .2)
        self.assertEqual(packet_overlap(None, (0, 0), packet), 0.)

    def test_segmented_hand_uses_initial_bench_id_after_live_track_swap(self):
        tracker = self.bridge()
        self.lock(tracker)
        tracker.tracks[0].in_hand = True  # Original trigger nominates a live ID.
        tracker.tracks[0].pos = np.array([300., 200.])
        hand = np.zeros((300, 400), np.uint8)
        hand[185:215, 285:315] = 255
        tracker.detections = []  # Jewel leaves its fixed bench spot.
        tracker.pickup_event(102., hand_mask=hand)
        pickup = tracker.pickup_event(102.2, hand_mask=hand)
        self.assertTrue(pickup.active)
        self.assertEqual(pickup.jewel_id, 3)  # Fixed initial spot, not moved J1.
        tracker.taken_ids = {3}
        self.assertFalse(tracker.pickup_event(102.2, hand_mask=hand).active)
        tracker.close()

    def test_fast_pickup_retains_contact_when_hand_leaves_before_gold_disappears(self):
        tracker = self.bridge()
        self.lock(tracker)
        hand = np.zeros((300, 400), np.uint8)
        hand[185:215, 285:315] = 255
        self.assertFalse(tracker.pickup_event(102.2, hand_mask=hand).active)
        # Hand departs immediately; two fresh gold observations establish that
        # the bangle left, using remembered contact rather than waiting for it.
        tracker.detections = []
        tracker.raw_detections = [self.detection(250, 180)]
        empty = np.zeros_like(hand)
        tracker.pickup_event(102.4, hand_mask=empty)
        event = tracker.pickup_event(102.6, hand_mask=empty)
        self.assertEqual(event.jewel_id, 3)
        self.assertIn(3, tracker.pickup_seeds)  # Associated courier seed is retained.
        tracker.close()

    def test_raw_hsv_pickup_without_segmented_hand_is_suppressed(self):
        tracker = self.bridge()
        self.lock(tracker)
        tracker.tracks[0].in_hand = True
        self.assertFalse(tracker.pickup_event(102.2, hand_mask=np.zeros((300, 400), np.uint8)).active)
        tracker.close()

    def test_final_pickup_keeps_gold_detector_running_without_id_reassignment(self):
        tracker = self.bridge()
        frame = self.lock(tracker)
        tracker.inactive_ids = {tr.id for tr in tracker.tracks}
        tracker.transfer_active = True
        tracker.detect.reset_mock()
        tracker.update_tracks = Mock()
        raw = tracker.process(frame, 103.)
        tracker.detect.assert_called_once()
        tracker.update_tracks.assert_not_called()
        self.assertTrue(raw)
        self.assertEqual(tracker.raw_timestamp, 103.)
        self.assertEqual(tracker.raw_detections, raw)
        tracker.transfer_active = False
        tracker.process(frame, 104.)
        self.assertEqual(tracker.raw_detections, [])
        tracker.close()

    def test_gold_inference_stops_when_all_jewels_are_confirmed(self):
        tracker = self.bridge()
        frame = self.lock(tracker)
        tracker.inactive_ids = {tr.id for tr in tracker.tracks}
        tracker.detect.reset_mock()
        tracker.process(frame, 103.)
        tracker.detect.assert_not_called()
        self.assertEqual(tracker.detections, [])
        self.assertFalse(tracker.pickup_event(103.).active)
        tracker.close()


if __name__ == "__main__":
    unittest.main()

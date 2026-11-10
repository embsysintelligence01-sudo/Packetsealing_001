import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import packet_app
from standalone.packet_motion import PacketMotion
from standalone.models import Detection, mask_iou
from standalone.app import PacketWorker
from standalone.flow import PackingFlow, Pickup, Observation, INSIDE, PICKED


class PacketMotionTests(unittest.TestCase):
    def scene(self):
        frame = np.full((360, 640, 3), 200, np.uint8)
        rng = np.random.default_rng(7)
        frame[80:240, 100:280] = rng.integers(20, 220, (160, 180, 3), dtype=np.uint8)
        mask = np.zeros(frame.shape[:2], np.uint8)
        mask[80:240, 100:280] = 255
        return frame, Detection(mask, (100, 80, 280, 240), .9)

    def test_moving_packet_is_followed_when_detector_misses(self):
        frame, bag = self.scene()
        tracker = PacketMotion()
        self.assertIs(tracker.update(frame, bag, 100.), bag)
        matrix = np.float32([[1, 0, 25], [0, 1, 12]])
        moved = cv2.warpAffine(frame, matrix, (640, 360))
        result = tracker.update(moved, None, 100.2)
        expected = cv2.warpAffine(bag.mask, matrix, (640, 360))
        self.assertEqual(tracker.source, 'motion')
        self.assertGreater(mask_iou(result.mask, expected), .95)

    def test_fresh_detection_at_new_position_is_accepted_immediately(self):
        frame, bag = self.scene()
        tracker = PacketMotion()
        tracker.update(frame, bag, 100.)
        moved = Detection(np.roll(bag.mask, 230, axis=1), (330, 80, 510, 240), .8)
        self.assertIs(tracker.update(frame, moved, 100.2), moved)
        self.assertEqual(tracker.source, 'model')

    def test_repeated_tracking_does_not_extend_detection_age_forever(self):
        frame, bag = self.scene()
        tracker = PacketMotion()
        tracker.update(frame, bag, 100.)
        for t in (100.5, 101., 101.5, 102.):
            self.assertIsNotNone(tracker.update(frame, None, t))
        self.assertIsNone(tracker.update(frame, None, 102.6))
        self.assertEqual(tracker.source, 'unavailable')

    def test_no_features_or_no_previous_bag_does_not_invent_packet(self):
        frame, bag = self.scene()
        tracker = PacketMotion()
        self.assertIsNone(tracker.update(frame, None, 100.))
        blank = np.zeros_like(frame)
        tracker.update(blank, bag, 100.2)
        self.assertIsNotNone(tracker.update(blank, None, 100.4))
        self.assertEqual(tracker.source, 'recent')
        self.assertIsNone(tracker.update(blank, None, 100.8))

    def test_completed_transfer_hand_is_not_checked_again(self):
        worker = PacketWorker(0, [])
        worker.flow = PackingFlow([1])
        worker.flow.jewels[1].status = INSIDE
        worker.packet_motion = Mock()
        worker.packet_motion.update.return_value = None
        worker._observations = Mock(return_value={})
        worker.transfer = Mock(owner=1)
        worker.transfer.observe.side_effect = AssertionError('Completed transfer must not be checked')
        worker._transfer_observations(np.zeros((300, 400, 3), np.uint8), 100., None, None, Pickup())
        worker.transfer.observe.assert_not_called()

    def test_worker_checks_outgoing_transfer_before_starting_new_pickup(self):
        worker = PacketWorker(0, [])
        worker.flow = PackingFlow([1, 2])
        worker.flow.update(100., {}, pickup=Pickup(True, 1), packet_ok=False, hands_clear=False, strip='uncertain')
        worker.tracker = SimpleNamespace(tracks=[SimpleNamespace(id=2, pos=(300, 100))])
        worker.packet_motion = Mock()
        worker.packet_motion.update.return_value = None
        worker._observations = Mock(return_value={})
        class Follower:
            owner = 1
            def observe(self, *args):
                return Observation((100, 100), near_packet=True, identity_ok=False) if self.owner == 1 else None
            def start(self, owner, *args):
                self.owner = owner
        worker.transfer = Follower()
        pickup = Pickup(True, 2)
        observations = worker._transfer_observations(np.zeros((300, 400, 3), np.uint8), 100.2,
                                                    None, None, pickup)
        self.assertIn(1, observations)
        worker.flow.update(100.2, observations, pickup=pickup, packet_ok=False, hands_clear=False, strip='uncertain')
        self.assertEqual(worker.flow.jewels[1].status, INSIDE)
        self.assertEqual(worker.flow.jewels[2].status, PICKED)

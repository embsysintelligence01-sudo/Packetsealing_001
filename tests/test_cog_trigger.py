"""Third mode requires fingertip contact and matching visible gold motion."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import cv2
import numpy as np
from pickup_trigger_test import COGMovementEngine, CandidateClips, Models, make_engine


class COGTriggerTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((240, 400, 3), 215, np.uint8)
        self.roi = np.full(self.frame.shape[:2], 255, np.uint8)
        self.box = (80, 95, 130, 135)
        self.events = []
        self.engine = COGMovementEngine(self.events.append, threshold=20.)
        self.assertIsInstance(self.engine, COGMovementEngine)
        self.engine.set_references(self.frame, [(1, self.detection(self.box))], [], self.roi, 0.)
        self.engine.compare = Mock(side_effect=AssertionError('No home verification'))
        self.engine.scene_stable = lambda gray: True

    def detection(self, box):
        mask = np.zeros(self.frame.shape[:2], np.uint8)
        mask[box[1]:box[3], box[0]:box[2]] = 255
        return SimpleNamespace(box=box, mask=mask, confidence=.95)

    def hand(self, x=0):
        hand = self.detection((70+x, 70, 150+x, 190))
        hand.points = np.full((21, 2), (100+x, 180), np.float32)
        hand.points[4] = (125+x, 110)
        hand.scale = 80.
        return hand

    def update(self, x, t, hand_x=None, fresh=True, visibility='VISIBLE', hands=True):
        box = (80+x, 95, 130+x, 135)
        tracked = {1: dict(box=box, seen=t if fresh else t-1, visibility=visibility)}
        self.engine.update(self.frame, [self.detection(box)],
                           [self.hand(x if hand_x is None else hand_x)] if hands else [],
                           [], self.roi, t, tracked)
        return self.engine.jewels[0]

    def motion(self):
        for i in range(4):
            jewel = self.update(i*10, 1.+i*.25)
        return jewel

    def test_contact_records_then_three_motion_observations_trigger(self):
        jewel = self.update(0, 1.)
        self.assertIsNotNone(jewel.contact_at)
        self.assertNotEqual('MOVEMENT_TRIGGERED', jewel.state)
        self.update(10, 1.25)
        self.update(20, 1.5)
        self.assertNotEqual('MOVEMENT_TRIGGERED', jewel.state)
        jewel = self.update(30, 1.75)
        self.assertEqual('MOVEMENT_TRIGGERED', jewel.state)
        self.assertEqual('coherent_motion_verified', jewel.verification_status)
        self.assertEqual('same_fingertip_and_jewel_motion', jewel.pickup_basis)
        self.assertIsNone(jewel.verified_at)  # no empty-home verification
        self.engine.compare.assert_not_called()

    def test_jewel_movement_without_hand_does_not_trigger(self):
        for i in range(4):
            jewel = self.update(i*10, 1.+i*.25, hands=False)
        self.assertEqual('ON_BENCH', jewel.state)

    def test_hand_movement_without_jewel_does_not_trigger(self):
        for i in range(4):
            jewel = self.update(0, 1.+i*.25, hand_x=i*10)
        self.assertNotEqual('MOVEMENT_TRIGGERED', jewel.state)

    def test_opposite_motion_does_not_trigger(self):
        for i in range(4):
            jewel = self.update(i*10, 1.+i*.25, hand_x=-i*10)
        self.assertNotEqual('MOVEMENT_TRIGGERED', jewel.state)

    def test_stale_hidden_or_missing_gold_resets_votes(self):
        for visibility, fresh in (('VISIBLE', False), ('UNDER_HAND', False)):
            with self.subTest(visibility=visibility, fresh=fresh):
                self.setUp()
                self.update(0, 1.)
                self.update(10, 1.25)
                self.assertEqual(1, self.engine.jewels[0].votes)
                jewel = self.update(20, 1.5, fresh=fresh, visibility=visibility)
                self.assertIsNone(jewel.current_cog)
                self.assertEqual(0, jewel.votes)
                self.assertNotEqual('MOVEMENT_TRIGGERED', jewel.state)

    def test_fresh_complete_gold_can_move_with_old_in_hand_label(self):
        for i in range(4):
            jewel = self.update(i*10, 1.+i*.25, visibility='IN_HAND')
        self.assertEqual('MOVEMENT_TRIGGERED', jewel.state)

    def test_fresh_under_hand_segmentation_can_supply_motion(self):
        for i in range(4):
            jewel = self.update(i*10, 1.+i*.25, visibility='UNDER_HAND')
        self.assertEqual('MOVEMENT_TRIGGERED', jewel.state)
        self.assertIsNotNone(jewel.current_cog)

    def test_under_hand_label_without_gold_cannot_invent_cog(self):
        self.update(0, 1.)
        self.engine.update(self.frame, [], [self.hand(20)], [], self.roi, 1.25,
                           {1:dict(box=self.box, seen=1.25, visibility='UNDER_HAND')})
        jewel = self.engine.jewels[0]
        self.assertIsNone(jewel.current_cog)
        self.assertEqual(0, jewel.votes)

    def test_small_partial_gold_mask_cannot_supply_cog(self):
        self.update(0, 1.)
        box = (100, 95, 150, 135)
        detection = self.detection(box)
        detection.mask[:, 115:] = 0
        self.engine.update(self.frame, [detection], [self.hand(20)], [], self.roi, 1.25,
                           {1:dict(box=box, seen=1.25, visibility='VISIBLE')})
        self.assertIsNone(self.engine.jewels[0].current_cog)
        self.assertEqual(0, self.engine.jewels[0].votes)

    def test_gap_and_camera_motion_reset_votes(self):
        self.update(0, 1.)
        self.update(10, 1.25)
        self.update(20, 2.5)
        self.assertEqual(0, self.engine.jewels[0].votes)
        self.engine.scene_stable = lambda gray: False
        self.update(30, 2.75)
        self.assertEqual(0, self.engine.jewels[0].votes)

    def test_lost_hand_cannot_transfer_pending_identity(self):
        self.update(0, 1.)
        self.update(10, 1.25)
        jewel = self.update(20, 1.5, hands=False)
        self.assertIsNone(jewel.associated_hand)
        self.update(30, 1.75)
        self.assertNotEqual('MOVEMENT_TRIGGERED', jewel.state)

    def test_stationary_candidate_cancels_and_requires_exit(self):
        for t in (1., 1.5, 2., 2.5, 2.75):
            jewel = self.update(0, t)
        self.assertEqual('PICKUP_CANCELLED', jewel.state)
        self.assertEqual(1., jewel.contact_at)
        self.update(0, 3., hands=False)
        self.update(0, 3.25)
        self.assertEqual(3.25, jewel.contact_at)

    def test_return_ends_motion_without_duplicate_triggers(self):
        self.motion()
        self.update(40, 2.)
        self.assertEqual(1, sum(e['state']=='MOVEMENT_TRIGGERED' for e in self.events))
        self.assertEqual('MOVEMENT_ENDED', self.update(0, 2.25).state)

    def test_merged_mask_cannot_supply_cog(self):
        box = (80, 95, 190, 135)
        self.engine.update(self.frame, [self.detection(box)], [self.hand()], [], self.roi, 1.,
            {1:dict(box=box,seen=1.),2:dict(box=(140,100,160,130),seen=1.)})
        self.assertIsNone(self.engine.jewels[0].current_cog)

    def test_two_equally_close_homes_do_not_get_arbitrary_hand(self):
        other = self.detection((80, 140, 130, 180))
        self.engine.set_references(self.frame, [(1, self.detection(self.box)), (2, other)], [], self.roi, 0.)
        hand = self.hand()
        hand.points[:] = (125, 137)
        self.engine.update(self.frame, [self.detection(self.box), other], [hand], [], self.roi, 1.,
            {1:dict(box=self.box,seen=1.,visibility='VISIBLE'),
             2:dict(box=other.box,seen=1.,visibility='VISIBLE')})
        self.assertFalse(self.engine.links)

    def test_proximity_opens_real_recording_and_return_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            clips = CandidateClips(Path(tmp), 5)
            self.addCleanup(clips.close)
            self.update(0, 1.)
            clips.update(self.frame, self.engine.jewels, 1.)
            self.assertIn(1, clips.writers)
            for i in range(1, 4):
                self.update(i*10, 1.+i*.25)
                clips.update(self.frame, self.engine.jewels, 1.+i*.25)
            self.update(0, 2.)
            clips.update(self.frame, self.engine.jewels, 2.)
            self.assertNotIn(1, clips.writers)
            cap = cv2.VideoCapture(str(next(Path(tmp).glob('*.avi'))))
            self.assertTrue(cap.read()[0])
            cap.release()

    def test_custom_distance_and_invalid_threshold(self):
        for threshold in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                make_engine('cog', Mock(), cog_threshold=threshold)
        self.engine.threshold = 40.
        self.assertNotEqual('MOVEMENT_TRIGGERED', self.motion().state)
        self.assertEqual('MOVEMENT_TRIGGERED', self.update(50, 2.).state)

    def test_full_frame_input_then_roi_filtering(self):
        models = object.__new__(Models)
        models.keypoints = Mock()
        outside = self.hand(200)
        inside = self.hand()
        models.keypoints.detect.return_value = [outside, inside]
        roi = np.zeros_like(self.roi)
        roi[80:140, 80:140] = 255
        result = models.detect_keypoints(self.frame, roi)
        self.assertEqual(len(result), 1)
        self.assertIs(result[0], inside)
        self.assertIs(models.keypoints.detect.call_args.args[0], self.frame)

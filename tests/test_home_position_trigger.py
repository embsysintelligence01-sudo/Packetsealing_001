import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np

from pickup_trigger_test import (CandidateClips, HomePositionEngine, PickupEngine,
                                 make_engine, region)


class HomePositionTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.full((240, 400, 3), 215, np.uint8)
        self.roi = np.full(self.frame.shape[:2], 255, np.uint8)
        self.box = (80, 95, 130, 135)
        cv2.rectangle(self.frame, self.box[:2], self.box[2:], (20, 120, 180), -1)
        self.gold = self.detection(self.box)
        self.events = []
        self.engine = HomePositionEngine(self.events.append)
        self.engine.set_references(self.frame, [(1, self.gold)], [], self.roi, 0.)
        self.engine.scene_stable = lambda gray: True
        self.engine.assess_scene = lambda frame, excluded: self.engine.scene_stable(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        self.jewel = self.engine.jewels[0]
        self.assertTrue(self.jewel.verification_info['valid'])

    def detection(self, box):
        mask = np.zeros(self.frame.shape[:2], np.uint8)
        mask[box[1]:box[3], box[0]:box[2]] = 255
        return SimpleNamespace(box=box, mask=mask, confidence=.95)

    def observation(self, t, box=None, visibility='VISIBLE', fresh=True):
        return {1: dict(box=box or self.box, seen=t if fresh else 0., visibility=visibility)}

    def update(self, t, frame=None, gold=None, hands=(), bags=(), tracked=None):
        self.engine.update(self.frame if frame is None else frame,
                           [self.gold] if gold is None else gold,
                           list(hands), list(bags), self.roi, t,
                           self.observation(t) if tracked is None else tracked)

    def test_modes_select_different_engines(self):
        self.assertIs(type(make_engine('previous', Mock())), PickupEngine)
        self.assertIs(type(make_engine('home', Mock())), HomePositionEngine)
        with self.assertRaises(ValueError):
            make_engine('unknown', Mock())

    def test_home_cog_uses_segmentation_and_remains_fixed(self):
        mask = self.gold.mask.copy()
        mask[:, 100:] = 0
        e = HomePositionEngine(Mock())
        e.set_references(self.frame, [(1, SimpleNamespace(box=self.box, mask=mask))], [], self.roi, 0.)
        self.assertLess(e.jewels[0].home_cog[0], 100.)
        home = self.jewel.home_cog.copy()
        moved = self.detection((180, 95, 230, 135))
        self.update(1., gold=[moved], tracked=self.observation(1., moved.box))
        np.testing.assert_array_equal(home, self.jewel.home_cog)

    def test_contact_starts_recording_immediately_and_covered_home_cannot_confirm(self):
        hand = self.detection((70, 80, 150, 150))
        with tempfile.TemporaryDirectory() as tmp:
            clips = CandidateClips(Path(tmp), 5)
            self.addCleanup(clips.close)
            for t in (1., 1.25, 1.6, 2.):
                self.update(t, frame=np.full_like(self.frame, 215), gold=[], hands=[hand],
                            tracked=self.observation(t, visibility='UNDER_HAND', fresh=False))
                clips.update(self.frame, self.engine.jewels, t)
                self.assertIsNotNone(self.jewel.contact_at)
                self.assertIsNone(self.jewel.current_cog)
                self.assertIsNone(self.jewel.pickup_at)
                self.assertIsNone(self.jewel.verified_at)
            self.assertIn(1, clips.writers)
            self.assertEqual(1., self.jewel.contact_at)
            self.assertTrue(any(e['state'] == 'PICKUP_CANDIDATE' for e in self.events))

    def test_hand_association_follows_hand_without_inventing_jewel_cog(self):
        hand = self.detection((70, 80, 150, 150))
        self.update(1., hands=[hand], gold=[], tracked=self.observation(1., visibility='UNDER_HAND', fresh=False))
        key = self.jewel.associated_hand
        shifted = self.detection((85, 80, 165, 150))
        self.update(1.2, hands=[shifted], gold=[], tracked=self.observation(1.2, visibility='UNDER_HAND', fresh=False))
        self.assertEqual(key, self.jewel.associated_hand)
        self.assertGreater(self.jewel.hand_position[0], 110.)
        self.assertIsNone(self.jewel.current_cog)
        self.assertIsNone(self.jewel.velocity)

    def test_stationary_touch_cancels_when_home_is_clear(self):
        self.update(1., hands=[self.detection((70, 80, 150, 150))])
        for t in (1.2, 1.45, 1.7):
            self.update(t)
        self.assertEqual('PICKUP_CANCELLED', self.jewel.state)
        self.update(1.9)
        self.assertEqual('PICKUP_CANCELLED', self.jewel.state)
        self.update(2.1, hands=[self.detection((70, 80, 150, 150))])
        self.assertEqual(2.1, self.jewel.contact_at)

    def test_under_hand_departure_waits_for_uncovered_empty_home(self):
        self.update(1., hands=[self.detection((70, 80, 150, 150))], gold=[],
                    tracked=self.observation(1., visibility='UNDER_HAND', fresh=False))
        for t in (1.2, 1.45, 1.7):
            self.update(t, frame=np.full_like(self.frame, 215), gold=[],
                        tracked=self.observation(t, visibility='NOT_SEEN', fresh=False))
        self.assertEqual('PICKUP_CONFIRMED', self.jewel.state)
        self.assertEqual(1.2, self.jewel.pickup_at)
        self.assertEqual('verified_empty_home_after_occlusion', self.jewel.pickup_basis)
        self.assertIsNone(self.jewel.current_cog)

    def test_visible_cog_displacement_and_velocity_with_empty_home(self):
        empty = np.full_like(self.frame, 215)
        for t, box in ((1., (180, 95, 230, 135)), (1.25, (190, 95, 240, 135))):
            gold = self.detection(box)
            frame = empty.copy()
            cv2.rectangle(frame, box[:2], box[2:], (20, 120, 180), -1)
            self.update(t, frame=frame, gold=[gold], tracked=self.observation(t, box))
        self.assertGreater(self.jewel.displacement, 90.)
        self.assertAlmostEqual(40., self.jewel.velocity)
        self.assertEqual('PICKUP_DETECTED', self.jewel.state)
        self.update(1.5, frame=empty, gold=[gold], tracked=self.observation(1.5, gold.box))
        self.assertEqual('PICKUP_CONFIRMED', self.jewel.state)

    def test_camera_bag_and_gap_block_verification(self):
        empty = np.full_like(self.frame, 215)
        self.update(1., hands=[self.detection((70, 80, 150, 150))])
        self.engine.scene_stable = lambda gray: False
        self.update(1.2, frame=empty, gold=[])
        self.assertEqual(0, self.jewel.votes)
        self.engine.scene_stable = lambda gray: True
        self.engine.assess_scene = lambda frame, excluded: self.engine.scene_stable(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        self.update(1.4, frame=empty, gold=[], bags=[self.detection((50, 60, 170, 170))])
        self.assertEqual(0, self.jewel.votes)
        self.update(3., frame=empty, gold=[])
        self.assertEqual(0, self.jewel.votes)
        self.assertIsNone(self.jewel.velocity)

    def test_shared_hand_keeps_two_homes_independent_and_merge_cog_unknown(self):
        second = self.detection((170, 95, 220, 135))
        cv2.rectangle(self.frame, second.box[:2], second.box[2:], (20, 120, 180), -1)
        self.engine = HomePositionEngine(self.events.append)
        self.engine.set_references(self.frame, [(1, self.gold), (2, second)], [], self.roi, 0.)
        self.engine.scene_stable = lambda gray: True
        self.engine.assess_scene = lambda frame, excluded: self.engine.scene_stable(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        hand = self.detection((70, 80, 240, 150))
        merged = self.detection((80, 95, 220, 135))
        for t in (1., 1.25, 1.6):
            tracked = {i: dict(box=merged.box, seen=t, visibility='VISIBLE') for i in (1, 2)}
            self.update(t, gold=[merged], hands=[hand], tracked=tracked)
        for jewel in self.engine.jewels:
            self.assertIsNotNone(jewel.contact_at)
            self.assertIsNone(jewel.current_cog)
            self.assertNotEqual('PICKUP_CONFIRMED', jewel.state)
        # Only J1 leaves; J2 remains and is independently cancelled.
        remaining = np.full_like(self.frame, 215)
        cv2.rectangle(remaining, second.box[:2], second.box[2:], (20, 120, 180), -1)
        for t in (1.8, 2.05, 2.3):
            tracked = {1: dict(box=self.box, seen=0., visibility='NOT_SEEN'),
                       2: dict(box=second.box, seen=t, visibility='VISIBLE')}
            self.update(t, frame=remaining, gold=[second], tracked=tracked)
        self.assertEqual('PICKUP_CONFIRMED', self.engine.jewels[0].state)
        self.assertEqual('PICKUP_CANCELLED', self.engine.jewels[1].state)

    def test_home_capture_waits_for_fresh_stable_uncovered_tracks(self):
        engine = HomePositionEngine(Mock())
        track = SimpleNamespace(id=7, det_box=self.box, det_mask=region(self.gold.mask, self.box),
                                det_origin=self.box[:2], det_seen=0.)
        hand = self.detection((70, 80, 150, 150))
        for t in (0., .5, 1., 1.6):
            track.det_seen = t
            engine.adopt_tracks(self.frame, [track], [], self.roi, t, [hand])
        self.assertFalse(engine.jewels)
        for t in (2., 2.5, 3., 3.6):
            track.det_seen = t
            engine.adopt_tracks(self.frame, [track], [], self.roi, t, [])
        self.assertEqual([7], [j.id for j in engine.jewels])

    def test_hand_merge_does_not_inherit_either_old_hand_identity(self):
        left = self.detection((40, 70, 110, 150))
        right = self.detection((130, 70, 200, 150))
        old = self.engine.associate_hands([left, right], 1.)
        merged = self.detection((40, 70, 200, 150))
        new = self.engine.associate_hands([merged], 1.2)
        self.assertFalse(set(old) & set(new))

    def test_packet_hint_never_confirms_a_covered_home(self):
        hand = self.detection((70, 80, 150, 150))
        self.update(1., hands=[hand])
        moved = self.detection((200, 100, 250, 140))
        bag = self.detection((180, 80, 280, 170))
        for t in (1.2, 1.45, 1.7):
            self.update(t, gold=[moved], hands=[hand], bags=[bag],
                        tracked=self.observation(t, moved.box))
        self.assertIn('insertion unverified', self.jewel.destination_hint)
        self.assertNotEqual('PICKUP_CONFIRMED', self.jewel.state)
        self.assertIsNone(self.jewel.verified_at)

    def test_dropout_at_unchanged_home_does_not_trigger(self):
        for t in (1., 1.25, 1.6):
            self.update(t, gold=[], tracked=self.observation(t, visibility='NOT_SEEN', fresh=False))
        self.assertEqual('ON_BENCH', self.jewel.state)

    def partial_sequence(self, *, hand_box=(115, 80, 150, 150), empty=True,
                         visibility='VISIBLE', fresh=True, moved=True):
        hand = self.detection(hand_box)
        for t in (1., 1.25, 1.5):
            box = (180+int((t-1)*20), 95, 230+int((t-1)*20), 135) if moved else self.box
            gold = self.detection(box)
            frame = np.full_like(self.frame, 215) if empty else self.frame.copy()
            cv2.rectangle(frame, box[:2], box[2:], (20, 120, 180), -1)
            cv2.rectangle(frame, hand_box[:2], hand_box[2:], (100, 150, 190), -1)
            self.update(t, frame=frame, gold=[gold], hands=[hand],
                        tracked=self.observation(t, box, visibility, fresh))
            if t == 1.:
                self.assertNotEqual('PICKUP_CONFIRMED', self.jewel.state)

    def test_partial_empty_home_and_same_visible_jewel_confirm(self):
        self.partial_sequence()
        self.assertEqual('PICKUP_CONFIRMED', self.jewel.state)
        self.assertEqual('visible_cog_and_partial_empty_home', self.jewel.pickup_basis)
        self.assertGreaterEqual(self.jewel.metrics['partial_visible_fraction'], .5)

    def test_partial_home_less_than_half_visible_stays_pending(self):
        self.partial_sequence(hand_box=(100, 80, 150, 150))
        self.assertNotEqual('PICKUP_CONFIRMED', self.jewel.state)

    def test_partial_home_original_gold_pixels_block_confirmation(self):
        self.partial_sequence(empty=False)
        self.assertNotEqual('PICKUP_CONFIRMED', self.jewel.state)

    def test_partial_home_stale_or_under_hand_jewel_cannot_confirm(self):
        for visibility, fresh in (('VISIBLE', False), ('UNDER_HAND', True)):
            with self.subTest(visibility=visibility, fresh=fresh):
                self.setUp()
                self.partial_sequence(visibility=visibility, fresh=fresh)
                self.assertNotEqual('PICKUP_CONFIRMED', self.jewel.state)

    def test_partial_home_stationary_jewel_cannot_confirm(self):
        self.partial_sequence(moved=False)
        self.assertNotEqual('PICKUP_CONFIRMED', self.jewel.state)

    def test_partial_home_camera_instability_blocks_confirmation(self):
        self.engine.scene_stable = lambda gray: False
        self.partial_sequence()
        self.assertEqual(0, self.jewel.votes)
        self.assertNotEqual('PICKUP_CONFIRMED', self.jewel.state)

    def test_partial_home_packet_blocks_confirmation(self):
        hand = self.detection((115, 80, 150, 150))
        bag = self.detection((70, 80, 110, 150))
        gold = self.detection((180, 95, 230, 135))
        for t in (1., 1.25, 1.5):
            self.update(t, frame=np.full_like(self.frame, 215), gold=[gold],
                        hands=[hand], bags=[bag], tracked=self.observation(t, gold.box))
        self.assertEqual(0, self.jewel.votes)
        self.assertNotEqual('PICKUP_CONFIRMED', self.jewel.state)

    def test_partial_votes_reset_when_visible_movement_is_lost(self):
        # Start a fresh candidate and collect only two votes before losing the ID.
        hand = self.detection((115, 80, 150, 150))
        gold = self.detection((180, 95, 230, 135))
        empty = np.full_like(self.frame, 215)
        for t in (1., 1.25):
            self.update(t, frame=empty, gold=[gold], hands=[hand], tracked=self.observation(t, gold.box))
        self.assertEqual(2, self.jewel.votes)
        self.update(1.5, frame=empty, gold=[], hands=[hand], tracked={})
        self.assertEqual(0, self.jewel.votes)
        self.assertNotEqual('PICKUP_CONFIRMED', self.jewel.state)

    def pose_hand(self, offset=0):
        hand = self.detection((70+offset, 80, 150+offset, 160))
        hand.points = np.full((21, 2), (140+offset, 150), np.float32)
        hand.points[0] = (140+offset, 160)
        hand.points[9] = (140+offset, 100)
        hand.points[4] = (105+offset, 115)
        hand.points[8] = (110+offset, 115)
        hand.scale = 60.
        return hand

    def test_28_percent_mask_is_accepted_for_provisional_tracking(self):
        partial=self.detection(self.box)
        partial.mask[:,94:]=0
        self.update(1., gold=[partial], hands=[self.pose_hand()],
                    tracked=self.observation(1.,visibility='IN_HAND'))
        self.assertIsNotNone(self.jewel.current_cog)
        self.assertAlmostEqual(.28,self.jewel.gold_quality['area_ratio'])
        self.assertTrue(self.jewel.gold_quality['partial'])
        self.assertEqual('partial_gold_provisional',self.jewel.gold_quality['reason'])
        self.assertNotEqual('PICKUP_CONFIRMED',self.jewel.state)

    def test_below_25_percent_mask_reports_the_actual_rejection(self):
        partial=self.detection(self.box)
        partial.mask[:,90:]=0
        self.update(1., gold=[partial], hands=[self.pose_hand()],
                    tracked=self.observation(1.,visibility='IN_HAND'))
        self.assertIsNone(self.jewel.current_cog)
        self.assertEqual('gold_mask_below_minimum',self.jewel.gold_quality['reason'])

    def test_partial_measurement_remains_partial_when_landmarks_drop_out(self):
        partial=self.detection(self.box)
        partial.mask[:,94:]=0
        self.update(1.,gold=[partial],hands=[],tracked=self.observation(1.,visibility='IN_HAND'))
        self.assertIsNotNone(self.jewel.current_cog)
        self.assertTrue(self.jewel.gold_quality['partial'])
        self.assertNotEqual('PICKUP_CONFIRMED',self.jewel.state)

    def test_pose_contact_records_and_passing_hand_cancels(self):
        self.update(1., hands=[self.pose_hand()])
        self.assertEqual(1., self.jewel.contact_at)
        self.assertTrue(self.jewel.metrics['fingertip_contact'])
        for t in (1.2, 1.45, 1.7):
            self.update(t)
        self.assertEqual('PICKUP_CANCELLED', self.jewel.state)

    def test_pose_pickup_confirms_after_hand_leaves_empty_home(self):
        self.update(1., hands=[self.pose_hand()])
        moved = self.detection((260, 95, 310, 135))
        for t in (1.2, 1.45, 1.7):
            self.update(t, frame=np.full_like(self.frame, 215), gold=[moved],
                        hands=[self.pose_hand(180)],
                        tracked=self.observation(t, moved.box, 'UNDER_HAND'))
        self.assertEqual('PICKUP_CONFIRMED', self.jewel.state)
        self.assertIsNotNone(self.jewel.current_cog)
        self.assertGreater(self.jewel.displacement, 100.)

    def test_pose_guard_cannot_claim_half_home_is_visible(self):
        moved = self.detection((260, 95, 310, 135))
        for t in (1., 1.25, 1.5):
            self.update(t, frame=np.full_like(self.frame, 215), gold=[moved],
                        hands=[self.pose_hand()], tracked=self.observation(t, moved.box))
        self.assertTrue(self.jewel.metrics['home_covered'])
        self.assertFalse(self.jewel.metrics['partial_home_enabled'])
        self.assertEqual(0, self.jewel.votes)
        self.assertNotEqual('PICKUP_CONFIRMED', self.jewel.state)

    def test_pose_palm_or_other_finger_does_not_replace_thumb_index_contact(self):
        hand = self.pose_hand()
        hand.points[4] = (175, 170)
        hand.points[8] = (180, 175)
        hand.points[12] = (105, 115)
        self.update(1., hands=[hand])
        self.assertTrue(self.jewel.metrics['home_covered'])
        self.assertFalse(self.jewel.metrics['fingertip_contact'])
        self.assertIsNone(self.jewel.contact_at)
        self.assertEqual('ON_BENCH', self.jewel.state)

    def test_finger_side_near_gold_can_start_without_tip_overlap(self):
        hand = self.pose_hand()
        hand.points[:] = (150, 170)
        hand.points[2] = (105, 85)
        hand.points[3] = (105, 115)
        hand.points[4] = (105, 160)
        hand.points[8] = (180, 180)
        # Thumb tip is 25 pixels below gold; its distal segment crosses gold.
        self.update(1., hands=[hand])
        self.assertEqual(1., self.jewel.contact_at)
        self.assertTrue(self.jewel.metrics['hand_jewel_proximity'])

    def test_proximity_uses_live_mask_away_from_fixed_home(self):
        moved = self.detection((260, 95, 310, 135))
        self.update(1., hands=[self.pose_hand(180)], gold=[moved],
                    tracked=self.observation(1., moved.box, 'UNDER_HAND'))
        self.assertTrue(self.jewel.metrics['hand_jewel_proximity'])
        self.assertEqual('live_gold_mask', self.jewel.metrics['proximity_source'])
        self.assertIsNotNone(self.jewel.associated_hand)

    def test_ring_hole_is_not_treated_as_gold_box_interior(self):
        from standalone.hand_keypoints import home_contact_mask
        hand = self.pose_hand()
        hand.points[:] = (105, 115)
        hand.scale = 40.
        mask = np.zeros(self.frame.shape[:2], np.uint8)
        cv2.circle(mask, (105, 115), 30, 255, 4)
        self.assertFalse(np.any((home_contact_mask(hand, self.frame.shape) > 0) & (mask > 0)))

    def test_pose_hand_near_two_jewels_keeps_association_ambiguous(self):
        second = self.detection((140, 95, 190, 135))
        self.engine.set_references(self.frame, [(2, second)], [], self.roi, 0.)
        hand = self.pose_hand()
        hand.points[2] = hand.points[3] = hand.points[4] = (135, 115)
        hand.points[6] = hand.points[7] = hand.points[8] = (135, 115)
        tracks = {1:dict(box=self.box, seen=1., visibility='VISIBLE'),
                  2:dict(box=second.box, seen=1., visibility='VISIBLE')}
        self.update(1., gold=[self.gold, second], hands=[hand], tracked=tracks)
        for jewel in self.engine.jewels:
            self.assertIsNone(jewel.associated_hand)
            self.assertIsNone(jewel.contact_at)

    def test_matching_finger_gold_motion_is_detected_pending_clear_home(self):
        for index in range(4):
            shift = index*6
            moved = self.detection((80+shift, 95, 130+shift, 135))
            self.update(1.+index*.25, frame=np.full_like(self.frame, 215), gold=[moved],
                        hands=[self.pose_hand(shift)],
                        tracked=self.observation(1.+index*.25, moved.box, 'UNDER_HAND'))
        self.assertEqual('PICKUP_DETECTED', self.jewel.state)
        self.assertTrue(self.jewel.metrics['hand_jewel_moving_together'])
        self.assertIsNone(self.jewel.verified_at)
        self.assertFalse(self.jewel.departed)

    def test_finger_motion_without_gold_motion_stays_unconfirmed(self):
        for index in range(4):
            self.update(1.+index*.25, hands=[self.pose_hand(index*6)])
        self.assertFalse(self.jewel.metrics['hand_jewel_moving_together'])
        self.assertIsNone(self.jewel.pickup_at)


if __name__ == '__main__':
    unittest.main()

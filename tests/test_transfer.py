import unittest
from types import SimpleNamespace
import cv2
import numpy as np
from standalone.transfer import TransferFollower


class TransferTests(unittest.TestCase):
    def setUp(self):
        self.shape = (400, 600, 3)
        mask = np.zeros(self.shape[:2], np.uint8)
        mask[100:300, 50:200] = 255
        self.bag = SimpleNamespace(mask=mask)
        self.follower = TransferFollower()
        self.follower.start(2, (500, 200), 0.)

    def hands(self, x=None):
        mask = np.zeros(self.shape[:2], np.uint8)
        cv2.rectangle(mask, (55, 150), (90, 260), 255, -1)  # Bag holder.
        if x is not None:
            cv2.circle(mask, (x, 200), 30, 255, -1)
        return mask

    def test_bag_holder_cannot_confirm_bench_pickup(self):
        for t in (.2, .4, .6):
            self.assertIsNone(self.follower.observe(t, self.hands(), self.bag, self.shape))
        self.assertFalse(self.follower.hand_acquired)

    def test_associated_courier_reaches_packet_without_hands_clear(self):
        result = None
        for i, x in enumerate((500, 450, 400, 350, 300, 250, 200, 180)):
            result = self.follower.observe(i * .2, self.hands(x), self.bag, self.shape)
        self.assertIsNotNone(result)
        self.assertTrue(result.transfer_verified)
        self.assertTrue(result.near_packet)
        self.assertTrue(result.transfer_verified)
        self.assertFalse(result.evidence["opening_verified"])
        self.assertEqual(self.follower.owner, 2)

    def test_lost_hand_does_not_jump_to_packet_holder(self):
        self.follower.observe(.2, self.hands(500), self.bag, self.shape)
        self.assertIsNone(self.follower.observe(2., self.hands(), self.bag, self.shape))

    def test_waiting_for_pickup_hand_does_not_expire_initial_bench_seed(self):
        self.assertIsNone(self.follower.observe(5., self.hands(500), self.bag, self.shape))
        self.assertTrue(self.follower.hand_acquired)

    def test_hand_path_recovers_locally_outside_packet_after_gap(self):
        self.follower.observe(.2, self.hands(500), self.bag, self.shape)
        self.assertIsNone(self.follower.observe(3., self.hands(500), self.bag, self.shape))
        result = None
        for i, x in enumerate((450, 400, 350, 300, 250, 200, 180)):
            result = self.follower.observe(3.2+i*.2, self.hands(x), self.bag, self.shape)
        self.assertIsNotNone(result)
        self.assertTrue(result.transfer_verified)

    def test_starting_on_packet_or_pressing_it_is_only_contact(self):
        self.follower.start(2, (120, 200), 0.)
        for t in (.2, .4, .6, .8):
            result = self.follower.observe(t, self.hands(120), self.bag, self.shape)
            self.assertIsNotNone(result)
            self.assertFalse(result.transfer_verified)
            self.assertEqual(result.method, 'pickup_owned_hand_contact')

    def test_packet_moving_under_stationary_hand_cannot_confirm(self):
        for i in range(11):
            mask = np.zeros(self.shape[:2], np.uint8)
            left = 50+40*i
            mask[100:300, left:min(600, left+150)] = 255
            hand = np.zeros_like(mask)
            cv2.circle(hand, (500, 200), 30, 255, -1)
            result = self.follower.observe(i*.2, hand, SimpleNamespace(mask=mask), self.shape)
            self.assertFalse(result is not None and result.transfer_verified)

    def test_flattening_packet_discards_old_approach(self):
        for i, x in enumerate((500, 450, 400, 350, 300, 250)):
            self.follower.observe(i*.2, self.hands(x), self.bag, self.shape)
        mask = np.zeros(self.shape[:2], np.uint8)
        mask[80:320, 50:290] = 255
        for t in (1.2, 1.4, 1.6):
            result = self.follower.observe(t, self.hands(250), SimpleNamespace(mask=mask), self.shape)
            self.assertIsNotNone(result)
            self.assertFalse(result.transfer_verified)

    def test_duplicate_frame_cannot_supply_second_arrival(self):
        for i, x in enumerate((500, 450, 400, 350, 300, 250, 200)):
            result = self.follower.observe(i*.2, self.hands(x), self.bag, self.shape)
        self.assertIsNotNone(result)
        self.assertFalse(result.transfer_verified)
        for _ in range(5):
            self.assertIsNone(self.follower.observe(1.2, self.hands(200), self.bag, self.shape))
        result = self.follower.observe(1.4, self.hands(180), self.bag, self.shape)
        self.assertTrue(result.transfer_verified)
        self.assertEqual(result.evidence['arrival_observations'], 2)

    def test_missing_packet_breaks_boundary_evidence(self):
        for i, x in enumerate((500, 450, 400, 350, 300, 250)):
            self.follower.observe(i*.2, self.hands(x), self.bag, self.shape)
        self.assertIsNone(self.follower.observe(1.2, self.hands(200), None, self.shape))
        for t in (1.4, 1.6):
            result = self.follower.observe(t, self.hands(180), self.bag, self.shape)
            self.assertIsNotNone(result)
            self.assertFalse(result.transfer_verified)

    def test_arrival_missed_then_hand_seen_on_packet_body_is_contact_only(self):
        for i, x in enumerate((500, 450, 400, 350, 300, 250)):
            self.follower.observe(i*.2, self.hands(x), self.bag, self.shape)
        near = np.zeros(self.shape[:2], np.uint8)
        cv2.circle(near, (210, 200), 30, 255, -1)
        self.follower.observe(1.2, near, self.bag, self.shape)
        # A jump to the packet centre misses the boundary crossing.
        hand = np.zeros(self.shape[:2], np.uint8)
        cv2.circle(hand, (125, 200), 30, 255, -1)
        for t in (1.4, 1.6, 1.8, 2.):
            result = self.follower.observe(t, hand, self.bag, self.shape)
            self.assertIsNotNone(result)
            self.assertFalse(result.transfer_verified)

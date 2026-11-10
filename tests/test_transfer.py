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
        for i, x in enumerate((500, 450, 400, 350, 300, 250, 200)):
            result = self.follower.observe(i * .2, self.hands(x), self.bag, self.shape)
        self.assertIsNotNone(result)
        self.assertTrue(result.near_packet)
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
        for i, x in enumerate((450, 400, 350, 300, 250, 200)):
            result = self.follower.observe(3.2+i*.2, self.hands(x), self.bag, self.shape)
        self.assertIsNotNone(result)

import unittest
from unittest.mock import Mock
import numpy as np

from standalone.models import Detection, PacketModels


class FilterTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.zeros((100, 100, 3), np.uint8)
        self.mask = np.full((100, 100), 255, np.uint8)
        self.bag = Detection(self.mask, (0, 0, 100, 100), .9)
        self.strip_mask = np.zeros((100, 100), np.uint8)
        self.strip_mask[5:15, 5:95] = 255
        self.strip = Detection(self.strip_mask, (5, 5, 95, 15), .9)
        self.models = PacketModels.__new__(PacketModels)
        self.models.bag = Mock()
        self.models.bag.detect.return_value = [self.bag]
        self.models.strip = Mock()
        self.models.strip.detect.return_value = [self.strip]
        self.models.hand = Mock()
        self.models.hand.detect.return_value = []
        self.models.filter_bag = Mock(return_value=.9)
        self.models.filter_strip = Mock(return_value=.9)

    def test_rejected_strip_candidate_is_uncertain(self):
        self.models.filter_strip.return_value = .1
        _, strip, _, _, _ = self.models.inspect(self.frame, self.mask)
        self.assertEqual(strip, "uncertain")

    def test_no_raw_strip_candidate_is_absent(self):
        self.models.strip.detect.return_value = []
        _, strip, _, _, _ = self.models.inspect(self.frame, self.mask)
        self.assertEqual(strip, "absent")

    def test_accepted_strip_is_present(self):
        _, strip, _, _, _ = self.models.inspect(self.frame, self.mask)
        self.assertEqual(strip, "present")

    def test_rejected_bag_never_verifies_packet(self):
        self.models.filter_bag.return_value = .1
        bag, _, _, _, _ = self.models.inspect(self.frame, self.mask)
        self.assertIsNone(bag)

    def test_multiple_packets_cannot_be_selected_silently(self):
        self.models.bag.detect.return_value = [self.bag, self.bag]
        bag, _, _, _, _ = self.models.inspect(self.frame, self.mask)
        self.assertIsNone(bag)

    def test_hand_prevents_clear_result(self):
        self.models.hand.detect.return_value = [self.strip]
        _, _, clear, _, _ = self.models.inspect(self.frame, self.mask)
        self.assertFalse(clear)

    def test_hand_false_positive_far_from_packet_does_not_block_clear(self):
        packet = np.zeros_like(self.mask)
        packet[40:90, 40:80] = 255
        self.models.bag.detect.return_value = [Detection(packet, (40, 40, 80, 90), .9)]
        self.models.hand.detect.return_value = [self.strip]  # Top fixture, far from packet.
        _, _, clear, _, hand = self.models.inspect(self.frame, self.mask)
        self.assertTrue(clear)
        self.assertTrue(np.any(hand))  # Raw hand evidence is kept for transfer tracking.

    def test_missing_bag_uses_matched_body_context_for_strip(self):
        self.models.bag.detect.return_value = []
        bag, strip, _, _, _ = self.models.inspect(self.frame, self.mask, context_mask=self.mask)
        self.assertIsNone(bag)
        self.assertEqual(strip, 'present')

    def test_missing_bag_and_context_is_unknown_not_absent(self):
        self.models.bag.detect.return_value = []
        self.models.strip.detect.return_value = []
        _, strip, _, _, _ = self.models.inspect(self.frame, self.mask)
        self.assertEqual(strip, 'unknown')

    def test_inference_error_propagates_instead_of_counting_absent(self):
        self.models.strip.detect.side_effect = RuntimeError("Synthetic failure")
        with self.assertRaises(RuntimeError):
            self.models.inspect(self.frame, self.mask)


if __name__ == "__main__":
    unittest.main()

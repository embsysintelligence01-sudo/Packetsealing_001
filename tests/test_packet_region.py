import unittest
import numpy as np
from standalone.packet_region import PacketRegion


class PacketRegionTests(unittest.TestCase):
    def setUp(self):
        self.region = PacketRegion()
        self.mask = np.zeros((100, 200), np.uint8)
        self.mask[10:90, 10:90] = 255

    def establish(self):
        for t in (100., 100.4, 100.8):
            valid, changed = self.region.update(self.mask, t, True)
        self.assertTrue(valid)
        self.assertFalse(changed)

    def test_full_bag_mask_establishes_roi_without_opening(self):
        self.establish()
        np.testing.assert_array_equal(self.region.reference, self.mask)

    def test_brief_hand_occlusion_suspends_without_replacing_packet(self):
        self.establish()
        self.assertEqual(self.region.update(None, 101.2, False), (False, False))
        self.assertEqual(self.region.update(None, 104., False), (False, False))
        self.region.update(self.mask, 104.4, True)
        self.assertEqual(self.region.update(self.mask, 104.8, True), (True, False))

    def test_clear_bench_packet_loss_is_reported_once(self):
        self.establish()
        self.assertEqual(self.region.update(None, 104., True), (False, True))
        self.assertEqual(self.region.update(None, 104.4, True), (False, False))

    def test_new_stable_packet_location_is_accepted_but_revokes_old_evidence(self):
        self.establish()
        moved = np.zeros_like(self.mask)
        moved[10:90, 110:190] = 255
        self.region.update(moved, 101.2, True)
        self.region.update(moved, 101.6, True)
        self.assertEqual(self.region.update(moved, 102., True), (True, True))

    def test_folding_the_bag_within_its_footprint_does_not_revoke_inside(self):
        self.establish()
        folded = np.zeros_like(self.mask)
        folded[45:90, 10:90] = 255
        for t in (101.2, 101.6, 102.):
            valid, changed = self.region.update(folded, t, True)
        self.assertTrue(valid)
        self.assertFalse(changed)

    def test_stationary_measures_packet_motion_independently_of_hands(self):
        self.establish()
        self.assertFalse(self.region.stationary)
        self.region.update(self.mask, 101.2, True)
        self.region.update(self.mask, 101.6, True)
        self.assertTrue(self.region.stationary)
        self.region.update(self.mask, 102., False)
        self.assertTrue(self.region.stationary)

    def test_cumulative_packet_motion_cannot_count_as_stationary(self):
        self.establish()
        for i in range(1, 8):
            shifted = np.roll(self.mask, 2 * i, axis=1)
            self.region.update(shifted, 100.8 + i * .4, True)
            self.assertFalse(self.region.stationary)


if __name__ == "__main__":
    unittest.main()

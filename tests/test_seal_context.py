import unittest
import cv2
import numpy as np
from standalone.seal_context import SealContext
from standalone.models import Detection
from standalone.flow import PackingFlow, INSIDE


class SealContextTests(unittest.TestCase):
    def scene(self):
        image = np.full((480, 640, 3), 180, np.uint8)
        rng = np.random.default_rng(44)
        image[90:410, 160:430] = rng.integers(20, 240, (320, 270, 3), dtype=np.uint8)
        mask = np.zeros(image.shape[:2], np.uint8)
        mask[90:410, 160:430] = 255
        strip = np.zeros_like(mask)
        strip[110:135, 180:410] = 255
        return image, Detection(mask, (160, 90, 430, 410), .9), strip

    def test_partial_body_supports_strip_context_without_full_packet_detector(self):
        image, bag, strip = self.scene()
        context = SealContext()
        context.arm(image, bag, strip)
        self.assertTrue(context.ready)
        partial = image.copy()
        partial[300:420, 150:440] = 180  # Bottom third is hidden, strip area visible.
        self.assertIsNotNone(context.locate(partial))
        self.assertTrue(context.strip_view_ok)

    def test_empty_bed_cannot_prove_strip_removal(self):
        image, bag, strip = self.scene()
        context = SealContext()
        context.arm(image, bag, strip)
        self.assertIsNone(context.locate(np.full_like(image, 180)))
        self.assertFalse(context.strip_view_ok)

    def test_visible_body_but_strip_outside_camera_does_not_prove_absence(self):
        image, bag, strip = self.scene()
        context = SealContext()
        context.arm(image, bag, strip)
        moved = cv2.warpAffine(image, np.float32([[1, 0, 0], [0, 1, -170]]), (640, 480))
        self.assertIsNotNone(context.locate(moved))
        self.assertFalse(context.strip_view_ok)

    def test_arming_needs_a_verified_strip_region(self):
        image, bag, strip = self.scene()
        context = SealContext()
        context.arm(image, bag, np.zeros_like(strip))
        self.assertFalse(context.ready)


class StripStageTests(unittest.TestCase):
    def setUp(self):
        self.flow = PackingFlow([1])
        self.flow.jewels[1].status = INSIDE
        self.t = 100.
        for _ in range(3):
            self.step()
        self.assertTrue(self.flow.strip_removal_requested)

    def step(self, **kwargs):
        self.t += .4
        args = dict(packet_ok=True, packet_stationary=True, hands_clear=True,
                    strip='present', closure='model_missing')
        args.update(kwargs)
        self.flow.update(self.t, {}, **args)

    def test_full_packet_dropout_during_removal_does_not_restart_stage(self):
        baseline = self.flow.bed_verified_at
        self.flow.pause_packet()
        for _ in range(6):
            self.step(packet_ok=False, packet_stationary=False, packet_context_ok=True, strip='absent')
        self.assertTrue(self.flow.strip_removed)
        self.assertEqual(self.flow.bed_verified_at, baseline)
        removed = self.flow.strip_removed_at
        for _ in range(8):
            self.step(packet_ok=False, packet_context_ok=False, strip='uncertain', hands_clear=False)
        self.assertTrue(self.flow.strip_removed)
        self.assertEqual(self.flow.strip_removed_at, removed)
        self.assertTrue(self.flow.strip_removal_requested)
        self.assertIn('retained', self.flow.message)
        self.assertTrue(self.flow.sealed)  # Verified removal infers sealing without closure weights.

    def test_absence_without_any_packet_context_is_not_removal(self):
        for _ in range(12):
            self.step(packet_ok=False, packet_context_ok=False, strip='absent')
        self.assertFalse(self.flow.strip_removed)
        self.assertTrue(self.flow.strip_removal_requested)

    def test_hand_covered_strip_does_not_count_as_removed(self):
        for _ in range(8):
            self.step(packet_ok=False, packet_context_ok=True, strip='absent', hands_clear=False)
        self.assertFalse(self.flow.strip_removed)

    def test_strip_zone_must_be_in_view_but_full_bag_is_not_required(self):
        for _ in range(8):
            self.step(packet_ok=False, packet_context_ok=True, strip='absent', strip_view_ok=False)
        self.assertFalse(self.flow.strip_removed)
        for _ in range(5):
            self.step(packet_ok=False, packet_context_ok=True, strip='absent', strip_view_ok=True)
        self.assertTrue(self.flow.strip_removed)

"""Rotated label geometry, consensus and asynchronous post-sealing behavior."""
from concurrent.futures import Future
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from ultralytics.data.augment import LetterBox

from standalone.packet_ocr import NumberConsensus, NumberDetector, PacketOCR


class OCRTests(unittest.TestCase):
    def test_three_matching_fresh_confident_reads_required(self):
        c = NumberConsensus()
        self.assertFalse(c.observe('pc123456', .98, 100.))
        self.assertFalse(c.observe('PC123456', .97, 100.6))
        self.assertFalse(c.observe('PC123456', .97, 100.6))  # Same frame cannot vote twice.
        self.assertTrue(c.observe('PC123456', .96, 101.2))
        self.assertEqual(c.text, 'PC123456')
        self.assertEqual(c.confidence, .96)
        self.assertTrue(c.observe('', 0., 102.))  # Retain the final number.

    def test_disagreement_low_confidence_missing_text_and_long_gap_reset_votes(self):
        for bad, score in [('PC654321', .99), ('PC123456', .5), ('', 0.), ('PACKET', .99)]:
            c = NumberConsensus()
            c.observe('PC123456', .99, 100.)
            c.observe('PC123456', .99, 100.6)
            self.assertFalse(c.observe(bad, score, 101.2))
            self.assertFalse(c.observe('PC123456', .99, 101.8))
        c = NumberConsensus()
        c.observe('PC123456', .99, 100.)
        c.observe('PC123456', .99, 100.6)
        self.assertFalse(c.observe('PC123456', .99, 120.))

    def detector(self, boxes):
        detector = NumberDetector.__new__(NumberDetector)
        detector.session = Mock()
        detector.input = 'images'
        detector.letterbox = LetterBox((640, 640), auto=False)
        raw = np.zeros((1, 6, 8400), np.float32)
        for i, box in enumerate(boxes):
            raw[0, :, i] = box
        detector.session.run.return_value = [raw]
        return detector

    def test_crop_undoes_letterbox_and_does_not_mirror_text(self):
        # 400x200 source letterboxed to 640x640: gain 1.6, top padding 160.
        frame = np.full((200, 400, 3), 255, np.uint8)
        frame[80:100, 100:125] = (0, 0, 255)  # Distinctive source top-left mark.
        detector = self.detector([[320, 320, 320, 64, .99, 0.]])
        crop, polygon = detector.crop(frame, [(0, 0), (399, 0), (399, 199), (0, 199)])
        self.assertEqual(crop.shape[:2], (40, 200))
        np.testing.assert_allclose(polygon[0], [100, 80])
        self.assertGreater(float(crop[:15, :15, 2].mean()), 240)
        self.assertLess(float(crop[:15, :15, 0].mean()), 20)

    def test_rotated_box_is_rectified_and_ambiguous_labels_are_rejected(self):
        frame = np.full((640, 640, 3), 255, np.uint8)
        roi = [(0, 0), (639, 0), (639, 639), (0, 639)]
        detector = self.detector([[320, 320, 220, 60, .99, .5]])
        crop, polygon = detector.crop(frame, roi)
        self.assertEqual(crop.shape[:2], (60, 220))
        detector = self.detector([[150, 150, 150, 40, .99, 0.], [450, 450, 150, 40, .99, 0.]])
        self.assertIsNone(detector.crop(frame, roi)[0])

    def test_missing_or_clipped_label_cannot_generate_crop(self):
        frame = np.zeros((640, 640, 3), np.uint8)
        roi = [(0, 0), (639, 0), (639, 639), (0, 639)]
        for boxes in ([], [[10, 10, 200, 50, .99, 0.]]):
            self.assertIsNone(self.detector(boxes).crop(frame, roi)[0])

    def test_async_error_is_explicit_and_shutdown_does_not_wait_for_camera(self):
        with tempfile.TemporaryDirectory() as root:
            service = PacketOCR(Path(root))
            service.future = Future()
            service.future.set_exception(RuntimeError('OCR environment missing'))
            state, accepted = service.update(np.zeros((10, 10, 3), np.uint8), [], 100.)
            self.assertEqual(state['status'], 'error')
            self.assertIn('Finish / Save', state['message'])
            self.assertIsNone(accepted)
            service.close()


if __name__ == '__main__':
    unittest.main()

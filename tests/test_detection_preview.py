import unittest
from types import SimpleNamespace

import cv2
import numpy as np
import packet_app
from standalone.preview import DetectionPreview
from standalone.app import PacketWorker
from standalone.flow import PackingFlow, PICKED


class DetectionPreviewTests(unittest.TestCase):
    def image(self):
        image = np.zeros((120, 180, 3), np.uint8)
        cv2.rectangle(image, (50, 50), (100, 90), (0, 255, 0), 2)
        return image

    def test_overlay_does_not_blink_during_slow_model_update(self):
        preview = DetectionPreview()
        preview.publish(self.image(), 100.)
        frame = preview.render(np.zeros_like(self.image()), 101.2)
        self.assertTrue(frame[50, 70].any())
        self.assertTupleEqual(tuple(frame[50, 70]), (0, 170, 255))
        self.assertFalse(preview.render(np.zeros_like(frame), 102.6).any())

    def test_one_missed_detection_holds_outline_but_not_evidence(self):
        preview = DetectionPreview()
        preview.publish(self.image(), 100., key='bag', hold_empty=True)
        preview.publish(np.zeros_like(self.image()), 100.2, key='bag', hold_empty=True)
        self.assertTrue(preview.render(np.zeros_like(self.image()), 100.4)[50, 70].any())
        # Repeated misses do not refresh the original detection timestamp.
        preview.publish(np.zeros_like(self.image()), 100.5, key='bag', hold_empty=True)
        self.assertFalse(preview.render(np.zeros_like(self.image()), 100.7).any())

    def test_pickup_clears_old_gold_outline_immediately(self):
        worker = PacketWorker(0, [])
        worker.gold_overlay.publish(self.image(), 100.)
        worker.flow = PackingFlow([1])
        worker.flow.jewels[1].status = PICKED
        worker.tracker = SimpleNamespace(mode='LOCKED', detections=[], tracks=[SimpleNamespace(id=1)])
        worker._publish_gold(self.image(), 100.1)
        self.assertFalse(worker.gold_overlay.render(np.zeros_like(self.image()), 100.1).any())

    def test_separate_layers_do_not_erase_each_other_and_reset_clears(self):
        preview = DetectionPreview()
        preview.publish(self.image(), 100., key='bag')
        preview.publish(np.zeros_like(self.image()), 100.1, key='strip')
        self.assertTrue(preview.render(np.zeros_like(self.image()), 100.2).any())
        preview.clear()
        self.assertFalse(preview.render(np.zeros_like(self.image()), 100.3).any())

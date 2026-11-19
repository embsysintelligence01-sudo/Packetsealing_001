import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import numpy as np
from standalone.flow import PackingFlow
from standalone.models import PacketModels, Detection
from standalone.app import PacketWorker
from standalone.seal_context import SealContext


class SealingDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.flow = PackingFlow([1])
        self.step(10., 'present')

    def step(self, t, strip='absent', **changes):
        args = dict(packet_ok=True, packet_context_ok=True, strip_view_ok=True, hands_clear=True, strip=strip)
        args.update(changes)
        self.flow.update(t, {}, **args)

    def test_short_uncertainty_pauses_without_counting_time_or_hits(self):
        self.step(10.2)
        self.step(10.5)
        self.step(10.7, 'uncertain')
        self.assertEqual(2, self.flow.strip_absent_hits)
        self.assertTrue(self.flow.sealing_diagnostic['uncertainty_paused'])
        self.step(10.9)
        self.assertAlmostEqual(.3, self.flow.strip_clear_seconds)
        self.step(11.2)
        self.step(11.5)
        self.assertFalse(self.flow.sealed)  # Five observations, but only .9 s of clear intervals.
        self.step(11.8)
        self.assertTrue(self.flow.sealed)

    def test_prolonged_uncertainty_cannot_retain_progress(self):
        self.step(10.2)
        self.step(10.5)
        self.step(11.4, 'uncertain')
        self.assertEqual(0, self.flow.strip_absent_hits)
        self.assertEqual(0., self.flow.strip_clear_seconds)

    def test_each_blocker_is_visible_and_resets_progress(self):
        cases = [('packet_not_detected', dict(packet_context_ok=False)),
                 ('strip_area_not_visible', dict(strip_view_ok=False)),
                 ('hand_covering_strip_area', dict(hands_clear=False)),
                 ('strip_still_detected', dict(strip='present'))]
        t = 11.
        for reason, changes in cases:
            self.step(t)
            self.step(t+.2, **changes)
            self.assertEqual(reason, self.flow.sealing_diagnostic['blocker'])
            self.assertEqual(0, self.flow.strip_absent_hits)
            self.assertFalse(self.flow.sealed)
            t += 1

    def test_observation_gap_is_recorded_and_resets_clear_time(self):
        self.step(10.2)
        self.step(10.6)
        self.step(14.)
        self.assertTrue(self.flow.sealing_diagnostic['observation_gap_reset'])
        self.assertEqual(1, self.flow.strip_absent_hits)
        self.assertEqual(0., self.flow.strip_clear_seconds)

    def models(self):
        models = PacketModels.__new__(PacketModels)
        mask = np.full((100, 100), 255, np.uint8)
        zone = np.zeros_like(mask)
        zone[5:15, 5:95] = 255
        loose = np.zeros_like(mask)
        loose[70:80, 5:95] = 255
        models.bag = Mock(detect=Mock(return_value=[Detection(mask, (0, 0, 100, 100), .9)]))
        models.strip = Mock(detect=Mock(return_value=[Detection(loose, (5, 70, 95, 80), .9)]))
        models.hand = Mock(detect=Mock(return_value=[]))
        models.filter_bag = Mock(return_value=.9)
        models.filter_strip = Mock(return_value=.9)
        return models, mask, zone, loose

    def test_detached_strip_away_from_attachment_zone_does_not_block(self):
        models, mask, zone, loose = self.models()
        result = models.inspect(np.zeros((100, 100, 3), np.uint8), mask, inspection_zone=zone)
        self.assertEqual('absent', result[1])
        self.assertEqual('outside_attachment_area', models.last_diagnostic['strip_candidates'][0]['decision'])
        models.strip.detect.return_value = [Detection(zone, (5, 5, 95, 15), .9)]
        self.assertEqual('present', models.inspect(np.zeros((100, 100, 3), np.uint8), mask, inspection_zone=zone)[1])

    def test_hand_away_from_attachment_area_does_not_block_but_nearby_hand_does(self):
        models, mask, zone, loose = self.models()
        frame = np.zeros((100, 100, 3), np.uint8)
        models.hand.detect.return_value = [Detection(loose, (5, 70, 95, 80), .9)]
        self.assertTrue(models.inspect(frame, mask, inspection_zone=zone)[2])
        models.hand.detect.return_value = [Detection(zone, (5, 5, 95, 15), .9)]
        self.assertFalse(models.inspect(frame, mask, inspection_zone=zone)[2])

    def test_log_keeps_source_time_and_only_one_scene_per_blocker(self):
        worker = PacketWorker(0, [])
        worker.flow = self.flow
        with tempfile.TemporaryDirectory() as tmp:
            worker.directory = Path(tmp)
            frame = np.zeros((100, 100, 3), np.uint8)
            for t in (12., 12.5):
                worker._log_sealing_diagnostic(frame, t, dict(timestamp=t, blocker='hand_covering_strip_area'))
            rows = [json.loads(row) for row in (Path(tmp)/'sealing_diagnostics.jsonl').read_text().splitlines()]
            self.assertEqual([12., 12.5], [r['source_timestamp'] for r in rows])
            self.assertEqual(1, len(list((Path(tmp)/'sealing_debug').glob('*.jpg'))))
            self.assertIn('scene', rows[0])
            self.assertNotIn('scene', rows[1])

    def test_slow_inference_resets_instead_of_adding_missing_clear_time(self):
        worker = PacketWorker(0, [])
        worker.flow = self.flow
        self.step(10.2)
        with tempfile.TemporaryDirectory() as tmp:
            worker.directory = Path(tmp)
            worker._log_sealing_diagnostic(None, 11., dict(timestamp=11., blocker='inference_too_slow'))
        self.assertEqual(0, self.flow.strip_absent_hits)
        self.assertEqual('inference_too_slow', self.flow.sealing_diagnostic['blocker'])

    def test_old_reference_loss_does_not_block_current_packet_detection(self):
        self.step(10.2)
        self.step(10.4, reference_lost=True)
        self.assertEqual('checking_removal', self.flow.sealing_diagnostic['blocker'])
        self.assertEqual(2, self.flow.strip_absent_hits)

    def test_attachment_zone_maps_back_to_large_camera_frames(self):
        models, _, _, _ = self.models()
        models.bag.detect.side_effect = lambda frame: [Detection(np.full(frame.shape[:2], 255, np.uint8),
            (0, 0, frame.shape[1], frame.shape[0]), .9)]
        models.strip.detect.return_value = []
        frame = np.zeros((720, 1280, 3), np.uint8)
        roi = np.full(frame.shape[:2], 255, np.uint8)
        zone = np.zeros_like(roi)
        zone[50:100, 200:1100] = 255
        bag, strip, clear, strips, hands = models.inspect(frame, roi, inspection_zone=zone)
        self.assertEqual((720, 1280), bag.mask.shape)
        self.assertEqual((720, 1280), strips.shape)
        self.assertEqual((720, 1280), hands.shape)
        self.assertEqual('absent', strip)
        self.assertTrue(clear)
        self.assertEqual('tracked_attachment_area', models.last_diagnostic['inspection_scope'])

import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
import numpy as np
from standalone.motion_classification import CUES, MotionSequences


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.collector = MotionSequences(self.root, window=4, session='recording-one')
        self.stone = (60, 60, 100, 100)

    def update(self, t, x=75, visible=True, stone=True, stable=True):
        frame = np.full((160, 160, 3), int(t*20), np.uint8)
        box = (x, 70, x+10, 80)
        mask = np.zeros((160,160), np.uint8)
        mask[70:80, x:x+10] = 255
        det = SimpleNamespace(box=box, mask=mask)
        return self.collector.update(frame, [det] if visible else [],
            {1:dict(box=box, seen=t if visible else -1, visibility='VISIBLE')}, [],
            self.stone if stone else None, (0,0,160,160), t, stable)

    def test_ordered_views_cues_and_manual_label(self):
        for i in range(4):
            result = self.update(1.+i*.2, x=75+(i%2)*5)
        self.assertEqual(result[1]['label'], 'UNTRAINED')
        path = next(self.root.glob('*.npz'))
        with np.load(path) as data:
            self.assertEqual(data['frames'].shape, (4,2,224,224,3))
            self.assertEqual(data['cues'].shape, (4,len(CUES)))
            self.assertTrue(np.all(np.diff(data['times']) > 0))
            self.assertTrue(np.all(data['cues'][:,2] == 1))
            self.assertGreater(data['cues'][:,7].sum(), 0)
            self.assertGreater(data['cues'][-1,8], 0)
        metadata = json.loads(path.with_suffix('.json').read_text())
        self.assertIsNone(metadata['label'])
        self.assertEqual(metadata['session'], 'recording-one')

    def test_missing_detection_is_explicit_not_invented_motion(self):
        for i in range(4):
            self.update(1.+i*.2, visible=i != 1)
        with np.load(next(self.root.glob('*.npz'))) as data:
            self.assertEqual(data['cues'][1,0], 0)
            self.assertEqual(data['frames'][1,1].sum(), 0)
            self.assertEqual(data['cues'][2,5:7].sum(), 0)

    def test_gap_does_not_join_different_actions(self):
        self.update(1.); self.update(1.2); self.update(3.); self.update(3.2)
        self.assertFalse(list(self.root.glob('*.npz')))

    def test_missing_stone_or_unstable_scene_resets(self):
        self.update(1.); self.update(1.2)
        self.update(1.4, stone=False)
        self.update(1.6); self.update(1.8)
        self.assertFalse(list(self.root.glob('*.npz')))
        self.update(2., stable=False)
        self.assertFalse(self.collector.tracks)

    def test_stone_jump_resets(self):
        self.update(1.); self.update(1.2)
        self.stone = (100, 60, 140, 100)
        self.update(1.4); self.update(1.6)
        self.assertFalse(list(self.root.glob('*.npz')))

    def test_untrained_cannot_claim_ok(self):
        from standalone.motion_classification import train
        with self.assertRaisesRegex(ValueError, 'separate recording'):
            train(self.root, self.root/'model.pt', 1)

    def test_two_jewel_ids_are_not_combined(self):
        self.update(1.); self.update(1.2)
        frame = np.zeros((160,160,3), np.uint8)
        for t in (1.4, 1.6):
            self.collector.update(frame, [], {2:dict(box=(10,10,20,20),seen=-1)}, [],
                                  self.stone, (0,0,160,160), t)
        self.assertFalse(list(self.root.glob('*.npz')))
        self.assertNotIn(1, self.collector.tracks)

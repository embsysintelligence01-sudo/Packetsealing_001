import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import numpy as np

from packet_mouth_lab.mouth_training import MouthDataset
from standalone.models import Detection, PacketBagDetector


class PacketCollectionTests(unittest.TestCase):
    def test_packet_only_detector_filters_and_clips(self):
        detector=PacketBagDetector.__new__(PacketBagDetector)
        frame=np.full((80,100,3),200,np.uint8)
        mask=np.full((80,100),255,np.uint8)
        detector.bag=SimpleNamespace(detect=Mock(return_value=[
            Detection(mask,(-5,-3,110,90),.9), Detection(mask,(10,10,60,60),.8)]))
        detector.filter_bag=Mock(side_effect=[.95,.2])
        result=detector.detect(frame)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0].box,(0,0,100,80))
        json.dumps(list(result[0].box))

    def test_collection_preserves_context_and_label_link(self):
        with tempfile.TemporaryDirectory() as folder:
            dataset=MouthDataset(folder)
            frame=np.full((80,100,3),200,np.uint8)
            mask=np.zeros((80,100),np.uint8)
            mask[10:70,20:90]=255
            detection=Detection(mask,(20,10,90,70),.95)
            source={'name':'example.mp4','kind':'video','time':2.5}
            context=dataset.add_packet(frame,detection,'recording_1',source,(25,15,40,25),'OPEN_CAMERA')
            path=dataset.add(frame[15:40,25:65],'OPEN_CAMERA','recording_1',context=context)
            record=Path(folder)/context
            meta=json.loads((record/'metadata.json').read_text())
            self.assertEqual(meta['source'],source)
            self.assertEqual(meta['packet_box_xyxy'],[20,10,90,70])
            self.assertEqual(meta['mouth_roi_xywh'],[25,15,40,25])
            self.assertEqual(meta['mouth_label'],'OPEN_CAMERA')
            self.assertEqual(cv2.imread(str(record/'packet.jpg')).shape[:2],(60,70))
            np.testing.assert_array_equal(cv2.imread(str(record/'packet_mask.png'),0),mask)
            self.assertEqual(json.loads(path.with_suffix('.json').read_text())['packet_context'],context)
            self.assertEqual(dataset.counts(),{'OPEN_CAMERA':1})
            unlabelled=dataset.add_packet(frame,detection,'recording_1',source)
            meta=json.loads((Path(folder)/unlabelled/'metadata.json').read_text())
            self.assertIsNone(meta['mouth_label'])
            self.assertIsNone(meta['mouth_roi_xywh'])
            with self.assertRaises(ValueError):
                dataset.add_packet(frame,None,'recording_1',source)


if __name__=='__main__':
    unittest.main()

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import cv2

from standalone.flow import PackingFlow, INSIDE, PICKED
from standalone.packet_opening import PacketOpening, opening_description


class PacketOpeningTests(unittest.TestCase):
    def observer(self):
        observer=PacketOpening('unused.joblib')
        observer.loaded=True
        observer.classifier=SimpleNamespace(predict=Mock(return_value=('OPEN_SIDE',.95,{'OPEN_SIDE':.95})))
        observer.model_info={'status':'ready','validation':{'accuracy':.6962}}
        return observer

    def test_preprocessing_consensus_dropouts_and_transfer_ownership(self):
        frame=np.full((1440,2560,3),180,np.uint8)
        bag=SimpleNamespace(box=(800,400,2000,1200))
        observer=self.observer()
        flow=PackingFlow([1,2]);flow.jewels[1].started=1.;flow.jewels[1].status=PICKED
        for t in (1.,1.2,1.5):
            observation=observer.update(frame,bag,t)
            observer.record(flow,t)
        self.assertEqual(observer.classifier.predict.call_args.args[0].shape[:2],(300,450))
        self.assertTrue(observation['consistent'])
        self.assertEqual(flow.jewels[1].packet_opening_observations[0]['label'],'OPEN_SIDE')
        self.assertEqual(flow.jewels[2].packet_opening_observations,[])
        self.assertEqual(flow.jewels[1].status,PICKED)
        self.assertFalse(flow.sealed)
        flow.jewels[1].ended=1.7
        flow.jewels[1].status=INSIDE
        observer.update(frame,bag,1.7);observer.record(flow,1.7)
        self.assertTrue(flow.jewels[1].packet_opening_at_placement['consistent'])
        previous=list(flow.jewels[1].packet_opening_observations)
        observer.update(frame,None,2.);observer.record(flow,2.)
        self.assertIsNone(observer.current['label'])
        self.assertFalse(observer.current['consistent'])
        self.assertEqual(flow.jewels[1].packet_opening_observations,previous)
        self.assertFalse(observer.update(frame,bag,2.1)['consistent'])

    def test_missing_model_or_prediction_failure_does_not_abort(self):
        frame=np.zeros((100,150,3),np.uint8);bag=SimpleNamespace(box=(10,10,100,90))
        with tempfile.TemporaryDirectory() as folder:
            observer=PacketOpening(Path(folder)/'missing.joblib')
            self.assertEqual(observer.update(frame,bag,1.)['status'],'model_missing')
        observer=self.observer()
        observer.classifier.predict.side_effect=ValueError('bad features')
        self.assertEqual(observer.update(frame,bag,1.)['status'],'classifier_error')
        self.assertIsNone(observer.current['label'])

    def test_low_confidence_never_becomes_confirmed_and_wording_requires_placement(self):
        frame=np.zeros((100,150,3),np.uint8);bag=SimpleNamespace(box=(10,10,100,90))
        observer=self.observer()
        observer.classifier.predict.return_value=('OPEN_CAMERA',.5,{'OPEN_CAMERA':.5})
        for t in (1.,1.2,1.5,1.8):
            self.assertFalse(observer.update(frame,bag,t)['consistent'])
        record={'id':1,'status':PICKED,'packet_opening_observations':[{'label':'OPEN_SIDE'}]}
        self.assertIn('apparent opening',opening_description(record))
        self.assertNotIn('placed inside',opening_description(record))
        record['status']=INSIDE
        self.assertIn('inferred placed inside from transfer evidence',opening_description(record))
        record['packet_opening_observations']=[]
        self.assertIn('could not be confirmed',opening_description(record))

    def test_single_frame_is_saved_and_reported_without_temporal_confirmation(self):
        observer=self.observer()
        observer.classifier.predict.return_value=('OPEN_CAMERA',.919,{'OPEN_CAMERA':.919})
        frame=np.full((200,300,3),180,np.uint8)
        frame[30:150,50:250]=(20,80,130)
        original=frame.copy()
        bag=SimpleNamespace(box=(50,30,250,150))
        flow=PackingFlow([1]);jewel=flow.jewels[1]
        jewel.started=jewel.ended=1.;jewel.status=INSIDE
        observer.update(frame,bag,1.);observer.record(flow,1.)
        self.assertFalse(jewel.packet_opening_at_placement['consistent'])
        with tempfile.TemporaryDirectory() as directory:
            observer.save_placement_frame(frame,jewel,1.,Path(directory))
            np.testing.assert_array_equal(frame,original)
            saved=cv2.imread(str(Path(directory)/jewel.packet_opening_at_placement['frame']))
            self.assertEqual(saved.shape,frame.shape)
            crop=cv2.imread(str(Path(directory)/jewel.packet_opening_at_placement['packet_crop']))
            self.assertEqual(crop.shape[:2],(120,200))
        record={'id':1,'status':INSIDE,'end':1.,'packet_opening_at_placement':jewel.packet_opening_at_placement}
        description=opening_description(record)
        self.assertIn('camera-facing packet opening',description)
        self.assertIn('91.9%',description)
        self.assertIn('temporal consistency not established',description)
        # A later missing detection cannot overwrite the latched placement image/result.
        observer.update(frame,None,2.);observer.record(flow,2.)
        self.assertEqual(jewel.packet_opening_at_placement['timestamp'],1.)
        self.assertEqual(jewel.packet_opening_at_placement['label'],'OPEN_CAMERA')

    def test_stale_or_missing_detection_cannot_substitute_a_placement_prediction(self):
        observer=self.observer()
        frame=np.zeros((100,150,3),np.uint8)
        bag=SimpleNamespace(box=(10,10,100,90))
        flow=PackingFlow([1]);jewel=flow.jewels[1]
        jewel.started=1.;jewel.ended=2.;jewel.status=INSIDE
        observer.update(frame,bag,1.)
        observer.record(flow,2.)
        snapshot=jewel.packet_opening_at_placement
        self.assertIsNone(snapshot['label'])
        self.assertEqual(snapshot['status'],'observation_unavailable')
        record={'id':1,'status':INSIDE,'end':2.,'packet_opening_at_placement':snapshot}
        self.assertIn('classification unavailable',opening_description(record))
        jewel.packet_opening_at_placement=None
        observer.update(frame,None,2.);observer.record(flow,2.)
        record['packet_opening_at_placement']=jewel.packet_opening_at_placement
        self.assertIn('no accepted packet detection',opening_description(record))

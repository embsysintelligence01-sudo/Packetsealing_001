import json
import os
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
import cv2
import numpy as np
from PyQt5 import QtWidgets
from packet_mouth_lab import app as lab


class BBoxUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_camera_requests_saved_size_and_reports_actual_frames(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(lab,'ROOT',Path(directory)):
            window=lab.MouthWindow()
            window.use_packet.setChecked(False)
            cap=Mock()
            cap.isOpened.return_value=True
            cap.read.return_value=(True,np.zeros((1440,2560,3),np.uint8))
            with patch.object(lab.cv2,'VideoCapture',return_value=cap), patch.object(lab,'saved_camera_resolution',return_value=(2560,1440)):
                window._open_capture(0,'camera')
                cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH,2560)
                cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT,1440)
                self.assertEqual(window.capture_size,(2560,1440))
                self.assertEqual(window.frame.shape[:2],(540,960))
                self.assertIn('Actual capture 2560×1440',window.capture_status.text())
                window._stop()
                cap.set.reset_mock()
                cap.get.return_value=0
                window._open_capture('example.mp4','video')
                cap.set.assert_not_called()
            window.close()

    def test_resolution_reads_saved_value_and_falls_back(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{'JEWEL_TRACKER_DIR':directory}):
            self.assertEqual(lab.saved_camera_resolution(),(2560,1440))
            path=Path(directory)/'camera.json'
            path.write_text(json.dumps({'usb':{'resolution':'1920x1080'}}))
            self.assertEqual(lab.saved_camera_resolution(),(1920,1080))
            path.write_text(json.dumps({'usb':{'resolution':'invalid'}}))
            self.assertEqual(lab.saved_camera_resolution(),(2560,1440))

    def test_collect_and_classify_same_detected_box_without_roi(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(lab,'ROOT',Path(directory)):
            window=lab.MouthWindow()
            self.assertFalse(window.use_classifier.isChecked())
            self.assertTrue(window.use_classifier.isEnabled())
            window.use_classifier.setChecked(True)
            self.assertTrue(window.use_classifier.isChecked())
            self.assertIn('No trained packet-box classifier',window.validation.text())
            self.assertIsNone(window.last_prediction)
            window.use_classifier.setChecked(False)
            window._new_source('image','test.jpg')
            window.frame=np.full((120,200,3),210,np.uint8)
            window.frame[20:100,40:150]=(20,80,150)
            mask=np.zeros((120,200),np.uint8);mask[20:100,40:150]=255
            window.packet_detections=[SimpleNamespace(box=(40,20,150,100),mask=mask,confidence=.95)]
            window._analyze(fresh=False)
            self.assertEqual(window.packet_roi(),(40,20,110,80))
            window.label_box.setCurrentText('OPEN_SIDE')
            window.label_sample()
            samples=list(window.dataset.root.glob('*.json'))
            self.assertEqual(len(samples),1)
            metadata=json.loads(samples[0].read_text())
            self.assertEqual(metadata['label'],'OPEN_SIDE')
            self.assertEqual(metadata['input_region'],'packet_bbox')
            crop=cv2.imread(str(window.dataset.root/metadata['image']))
            self.assertEqual(crop.shape[:2],(80,110))
            context=json.loads((window.dataset.root/metadata['packet_context']/'metadata.json').read_text())
            self.assertEqual(context['packet_label'],'OPEN_SIDE')
            self.assertIsNone(context['mouth_roi_xywh'])
            classifier=SimpleNamespace(ready=True,payload={'input_region':'packet_bbox'},
                                       predict=Mock(return_value=('OPEN_CAMERA',.91,{'OPEN_CAMERA':.91})))
            with patch.object(lab,'MouthClassifier',return_value=classifier), patch.object(lab.cv2,'putText',wraps=cv2.putText) as text:
                window.use_classifier.setEnabled(True)
                window.use_classifier.setChecked(True)
                np.testing.assert_array_equal(classifier.predict.call_args.args[0],window.frame[20:100,40:150])
                self.assertTrue(any('OPEN_CAMERA' in call.args[1] for call in text.call_args_list))
                window.use_classifier.setChecked(False)
                self.assertIsNone(window.last_prediction)
            window.packet_detections=[]
            window._analyze(fresh=True)
            self.assertIsNone(window.packet_roi())
            self.assertIsNone(window.last_prediction)
            self.assertEqual(window.confirmed,'UNCONFIRMED')
            window.close()


if __name__=='__main__':
    unittest.main()

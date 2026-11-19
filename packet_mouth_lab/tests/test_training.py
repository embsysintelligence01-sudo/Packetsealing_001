import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from mouth_training import MouthDataset, MouthClassifier, train_validated


class TrainingTests(unittest.TestCase):
    def test_train_requires_diverse_groups(self):
        with tempfile.TemporaryDirectory() as root:
            data=MouthDataset(Path(root)/'data')
            base=np.full((100,120,3),230,np.uint8)
            data.add(base,'OPEN_CAMERA','one')
            with self.assertRaises(ValueError):
                train_validated(data,Path(root)/'model.joblib')

    def test_train_and_prediction(self):
        with tempfile.TemporaryDirectory() as root:
            data=MouthDataset(Path(root)/'data')
            colors={
                'OPEN_CAMERA':(160,75,20),
                'OPEN_SIDE':(40,140,170),
                'CLOSED_FOLDED':(85,40,130),
                'UNCLEAR':(205,210,220)
            }
            rng=np.random.default_rng(15)
            for session in range(4):
                for label,color in colors.items():
                    for idx in range(4):
                        img=np.full((100,120,3),color,np.uint8)
                        if label=='OPEN_CAMERA':cv2.ellipse(img,(60,50),(44,25),0,0,360,(225,230,230),3)
                        if label=='OPEN_SIDE':cv2.line(img,(15,50),(105,50),(240,245,250),5)
                        if label=='CLOSED_FOLDED':cv2.line(img,(15,35),(105,65),(235,230,225),5)
                        noise=rng.integers(-4,5,img.shape,dtype=np.int16)
                        img=np.uint8(np.clip(img.astype(np.int16)+noise,0,255))
                        data.add(img,label,f'physical_packet_{session}')
            model=Path(root)/'mouth_classifier.joblib'
            report=train_validated(data,model,min_per_class=12)
            self.assertEqual(len(report['labels']),4)
            self.assertGreater(report['accuracy'],.25)
            cls=MouthClassifier(model)
            self.assertTrue(cls.ready)
            label,p,probs=cls.predict(np.full((100,120,3),(205,210,220),np.uint8))
            self.assertIn(label,colors)
            self.assertGreater(p,0)
            self.assertTrue(model.exists())

if __name__=='__main__':
    unittest.main()

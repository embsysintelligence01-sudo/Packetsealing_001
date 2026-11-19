import sys
from pathlib import Path
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from mouth_core import (MouthROITracker, mouth_features, TemporalConsensus,
                        clamp_roi, crop_roi, diagnostic_metrics, visible_gap_candidate)


class CoreTests(unittest.TestCase):
    def test_roi_and_features(self):
        img=np.full((140,240,3),220,np.uint8)
        cv2.rectangle(img,(35,45),(170,95),(15,12,80),-1)
        roi=clamp_roi((30,40,150,70),img.shape)
        self.assertEqual(roi,(30,40,150,70))
        self.assertGreater(len(mouth_features(crop_roi(img,roi))),500)
        self.assertIn('edge_density',diagnostic_metrics(crop_roi(img,roi)))

    def test_track_translation(self):
        rng=np.random.default_rng(7)
        bg=np.full((260,350,3),225,np.uint8)
        # Add textured features with rigid translation.
        for _ in range(90):
            x=int(rng.integers(110,195)); y=int(rng.integers(92,142))
            col=int(rng.integers(20,190))
            cv2.circle(bg,(x,y),2,(col,col,col),-1)
        tr=MouthROITracker()
        state=tr.seed(bg,(97,78,110,80))
        self.assertEqual(state.status,'SEEDED')
        M=np.float32([[1,0,6],[0,1,3]])
        moved=cv2.warpAffine(bg,M,(350,260),borderValue=(225,225,225))
        result=tr.update(moved)
        self.assertEqual(result.status,'TRACKED',result.reason)
        self.assertLess(abs(result.roi[0]-103),5)
        self.assertLess(abs(result.roi[1]-81),5)

    def test_temporal_consensus(self):
        consensus=TemporalConsensus(min_samples=3,min_span=.3,min_prob=.78)
        self.assertEqual(consensus.add('OPEN_CAMERA',.9,1.0),'UNCONFIRMED')
        self.assertEqual(consensus.add('OPEN_CAMERA',.9,1.2),'UNCONFIRMED')
        self.assertEqual(consensus.add('OPEN_CAMERA',.9,1.4),'OPEN_CAMERA_CONSISTENT')
        self.assertEqual(consensus.add('CLOSED_FOLDED',.9,1.6),'UNCONFIRMED')
        self.assertEqual(consensus.add('OPEN_CAMERA',.4,1.8),'UNCONFIRMED')

    def test_gap_candidate_and_ambiguous_views(self):
        image=np.full((120,200,3),220,np.uint8)
        self.assertEqual(visible_gap_candidate(image)['label'],'UNCLEAR')
        cv2.ellipse(image,(100,60),(55,18),20,0,360,(30,30,30),-1)
        result=visible_gap_candidate(image)
        self.assertEqual(result['label'],'GAP_CANDIDATE')
        self.assertGreater(result['contrast'],25)
        self.assertIsNotNone(result['contour'])
        # A narrow seam and a dark region continuing off the crop are ambiguous.
        seam=np.full_like(image,220)
        cv2.line(seam,(25,60),(175,60),(20,20,20),2)
        self.assertEqual(visible_gap_candidate(seam)['label'],'UNCLEAR')
        border=np.full_like(image,220)
        cv2.rectangle(border,(0,25),(80,90),(20,20,20),-1)
        self.assertEqual(visible_gap_candidate(border)['label'],'UNCLEAR')
        self.assertEqual(visible_gap_candidate(None)['label'],'UNCLEAR')

    def test_repeated_rotation_does_not_expand_roi(self):
        rng=np.random.default_rng(9)
        image=np.full((320,320,3),225,np.uint8)
        texture=rng.integers(20,200,(70,100,3),dtype=np.uint8)
        image[125:195,110:210]=texture
        tracker=MouthROITracker()
        tracker.seed(image,(105,120,110,80))
        for angle in [5,-5,5,-5,5,0]:
            transform=cv2.getRotationMatrix2D((160,160),angle,1)
            moved=cv2.warpAffine(image,transform,(320,320),borderValue=(225,225,225))
            state=tracker.update(moved)
            self.assertEqual(state.status,'TRACKED',state.reason)
        self.assertLess(state.roi[2],125)
        self.assertLess(state.roi[3],95)


if __name__ == '__main__':
    unittest.main()

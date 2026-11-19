import unittest
from types import SimpleNamespace
import cv2
import numpy as np
from standalone.gold_mask_tracks import GoldMaskTracks


class GoldMaskTrackTests(unittest.TestCase):
    def setUp(self):
        self.shape=(220,360,3)
        self.observer=GoldMaskTracks()
        self.scene=dict(trusted=True,matrix=[[1,0,0],[0,1,0]])

    def gold(self,box):
        mask=np.zeros(self.shape[:2],np.uint8)
        x1,y1,x2,y2=box;mask[y1:y2,x1:x2]=255
        return SimpleNamespace(box=box,mask=mask)

    def hand(self,x,y=100):
        return SimpleNamespace(points=np.full((21,2),(x,y),np.float32),scale=80.)

    def test_unknown_gold_still_has_mask_centroid(self):
        result=self.observer.update([self.gold((30,80,70,120))],[],{},self.shape,1.)
        self.assertEqual([49.5,99.5],result[0]['cog'])
        self.assertEqual('Gold-T1',result[0]['track'])
        self.assertIsNone(result[0]['jewel_id'])
        self.assertFalse(result[0]['pickup_verified'])

    def test_ring_centroid_preserves_hole(self):
        det=self.gold((30,30,131,131));det.mask[:]=0
        cv2.circle(det.mask,(80,80),40,255,5)
        result=self.observer.update([det],[],{},self.shape,1.)
        np.testing.assert_allclose(result[0]['cog'],[80,80],atol=1.)
        self.assertEqual(0,det.mask[80,80])

    def test_unknown_gold_movement_can_follow_hand(self):
        for i in range(4):
            box=(30+i*10,80,70+i*10,120)
            result=self.observer.update([self.gold(box)],[self.hand(50+i*10)],{},self.shape,1.+i*.25,self.scene)
        self.assertEqual('Gold-T1',result[0]['track'])
        self.assertEqual('GOLD_MOVING_WITH_HAND',result[0]['state'])
        self.assertIsNone(result[0]['jewel_id'])
        self.assertFalse(result[0]['pickup_verified'])

    def test_hand_motion_without_gold_motion_does_not_qualify(self):
        for i in range(4):
            result=self.observer.update([self.gold((30,80,70,120))],[self.hand(50+i*4)],{},self.shape,1.+i*.25,self.scene)
        self.assertNotEqual('GOLD_MOVING_WITH_HAND',result[0]['state'])

    def test_identity_requires_repeated_unique_fresh_matches(self):
        box=(30,80,70,120)
        first=self.observer.update([self.gold(box)],[],{7:dict(box=box,seen=1.)},self.shape,1.)
        self.assertIsNone(first[0]['jewel_id'])
        second=self.observer.update([self.gold(box)],[],{7:dict(box=box,seen=1.25)},self.shape,1.25)
        self.assertEqual(7,second[0]['jewel_id'])
        missing=self.observer.update([self.gold(box)],[],{},self.shape,1.5)
        self.assertIsNone(missing[0]['jewel_id'])
        self.assertEqual('Gold-T1',missing[0]['track'])

    def test_merge_does_not_inherit_an_individual_id(self):
        a,b=(30,80,70,120),(90,80,130,120)
        self.observer.update([self.gold(a),self.gold(b)],[],{},self.shape,1.)
        merged=self.observer.update([self.gold((30,80,130,120))],[],{},self.shape,1.25)
        self.assertEqual('GOLD_MASK_AMBIGUOUS',merged[0]['state'])
        self.assertIsNone(merged[0]['jewel_id'])

    def test_gap_creates_new_temporary_track_and_resets_motion(self):
        det=self.gold((30,80,70,120))
        self.observer.update([det],[],{},self.shape,1.)
        result=self.observer.update([det],[],{},self.shape,3.)
        self.assertEqual('Gold-T2',result[0]['track'])
        self.assertEqual(0,result[0]['motion_hits'])

    def test_camera_translation_is_not_independent_gold_motion(self):
        for i in range(4):
            scene=dict(trusted=True,matrix=[[1,0,i*10],[0,1,0]])
            result=self.observer.update([self.gold((30+i*10,80,70+i*10,120))],[self.hand(50+i*10)],{},self.shape,1.+i*.25,scene)
        self.assertNotEqual('GOLD_MOVING_WITH_HAND',result[0]['state'])

    def test_ambiguous_hands_do_not_produce_motion(self):
        for i in range(4):
            result=self.observer.update([self.gold((30+i*10,80,70+i*10,120))],[self.hand(50+i*10),self.hand(55+i*10)],{},self.shape,1.+i*.25,self.scene)
        self.assertEqual('GOLD_MASK_AMBIGUOUS',result[0]['state'])
        self.assertIsNone(result[0]['jewel_id'])

    def test_missing_mask_has_no_invented_centroid(self):
        det=self.gold((30,80,70,120));det.mask[:]=0
        self.assertEqual([],self.observer.update([det],[],{},self.shape,1.))

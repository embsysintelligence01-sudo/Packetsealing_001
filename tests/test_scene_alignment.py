import unittest
import cv2
import numpy as np
from standalone.scene_alignment import SceneAlignment, reference_point


class SceneAlignmentTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(42)
        self.reference=cv2.GaussianBlur(rng.integers(0,256,(240,320),dtype=np.uint8),(3,3),0)
        self.mask=np.full(self.reference.shape,255,np.uint8)
        self.clear=np.zeros_like(self.mask)
        self.alignment=SceneAlignment(self.reference,self.mask)

    def test_static_scene_with_moving_occluded_region_stays_stable(self):
        current=self.reference.copy();current[65:190,85:235]=0
        blocked=self.clear.copy();blocked[60:195,80:240]=255
        status=self.alignment.measure(current,blocked)
        self.assertEqual('STABLE',status['state'])
        self.assertTrue(status['trusted'])

    def test_translation_is_measured_and_compensated(self):
        matrix=np.array([[1.,0.,12.],[0.,1.,7.]])
        current=cv2.warpAffine(self.reference,matrix,(320,240))
        status=self.alignment.measure(current,self.clear)
        self.assertEqual('CAMERA_MOVED',status['state'])
        self.assertTrue(status['trusted'])
        np.testing.assert_allclose(reference_point([112,107],status),[100,100],atol=1.)

    def test_rotation_is_compensated(self):
        matrix=cv2.getRotationMatrix2D((160,120),5.,1.)
        current=cv2.warpAffine(self.reference,matrix,(320,240))
        status=self.alignment.measure(current,self.clear)
        self.assertEqual('CAMERA_MOVED',status['state'])
        self.assertTrue(status['trusted'])
        point=matrix[:,:2]@np.array([110.,95.])+matrix[:,2]
        np.testing.assert_allclose(reference_point(point,status),[110,95],atol=2.)

    def test_featureless_background_is_unknown_not_camera_motion(self):
        blank=np.full_like(self.reference,215)
        alignment=SceneAlignment(blank,self.mask)
        result=alignment.measure(blank,self.clear)
        self.assertEqual('INSUFFICIENT_EVIDENCE',result['state'])
        self.assertEqual('too_few_background_points',result['reason'])
        self.assertFalse(result['trusted'])
        self.assertIsNone(reference_point([20,30],result))

    def test_all_background_occluded_cannot_become_stable(self):
        result=self.alignment.measure(self.reference,self.mask)
        self.assertFalse(result['trusted'])
        self.assertEqual('INSUFFICIENT_EVIDENCE',result['state'])

    def test_descriptor_reference_recovers_when_optical_flow_unavailable(self):
        self.alignment.points=np.empty((0,1,2),np.float32)
        result=self.alignment.measure(self.reference,self.clear)
        self.assertEqual('STABLE',result['state'])
        self.assertEqual('fixed_background_descriptors',result['source'])
        self.assertEqual('too_few_background_points',result['flow_failure'])

    def test_one_local_moving_cluster_is_not_a_camera_reference(self):
        before=np.array([[100+x,100+y] for x in range(4) for y in range(4)],np.float32)
        result=self.alignment.fit(before,before+10,'test')
        self.assertFalse(result['trusted'])
        self.assertEqual('background_points_not_distributed',result['reason'])

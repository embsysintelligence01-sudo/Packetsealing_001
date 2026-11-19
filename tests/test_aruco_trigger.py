import unittest
import cv2
import numpy as np
from types import SimpleNamespace
from pickup_trigger_test import make_engine,ArucoHomeEngine,HomePositionEngine
from standalone.aruco_reference import ArucoReference
from standalone.scene_alignment import reference_point,scene_area_scale


class ArucoTriggerTests(unittest.TestCase):
    def setUp(self):
        self.frame=np.full((480,640,3),230,np.uint8)
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        marker=cv2.aruco.generateImageMarker(dictionary,23,100)
        self.frame[40:140,50:150]=cv2.cvtColor(marker,cv2.COLOR_GRAY2BGR)

    def acquire(self,reference):
        for _ in range(3):reference.detect(self.frame)
        self.assertTrue(reference.freeze())

    def test_full_frame_top_left_marker_auto_identifies_dictionary_and_id(self):
        reference=ArucoReference();self.acquire(reference)
        self.assertEqual(('DICT_4X4_1000',23),reference.identity)
        status=reference.alignment(self.frame.shape)
        self.assertEqual('STABLE',status['state'])
        self.assertTrue(status['trusted'])

    def test_camera_shift_is_compensated_even_away_from_marker(self):
        reference=ArucoReference();self.acquire(reference)
        shifted=cv2.warpAffine(self.frame,np.float32([[1,0,12],[0,1,9]]),(640,480),borderValue=(230,230,230))
        reference.detect(shifted);status=reference.alignment(self.frame.shape)
        self.assertEqual('CAMERA_MOVED',status['state'])
        np.testing.assert_allclose(reference_point([512,359],status),[500,350],atol=1.)
        self.assertAlmostEqual(1.,scene_area_scale([512,359],status),places=2)

    def test_missing_or_wrong_marker_does_not_reuse_stale_alignment(self):
        reference=ArucoReference();self.acquire(reference)
        blank=np.full_like(self.frame,230)
        reference.detect(blank)
        self.assertIsNone(reference.alignment(blank.shape))
        wrong=self.frame.copy()
        wrong[40:140,50:150]=cv2.cvtColor(cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000),24,100),cv2.COLOR_GRAY2BGR)
        reference.detect(wrong)
        self.assertIsNone(reference.current)

    def test_too_small_marker_rejected(self):
        frame=np.full_like(self.frame,230)
        marker=cv2.aruco.generateImageMarker(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000),23,18)
        frame[40:58,50:68]=cv2.cvtColor(marker,cv2.COLOR_GRAY2BGR)
        reference=ArucoReference()
        for _ in range(4):reference.detect(frame)
        self.assertIsNone(reference.identity)

    def test_third_mode_is_separate_from_home_mode(self):
        third=make_engine('cog',lambda event:None)
        home=make_engine('home',lambda event:None)
        self.assertIsInstance(third,ArucoHomeEngine)
        self.assertIs(type(home),HomePositionEngine)
        self.assertFalse(hasattr(home,'aruco'))
        self.assertTrue(hasattr(third,'home_rubbing'))
        self.assertTrue(hasattr(third,'gold_masks'))

    def test_no_marker_means_no_initial_home_assignment(self):
        engine=make_engine('cog',lambda event:None)
        frame=np.full_like(self.frame,230);roi=np.full(frame.shape[:2],255,np.uint8)
        tracks=[SimpleNamespace(id=1,det_seen=0.,det_box=(300,300,350,350))]
        for t in (0.,.5,1.,2.):
            tracks[0].det_seen=t;engine.adopt_tracks(frame,tracks,[],roi,t,[])
        self.assertFalse(engine.jewels)
        self.assertIn('marker',engine.message)

    def test_stone_short_occlusion_reuses_marker_aligned_location_only(self):
        engine=make_engine('cog',lambda event:None)
        self.acquire(engine.aruco)
        engine.scene_status=engine.aruco.alignment(self.frame.shape)
        self.assertEqual((300,300,400,400),engine.resolve_stone((300,300,400,400),self.frame,1.))
        np.testing.assert_allclose(engine.resolve_stone(None,self.frame,1.5),(300,300,400,400),atol=1.)
        self.assertIsNone(engine.resolve_stone(None,self.frame,3.5))
        engine.scene_status={'trusted':False}
        self.assertIsNone(engine.resolve_stone(None,self.frame,1.5))

    def test_marker_loss_uses_existing_background_alignment(self):
        engine=make_engine('cog',lambda event:None)
        engine.aruco.alignment=lambda shape:None
        from unittest.mock import patch
        def fallback(instance,frame,excluded):
            instance.scene_status=dict(state='STABLE',source='background_test',trusted=True)
            return True
        with patch.object(HomePositionEngine,'assess_scene',fallback):
            self.assertTrue(engine.assess_scene(self.frame,np.zeros(self.frame.shape[:2],np.uint8)))
        self.assertEqual('background_test',engine.scene_status['source'])
        self.assertIn('aruco_status',engine.scene_status)

    def test_two_top_left_markers_do_not_select_an_arbitrary_reference(self):
        frame=self.frame.copy()
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000)
        frame[160:260,170:270]=cv2.cvtColor(cv2.aruco.generateImageMarker(dictionary,24,100),cv2.COLOR_GRAY2BGR)
        reference=ArucoReference()
        for _ in range(3):reference.detect(frame)
        self.assertIsNone(reference.identity)
        self.assertEqual('MARKER_UNAVAILABLE',reference.status['state'])

    def test_perspective_point_round_trip_and_area_scale(self):
        matrix=np.array([[1.01,.01,4.],[.01,1.02,2.],[.0001,.00015,1.]])
        status=dict(trusted=True,matrix=matrix.tolist())
        point=np.array([500.,350.])
        projected=cv2.perspectiveTransform(point.reshape(1,1,2),matrix)[0,0]
        np.testing.assert_allclose(reference_point(projected,status),point,atol=1e-6)
        self.assertGreater(scene_area_scale(projected,status),0.)

    def test_home_capture_after_marker_acquisition_uses_same_frame_reference(self):
        from types import SimpleNamespace
        engine=make_engine('cog',lambda event:None)
        frame=self.frame.copy()
        cv2.rectangle(frame,(300,300),(350,350),(20,120,180),-1)
        mask=np.full((50,50),255,np.uint8)
        track=SimpleNamespace(id=1,det_seen=0.,det_box=(300,300,350,350),det_mask=mask,det_origin=(300,300))
        roi=np.full(frame.shape[:2],255,np.uint8)
        for t in (0.,.25,.5,1.,1.5,2.,2.25):
            track.det_seen=t
            engine.adopt_tracks(frame,[track],[],roi,t,[])
        self.assertEqual(1,len(engine.jewels))
        self.assertIsNotNone(engine.aruco.reference)
        self.assertTrue(engine.assess_scene(frame,np.zeros_like(roi)))
        self.assertEqual('aruco_single_marker',engine.scene_status['source'])

    def test_marker_roi_maps_corners_back_to_full_frame(self):
        reference=ArucoReference((40/640,30/480,120/640,120/480))
        self.acquire(reference)
        self.assertGreater(reference.status['upscale'],1.)
        np.testing.assert_allclose(np.mean(reference.current,axis=0),[99.5,89.5],atol=1.)
        self.assertEqual('MARKER_VISIBLE',reference.status['state'])

    def test_custom_roi_can_find_marker_outside_default_top_left_search(self):
        frame=np.full_like(self.frame,230)
        frame[300:400,450:550]=self.frame[40:140,50:150]
        reference=ArucoReference((440/640,290/480,120/640,120/480))
        for _ in range(3):reference.detect(frame)
        self.assertEqual(23,reference.identity[1])
        np.testing.assert_allclose(reference.current.mean(0),[499.5,349.5],atol=1.)

    def test_apriltag_36h11_is_supported_in_marker_roi(self):
        dictionary=cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        frame=self.frame.copy()
        frame[40:140,50:150]=cv2.cvtColor(cv2.aruco.generateImageMarker(dictionary,1,100),cv2.COLOR_GRAY2BGR)
        reference=ArucoReference((40/640,30/480,120/640,120/480))
        for _ in range(3):reference.detect(frame)
        self.assertEqual(('DICT_APRILTAG_36h11',1),reference.identity)

    def test_rejection_reasons_are_separate(self):
        reference=ArucoReference()
        reference.detect(np.full_like(self.frame,230))
        self.assertEqual('no_supported_marker_decoded',reference.status['reason'])
        tiny=ArucoReference((0,0,.001,.001))
        tiny.detect(self.frame)
        self.assertEqual('marker_roi_too_small',tiny.status['reason'])

    def test_marker_roi_settings_are_normalized_and_validated(self):
        import tempfile
        from pathlib import Path
        from standalone.marker_roi_ui import load_marker_roi,save_marker_roi
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'roi.json'
            save_marker_roi(path,(.1,.05,.2,.2))
            self.assertEqual((.1,.05,.2,.2),load_marker_roi(path))
            save_marker_roi(path,None)
            self.assertIsNone(load_marker_roi(path))
            path.write_text('{"roi":[0,0,5,1]}')
            self.assertIsNone(load_marker_roi(path))

    def test_marker_acquisition_allows_corners_to_move(self):
        reference=ArucoReference()
        for shift in (0,5,10):
            frame=cv2.warpAffine(self.frame,np.float32([[1,0,shift],[0,1,0]]),(640,480),borderValue=(230,230,230))
            reference.detect(frame)
        self.assertEqual(('DICT_4X4_1000',23),reference.identity)
        self.assertTrue(reference.freeze())

    def test_home_processing_does_not_require_background_stability(self):
        home=make_engine('home',lambda event:None)
        self.assertTrue(home.assess_scene(np.full_like(self.frame,230),np.zeros(self.frame.shape[:2],np.uint8)))
        self.assertEqual('NOT_CHECKED',home.scene_status['state'])
        self.assertFalse(home.scene_status['stability_checked'])

    def test_missing_marker_uses_raw_frame_without_stability_gate(self):
        engine=make_engine('cog',lambda event:None)
        self.assertTrue(engine.assess_scene(self.frame,np.zeros(self.frame.shape[:2],np.uint8)))
        self.assertEqual('raw_frame',engine.scene_status['source'])
        self.assertTrue(engine.scene_status['trusted'])
        self.assertFalse(engine.scene_status['stability_checked'])

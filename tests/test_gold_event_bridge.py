import unittest
from types import SimpleNamespace
import numpy as np
from standalone.gold_mask_tracks import GoldMaskTracks
from standalone.gold_event_bridge import GoldEventBridge
from standalone.home_rubbing import HomeRubbingFlow


class GoldEventBridgeTests(unittest.TestCase):
    def setUp(self):
        self.shape=(180,220,3);self.roi=np.full(self.shape[:2],255,np.uint8)
        self.events=[];self.time=0.
        home=np.zeros(self.shape[:2],np.uint8);home[10:30,10:30]=255
        self.jewel=SimpleNamespace(id=1,box=(10,10,30,30),gold_mask=home,home_cog=np.array([20.,20.]),
            current_cog=None,associated_hand=1,metrics={},state='UNCERTAIN',contact_at=0.,verified_at=None)
        self.engine=SimpleNamespace(jewels=[self.jewel],gold_masks=GoldMaskTracks(),home_rubbing=HomeRubbingFlow(self.events.append),
            live_gold={},hand_tracks={},home_guard=np.zeros_like(self.roi),global_uncertain=False,
            scene_status=dict(state='NOT_CHECKED',trusted=True,matrix=[[1,0,0],[0,1,0]]))
        self.bridge=GoldEventBridge(self.events.append)

    def step(self,x=20,y=20,named=True,visible=True,hand=True,with_jewel=True):
        self.time+=.25
        mask=np.zeros(self.shape[:2],np.uint8);mask[y-10:y+10,x-10:x+10]=255
        detection=SimpleNamespace(box=(x-10,y-10,x+10,y+10),mask=mask)
        handdet=SimpleNamespace(points=np.full((21,2),(x,y),np.float32),scale=80.)
        self.engine.hand_tracks={1:dict(detection=handdet,cog=np.array([x,y],float))} if hand else {}
        gold=[detection] if visible else []
        tracked={1:dict(box=detection.box,seen=self.time)} if named and visible and with_jewel else {}
        self.engine.gold_masks.update(gold,[handdet] if hand else [],tracked,self.shape,self.time,self.engine.scene_status)
        self.engine.jewels=[self.jewel] if with_jewel else []
        self.engine.live_gold={1:detection} if named and visible and with_jewel else {}
        self.jewel.current_cog=np.array([x,y],float) if named and visible else None
        self.jewel.displacement=float(np.linalg.norm(np.array([x,y])-[20,20]))
        self.jewel.associated_hand=1 if named and hand else None
        self.jewel.metrics=dict(hand_jewel_proximity=named and hand,present=False,home_covered=hand)
        self.jewel.gold_quality=dict(accepted=named and visible,partial=False,reason='complete_gold_support' if named and visible else 'no_fresh_gold_detection')
        self.bridge.prepare(self.engine,self.time)
        self.engine.home_rubbing.update(self.engine,(75,75,125,125),self.roi,self.time)
        self.bridge.update_unassigned(self.engine,(75,75,125,125),self.roi,self.time)

    def test_anonymous_continuity_supports_candidate_without_claiming_named_pickup(self):
        self.step()
        for x in range(26,81,6):self.step(x=x,named=False)
        flow=self.jewel.workflow
        self.assertTrue(flow['departure_observed'])
        self.assertFalse(flow['departure_verified'])
        self.assertEqual('MOVEMENT_OBSERVED_ID_PENDING',flow['state'])
        self.assertIsNotNone(flow['temporary_track'])
        self.assertEqual('associated',flow['hand_status'])

    def test_fresh_identity_recovery_can_verify_after_continuity(self):
        self.step()
        for x in range(26,81,6):self.step(x=x,named=False)
        self.step(x=80,named=True);self.step(x=80,named=True)
        self.assertTrue(self.jewel.workflow['departure_verified'])

    def test_unassigned_gold_can_record_provisional_rubbing(self):
        for x in (90,100,110,100,90,100,110,100,90,100,110,100):
            self.step(x=x,y=100,named=False,with_jewel=False)
        rubbing=[event for event in self.events if event['state']=='UNASSIGNED_RUBBING_PROVISIONAL']
        self.assertTrue(rubbing)
        self.assertFalse(rubbing[0]['workflow']['departure_verified'])
        self.assertIsNone(rubbing[0]['workflow']['jewel_id'])
        self.assertTrue(self.bridge.records)
        self.assertTrue(any(event['rubbing_at'] is not None for event in self.bridge.unassigned))

    def test_short_detection_dropout_preserves_temp_id_without_inventing_cog(self):
        self.step()
        track=self.engine.gold_masks.snapshots()[0]['track']
        self.step(visible=False,named=False)
        snapshot=self.engine.gold_masks.snapshots()[0]
        self.assertEqual(track,snapshot['track'])
        self.assertFalse(snapshot['visible'])
        self.assertIsNone(snapshot['cog'])
        self.step(x=25)
        self.assertEqual(track,self.engine.gold_masks.snapshots()[0]['track'])

    def test_unknown_mask_without_prior_identity_does_not_attach_to_jewel(self):
        self.step(x=100,y=100,named=False)
        self.assertEqual({},self.engine.provisional_gold)
        self.assertNotIn(1,self.engine.home_rubbing.events)

    def test_unknown_event_records_close_when_track_is_lost(self):
        self.step(x=100,y=100,named=False,with_jewel=False)
        for _ in range(10):self.step(visible=False,named=False,hand=False,with_jewel=False)
        self.assertTrue(self.bridge.unassigned)
        self.assertFalse(self.bridge.unassigned[0]['active'])

    def test_brief_dropout_preserves_partial_movement_evidence(self):
        self.step()
        for x in (26,32,38,44,50):self.step(x=x,named=False)
        before=self.engine.home_rubbing.events[1].away_hits
        self.assertGreater(before,0)
        self.step(x=56,named=False,visible=False)
        self.assertEqual(before,self.engine.home_rubbing.events[1].away_hits)
        self.step(x=62,named=False)
        self.assertTrue(self.jewel.workflow['departure_observed'])
        self.assertFalse(self.jewel.workflow['departure_verified'])

    def test_unassigned_recording_is_saved_and_resolved_without_named_id(self):
        import tempfile
        from pathlib import Path
        from pickup_trigger_test import CandidateClips
        with tempfile.TemporaryDirectory() as tmp:
            clips=CandidateClips(Path(tmp),5)
            self.addCleanup(clips.close)
            self.step(x=100,y=100,named=False,with_jewel=False)
            clips.update(np.zeros(self.shape,np.uint8),self.bridge.records,self.time)
            self.assertTrue(clips.writers)
            for _ in range(10):self.step(visible=False,named=False,hand=False,with_jewel=False)
            clips.update(np.zeros(self.shape,np.uint8),self.bridge.records,self.time)
            self.assertFalse(clips.writers)
            self.assertTrue(list(Path(tmp).glob('*Gold-T*.avi')))

    def test_palm_and_wrist_can_support_continuity_without_tip_overlap(self):
        from standalone.hand_keypoints import home_contact_mask,hand_continuity_mask
        points=np.full((21,2),(100,100),np.float32)
        points[2:5]=(170,170);points[6:9]=(170,170)
        hand=SimpleNamespace(points=points,scale=40.)
        support=np.zeros(self.shape[:2],bool);support[90:110,90:110]=True
        self.assertFalse(np.any((home_contact_mask(hand,self.shape)>0)&support))
        self.assertTrue(np.any((hand_continuity_mask(hand,self.shape)>0)&support))

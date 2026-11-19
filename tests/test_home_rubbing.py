import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np
from standalone.home_rubbing import HomeRubbingFlow
from pickup_trigger_test import CandidateClips, make_engine


class HomeRubbingTests(unittest.TestCase):
    def setUp(self):
        self.events=[]
        self.flow=HomeRubbingFlow(self.events.append)
        self.roi=np.full((180,180),255,np.uint8)
        self.stone=(75,75,125,125)
        self.jewel=SimpleNamespace(id=1,box=(10,10,30,30),state='UNCERTAIN',
            contact_at=0.,associated_hand=1,metrics={},current_cog=None,displacement=None)
        self.engine=SimpleNamespace(jewels=[self.jewel],live_gold={},hand_tracks={},
            home_guard=np.zeros_like(self.roi),global_uncertain=False)
        self.t=0.

    def update(self, x=20, y=20, visible=True, near=True, hand=True, present=False,
               dt=.25, stone=True, stable=True, hand_id=1, extra_hand=None):
        self.t+=dt
        self.engine.global_uncertain=not stable
        self.jewel.associated_hand=hand_id if hand else None
        self.jewel.metrics=dict(hand_jewel_proximity=near,present=present,home_covered=hand)
        self.engine.hand_tracks={}
        if hand:
            points=np.full((21,2),(x,y),np.float32)
            detection=SimpleNamespace(points=points,scale=40.)
            self.engine.hand_tracks[hand_id]=dict(detection=detection,cog=np.array([x,y],float))
        if extra_hand is not None:
            other=SimpleNamespace(points=np.full((21,2),extra_hand,np.float32),scale=40.)
            self.engine.hand_tracks[1 if hand_id!=1 else 99]=dict(detection=other,cog=np.array(extra_hand,float))
        self.engine.live_gold={}
        self.jewel.current_cog=None
        self.jewel.displacement=None
        if visible:
            mask=np.zeros_like(self.roi)
            mask[int(y)-10:int(y)+10,int(x)-10:int(x)+10]=255
            self.engine.live_gold[1]=SimpleNamespace(mask=mask,box=(int(x)-10,int(y)-10,int(x)+10,int(y)+10))
            self.jewel.current_cog=np.array([x,y],float)
            self.jewel.displacement=float(np.linalg.norm(self.jewel.current_cog-[20,20]))
        self.flow.update(self.engine,self.stone if stone else None,self.roi,self.t)
        return self.jewel.workflow

    def rub(self, hidden=False):
        for x in (90,110,90,110,90,110,90,110):
            result=self.update(x,100,visible=not hidden)
        return result

    def return_home(self, x=20,y=20):
        for _ in range(5):
            result=self.update(x,y,hand=False,near=False,present=x==20 and y==20)
        return result

    def test_grip_records_before_departure_and_passing_cancels(self):
        initial=self.update(present=True)
        self.assertEqual('GRIP_CANDIDATE',initial['state'])
        self.assertTrue(initial['active'])
        result=self.return_home()
        self.assertEqual('CANCELLED',result['state'])
        self.assertFalse(result['active'])

    def test_visible_departure_rubbing_and_return_complete_provisionally(self):
        self.update(present=True)
        result=self.rub()
        self.assertEqual('RUBBING_PROVISIONAL',result['state'])
        self.assertTrue(result['departure_verified'])
        result=self.return_home()
        self.assertEqual('COMPLETED_PROVISIONAL',result['state'])
        self.assertFalse(result['physical_contact_verified'])
        self.assertFalse(result['active'])

    def test_hidden_pickup_rub_return_remains_explicitly_inferred(self):
        self.update(present=True)
        result=self.rub(hidden=True)
        self.assertEqual('RUBBING_PROVISIONAL',result['state'])
        self.assertFalse(result['departure_verified'])
        result=self.return_home()
        self.assertEqual('COMPLETED_INFERRED',result['state'])
        self.assertFalse(result['departure_verified'])

    def test_immediate_replacement_without_rubbing_retains_nonrub_event(self):
        self.update(present=True)
        for x in (50,60,70):self.update(x,40)
        result=self.return_home()
        self.assertEqual('RETURNED_RUBBING_UNVERIFIED',result['state'])
        self.assertIsNone(result['rubbing_at'])

    def test_stationary_hover_cannot_trigger_rubbing(self):
        self.update(present=True)
        for _ in range(12):result=self.update(100,100,visible=False)
        self.assertIsNone(result['rubbing_at'])
        self.assertEqual(0,result['reversal_count'])

    def test_one_way_hand_motion_cannot_trigger_rubbing(self):
        self.update(present=True)
        for x in range(80,121,5):result=self.update(x,100,visible=False)
        self.assertIsNone(result['rubbing_at'])

    def test_jitter_below_minimum_excursion_cannot_trigger(self):
        self.update(present=True)
        for x in (100,100.5)*8:result=self.update(x,100,visible=False)
        self.assertIsNone(result['rubbing_at'])

    def test_missing_stone_or_unstable_camera_does_not_infer_rubbing(self):
        for missing,unstable in ((True,False),(False,True)):
            with self.subTest(missing=missing,unstable=unstable):
                self.setUp();self.update(present=True)
                for x in (90,110)*5:result=self.update(x,100,visible=False,stone=not missing,stable=not unstable)
                self.assertIsNone(result['rubbing_at'])

    def test_gap_resets_motion_without_dropping_candidate(self):
        self.update(present=True)
        for x in (90,110,90):self.update(x,100,visible=False)
        result=self.update(110,100,visible=False,dt=2.)
        self.assertTrue(result['active'])
        self.assertEqual(0,result['reversal_count'])
        self.assertIsNone(result['rubbing_at'])

    def test_return_can_be_at_a_new_bench_position(self):
        self.update(present=True);self.rub()
        result=self.return_home(150,30)
        self.assertEqual('COMPLETED_PROVISIONAL',result['state'])

    def test_hidden_disappearance_then_return_alone_is_not_rubbing(self):
        self.update(present=True)
        self.update(40,40,visible=False)
        result=self.return_home()
        self.assertEqual('REVIEW_REQUIRED',result['state'])
        self.assertIsNone(result['rubbing_at'])

    def test_previous_pickup_confirmation_cannot_verify_a_new_cycle(self):
        self.jewel.state='PICKUP_CONFIRMED'
        self.jewel.verified_at=-10.
        result=self.update(present=True)
        self.assertFalse(result['departure_verified'])

    def test_candidate_at_new_bench_position_needs_new_departure(self):
        self.flow.bed_positions[1]=[150.,30.]
        for _ in range(4):result=self.update(150,30)
        self.assertFalse(result['departure_verified'])

    def test_first_grip_already_away_from_initial_home_can_verify_relocation(self):
        for _ in range(4):result=self.update(150,30)
        self.assertTrue(result['departure_verified'])
        self.assertEqual([20.,20.],result['origin_cog'])

    def test_verified_pickup_stays_verified_when_rubbing_is_unverified(self):
        self.update(present=True)
        for x in (50,60,70):self.update(x,40)
        result=self.return_home()
        self.assertEqual('VERIFIED',result['pickup_status'])
        self.assertEqual('UNVERIFIED',result['rubbing_status'])
        self.assertEqual('RETURNED',result['return_status'])
        self.assertIsNotNone(result['pickup_verified_at'])

    def test_different_position_cannot_cancel_when_departure_was_missed(self):
        self.update(present=True)
        self.jewel.gold_quality=dict(partial=False,accepted=True)
        # Deliberately interrupt departure accumulation; full mask at a new place
        # still must never be described as staying at its starting position.
        event=self.flow.events[1]
        for _ in range(5):
            event.away_hits=0;event.away_since=None
            result=self.update(60,40,hand=False,near=False)
        self.assertEqual('REVIEW_REQUIRED',result['state'])
        self.assertNotEqual('CANCELLED_NO_DEPARTURE',result['pickup_status'])

    def test_new_event_keeps_previous_verified_result_in_history(self):
        self.update(present=True);self.rub();self.return_home()
        old=self.jewel.workflow
        self.update(20,20,hand=False,near=False,present=True)
        result=self.update(20,20)
        self.assertEqual(2,result['event_number'])
        self.assertEqual('VERIFIED',result['previous_event']['pickup_status'])
        self.assertEqual(1,len(self.flow.history[1]))
        self.assertTrue(old['departure_verified'])

    def test_stone_box_is_recorded_even_when_not_detected_next_frame(self):
        result=self.update(present=True)
        self.assertEqual(list(self.stone),result['stone_observation'])
        result=self.update(stone=False)
        self.assertIsNone(result['stone_observation'])

    def test_rubbing_measurements_are_preserved_after_return(self):
        self.update(present=True);self.rub()
        result=self.return_home()
        self.assertGreaterEqual(result['rubbing_evidence']['reversals'],3)
        self.assertGreaterEqual(result['rubbing_evidence']['active_seconds'],1.2)

    def test_rubbing_not_cancelled_by_legacy_home_cancellation(self):
        self.update(present=True);self.rub(hidden=True)
        self.jewel.state='PICKUP_CANCELLED'
        result=self.update(100,100,visible=False)
        self.assertEqual('RUBBING_PROVISIONAL',result['state'])
        self.assertTrue(result['active'])

    def test_wrong_or_missing_return_identity_stays_pending(self):
        self.update(present=True);self.rub(hidden=True)
        for _ in range(6):result=self.update(20,20,visible=False,near=False,hand=False)
        self.assertTrue(result['active'])
        self.assertIsNone(result['returned_at'])

    def test_recording_stays_open_through_home_cancellation_until_return(self):
        frame=np.zeros((180,180,3),np.uint8)
        with tempfile.TemporaryDirectory() as tmp:
            clips=CandidateClips(Path(tmp),5)
            self.addCleanup(clips.close)
            self.update(present=True)
            clips.update(frame,[self.jewel],self.t)
            self.assertIn(1,clips.writers)
            self.rub(hidden=True);self.jewel.state='PICKUP_CANCELLED'
            clips.update(frame,[self.jewel],self.t)
            self.assertIn(1,clips.writers)
            self.return_home()
            clips.update(frame,[self.jewel],self.t)
            self.assertNotIn(1,clips.writers)

    def test_home_factory_enables_flow_other_modes_do_not(self):
        self.assertIsInstance(make_engine('home',lambda e:None).home_rubbing,HomeRubbingFlow)
        self.assertTrue(hasattr(make_engine('cog',lambda e:None),'home_rubbing'))

    def test_one_hand_cannot_start_two_jewel_sequences(self):
        other=SimpleNamespace(**vars(self.jewel))
        other.id=2
        other.metrics=dict(hand_jewel_proximity=True,present=True)
        self.engine.jewels.append(other)
        result=self.update(present=True)
        self.assertEqual([2],result['remaining_at_home'])
        self.assertEqual([1],list(self.flow.events))

    def test_unresolved_timeout_retains_review_event(self):
        self.update(present=True)
        self.flow.events[1].started=self.t-121
        result=self.update(100,100,visible=False,hand=False,near=False)
        self.assertEqual('REVIEW_REQUIRED',result['state'])
        self.assertFalse(result['active'])

    def test_camera_translation_alone_cannot_verify_departure(self):
        self.engine.scene_status=dict(state='STABLE',reason='fixed_background_aligned',source='test',trusted=True,matrix=[[1,0,0],[0,1,0]])
        self.update(present=True)
        for i in range(1,9):
            self.engine.scene_status.update(state='CAMERA_MOVED',reason='measured_camera_shift',matrix=[[1,0,i*8],[0,1,0]])
            result=self.update(20+i*8,20,stable=False)
        self.assertFalse(result['departure_verified'])
        self.assertEqual('CAMERA_MOVED',result['stability']['state'])

    def test_independent_jewel_motion_verifies_with_trusted_camera_compensation(self):
        self.engine.scene_status=dict(state='STABLE',reason='fixed_background_aligned',source='test',trusted=True,matrix=[[1,0,0],[0,1,0]])
        self.update(present=True)
        for i in range(1,9):
            self.engine.scene_status.update(state='CAMERA_MOVED',reason='measured_camera_shift',matrix=[[1,0,i*6],[0,1,0]])
            result=self.update(20+i*12,20,stable=False)
        self.assertTrue(result['departure_verified'])
        self.assertEqual('PICKUP_VERIFIED',result['state'])

    def test_insufficient_alignment_preserves_candidate_and_tracks_hand(self):
        self.update(present=True)
        self.engine.scene_status=dict(state='INSUFFICIENT_EVIDENCE',reason='too_few_background_points',source='test',trusted=False,matrix=None)
        result=self.update(90,100,visible=False)
        self.assertFalse(result['departure_verified'])
        self.assertTrue(result['active'])
        self.assertEqual(self.t,self.flow.events[1].hand_seen)
        self.assertIn('too_few_background_points',result['reason'])
        self.assertNotIn('unstable camera',result['reason'])

    def test_frame_gap_reason_is_separate_from_camera_alignment(self):
        self.update(present=True)
        result=self.update(90,100,visible=False,dt=1.5)
        self.assertIn('Frame gap',result['reason'])
        self.assertNotIn('camera',result['reason'])

    def test_new_hand_id_reconnects_after_long_gap_and_rubbing_resumes(self):
        self.update(present=True)
        self.update(visible=False,hand=False,near=False,dt=1.5)
        first=self.update(90,100,hand_id=7)
        self.assertEqual('confirming_new_hand_id',first['hand_status'])
        for x in (110,90,110,90,110,90,110,90):
            result=self.update(x,100,hand_id=7)
        self.assertEqual(7,result['hand_id'])
        self.assertEqual(1,result['hand_rebindings'])
        self.assertIsNotNone(result['rubbing_at'])

    def test_stale_old_hand_does_not_override_unique_live_gold_association(self):
        self.update(present=True)
        self.update(90,100,hand_id=7,extra_hand=(20,20))
        result=self.update(110,100,hand_id=7,extra_hand=(20,20))
        self.assertEqual(7,result['hand_id'])

    def test_two_hands_near_live_gold_do_not_rebind_or_supply_rubbing(self):
        self.update(present=True)
        for x in (90,110)*5:
            result=self.update(x,100,hand_id=7,extra_hand=(x,100))
        self.assertEqual('ambiguous_hands_near_gold',result['hand_status'])
        self.assertIsNone(result['rubbing_at'])
        self.assertEqual(1,result['hand_id'])

    def test_partial_cog_motion_with_stationary_hand_cannot_verify(self):
        self.update(present=True)
        self.jewel.gold_quality=dict(partial=True,area_ratio=.27,reason='partial_gold_provisional')
        # Re-evaluate fresh gold positions with the hand fixed near the mask.
        for x in (40,45,50,55,60):
            self.update(x,20,hand=False,near=True)
            self.engine.hand_tracks[1]=dict(detection=SimpleNamespace(points=np.full((21,2),(50,20),np.float32),scale=40.),cog=np.array([50.,20.]))
            self.t+=.05
            self.flow.update(self.engine,self.stone,self.roi,self.t)
        self.assertFalse(self.jewel.workflow['departure_verified'])

    def test_partial_masks_with_consistent_hand_motion_can_verify(self):
        self.update(present=True)
        self.jewel.gold_quality=dict(partial=True,area_ratio=.27,reason='partial_gold_provisional')
        for x in range(26,81,6):result=self.update(x,20)
        self.assertTrue(result['departure_verified'])

    def test_partial_mask_cannot_finish_return_identity_check(self):
        self.update(present=True);self.rub(hidden=True)
        self.jewel.gold_quality=dict(partial=True,area_ratio=.27,reason='partial_gold_provisional')
        result=self.return_home()
        self.assertTrue(result['active'])
        self.assertIsNone(result['returned_at'])

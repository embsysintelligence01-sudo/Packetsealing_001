import unittest
from standalone.flow import PackingFlow, Observation, Pickup, INSIDE, PICKED, UNVERIFIED, INTERACTING, AT_PACKET


class PackingTests(unittest.TestCase):
    def setUp(self):
        self.flow = PackingFlow([1, 2])
        self.t = 100.
        self.bench = Observation((20., 20.))
        self.bag = Observation((100., 20.), packet_overlap=.9, transfer_verified=True)

    def step(self, obs=None, **kwargs):
        self.t += .4
        args = dict(packet_ok=True, hands_clear=True, strip='present', packet_stationary=True, closure='closed')
        args.update(kwargs)
        self.flow.update(self.t, {} if obs is None else obs, **args)

    def take(self, jid=1):
        self.step(pickup=Pickup(True, jid, False, (jid,)), packet_ok=False, hands_clear=False)

    def pack(self, jid=1):
        self.take(jid)
        self.step({jid: self.bag}, hands_clear=False, packet_stationary=False)
        self.assertEqual(self.flow.jewels[jid].status, INSIDE)

    def test_original_pickup_starts_recording_without_packet_or_gallery(self):
        self.step(pickup=Pickup(True, 1), packet_ok=False, identities_ready=False)
        self.assertTrue(self.flow.recording_required)
        self.assertEqual(self.flow.jewels[1].status, PICKED)
        self.assertEqual(self.flow.jewels[1].started, self.t)

    def test_motion_or_bag_visibility_alone_is_not_pickup(self):
        self.step({1: self.bag}, strip='unknown')
        self.assertFalse(self.flow.recording_required)
        self.assertIsNone(self.flow.jewels[1].started)
        self.assertFalse(self.flow.all_packed)

    def test_first_accepted_strip_starts_sealing_without_pickups_or_placements(self):
        self.step(packet_ok=False, packet_context_ok=True, packet_stationary=False, hands_clear=False)
        started=self.t
        self.assertTrue(self.flow.strip_removal_requested)
        self.assertTrue(self.flow.strip_seen)
        self.assertTrue(self.flow.recording_required)
        self.assertEqual(self.flow.sealing_started_at,started)
        self.assertIsNone(self.flow.bed_verified_at)
        self.assertFalse(self.flow.all_packed)
        self.assertTrue(all(j.started is None for j in self.flow.jewels.values()))
        self.assertIn('remove the sealing strip',self.flow.message)
        self.assertIn('Placement unconfirmed: J1, J2',self.flow.message)
        for _ in range(6):
            self.step(strip='absent',packet_ok=False,packet_context_ok=True,packet_stationary=False,hands_clear=False)
        self.assertFalse(self.flow.sealed)
        for _ in range(5):
            self.step(strip='absent',packet_ok=False,packet_context_ok=True,packet_stationary=False,hands_clear=True)
        self.assertTrue(self.flow.sealed)
        self.assertEqual(self.flow.sealing_started_at,started)
        self.assertEqual(self.flow.unconfirmed_placements,[1,2])
        self.assertTrue(all(j.status=='On bench' for j in self.flow.jewels.values()))

    def test_unknown_rejected_or_context_free_strip_cannot_start_sealing(self):
        for value in ('unknown','uncertain','absent'):
            self.step(strip=value)
            self.assertFalse(self.flow.strip_removal_requested)
        self.step(strip='present',packet_context_ok=False)
        self.assertFalse(self.flow.strip_removal_requested)
        self.assertIsNone(self.flow.sealing_started_at)

    def test_strip_detection_does_not_change_unconfirmed_pickup_to_inside(self):
        self.take()
        self.step(packet_ok=False,packet_context_ok=True,packet_stationary=False)
        self.assertTrue(self.flow.strip_removal_requested)
        self.assertEqual(self.flow.jewels[1].status,PICKED)
        self.assertIsNone(self.flow.jewels[1].ended)

    def test_contact_only_never_confirms_placement_even_when_sealing_finishes(self):
        self.take()
        contact = Observation((100., 20.), near_packet=True, method='pickup_owned_hand_contact')
        self.step({1: contact})
        self.assertEqual(self.flow.jewels[1].status, AT_PACKET)
        self.assertIsNone(self.flow.jewels[1].ended)
        self.assertIn('insertion unconfirmed', self.flow.jewels[1].note)
        for _ in range(5):
            self.step({1: contact}, strip='absent')
        self.assertTrue(self.flow.sealed)
        self.assertEqual(self.flow.unconfirmed_placements, [1, 2])
        self.assertEqual(self.flow.jewels[1].status, AT_PACKET)

    def test_placement_does_not_wait_for_clear_hands_or_stationary_bag(self):
        self.pack()
        self.assertEqual(self.flow.jewels[1].ended, self.t)
        self.assertEqual(self.flow.jewels[1].placement_method, 'pickup_owned_packet_interaction')

    def test_verified_owned_hand_arrival_can_infer_placement_with_identity_hidden(self):
        self.take()
        self.step({1: Observation((90., 20.), identity_ok=False, near_packet=True, transfer_verified=True)}, packet_ok=False, hands_clear=False)
        self.assertEqual(self.flow.jewels[1].status, INSIDE)

    def test_pickup_without_interaction_never_becomes_inside_or_lost(self):
        self.take()
        for _ in range(40):
            self.step({}, packet_ok=False)
        self.assertEqual(self.flow.jewels[1].status, PICKED)
        self.assertIsNone(self.flow.jewels[1].entry)

    def test_next_pickup_handoffs_clip_immediately(self):
        self.pack()
        ended = self.flow.jewels[1].ended
        self.take(2)
        self.assertEqual(self.flow.jewels[1].clip_ended, self.t)
        self.assertEqual(self.flow.jewels[2].started, self.t)
        self.assertEqual(self.flow.jewels[1].ended, ended)
        self.assertEqual(self.flow.jewels[1].status, INSIDE)

    def test_next_pickup_retains_previous_interaction_stage(self):
        self.take()
        self.take(2)
        self.assertEqual(self.flow.jewels[1].status, INTERACTING)
        self.assertEqual(self.flow.jewels[1].clip_ended, self.t)
        self.step({1: self.bag})
        self.assertEqual(self.flow.jewels[1].status, INSIDE)

    def test_handoff_frame_can_confirm_outgoing_transfer_without_identity_check(self):
        self.take()
        self.step({1: Observation((100., 20.), near_packet=True, identity_ok=False, transfer_verified=True)},
                  pickup=Pickup(True, 2), packet_ok=False, hands_clear=False)
        self.assertEqual(self.flow.jewels[1].status, INSIDE)
        self.assertEqual(self.flow.jewels[2].status, PICKED)
        self.assertEqual(self.flow.jewels[1].clip_ended, self.t)

    def test_detector_dropout_and_next_pickup_never_fail_old_interaction(self):
        self.take()
        self.take(2)
        self.flow.pause_packet()
        for _ in range(50):
            self.step(packet_ok=False, hands_clear=False)
        self.assertEqual(self.flow.jewels[1].status, INTERACTING)
        self.assertEqual(self.flow.jewels[2].status, PICKED)
        self.assertTrue(self.flow.recording_required)
        self.assertFalse(self.flow.all_packed)

    def test_completed_placement_is_never_reopened(self):
        self.pack()
        entry = self.flow.jewels[1].entry
        for _ in range(10):
            self.step({1: self.bench}, pickup=Pickup(True, 1))
        self.flow.invalidate_packet('Camera gap')
        self.flow.pause_packet()
        self.assertEqual(self.flow.jewels[1].status, INSIDE)
        self.assertEqual(self.flow.jewels[1].entry, entry)
        self.assertEqual(self.flow.jewels[1].attempts, [])

    def test_duplicate_and_old_timestamps_do_not_handoff(self):
        self.take()
        self.flow.update(self.t, {1: self.bag}, packet_ok=True, hands_clear=False, strip='present', pickup=Pickup(True, 2))
        self.assertEqual(self.flow.active_jewel_id, 1)
        self.assertIsNone(self.flow.jewels[1].entry)

    def test_ambiguous_pickup_records_without_guessing_or_erasing(self):
        self.pack()
        self.step(pickup=Pickup(True, None, True, (1, 2)))
        self.assertEqual(self.flow.jewels[1].status, INSIDE)
        self.assertIsNone(self.flow.jewels[2].started)
        self.assertEqual(len(self.flow.unassigned_pickups), 1)

    def both(self):
        self.pack(1)
        self.pack(2)
        for _ in range(3):
            self.step()
        self.assertTrue(self.flow.strip_removal_requested)

    def test_verified_strip_removal_completes_without_closure_model(self):
        self.both()
        for _ in range(4):
            self.step(strip='absent', closure='model_missing')
        self.assertFalse(self.flow.sealed)
        self.step(strip='absent', closure='model_missing')
        self.assertTrue(self.flow.strip_removed)
        self.assertTrue(self.flow.sealed)
        self.assertEqual(self.flow.sealing_method, 'strip_removal_inferred')
        self.assertEqual(self.flow.closure_status, 'inferred_from_strip_removal')
        removed_at = self.flow.strip_removed_at
        self.flow.pause_packet()
        self.flow.invalidate_packet('Packet out of view')
        self.step(strip='present', closure='open', packet_ok=False,
                  packet_context_ok=False, hands_clear=False)
        self.assertTrue(self.flow.sealed)
        self.assertEqual(self.flow.strip_removed_at, removed_at)
        self.assertIn('inferred', self.flow.message)

    def test_closed_classifier_cannot_bypass_strip_removal(self):
        self.both()
        for _ in range(10):
            self.step(strip='uncertain', closure='closed')
        self.assertFalse(self.flow.sealed)

    def test_strip_baseline_required_and_preserved_during_packet_dropout(self):
        self.both()
        self.flow.pause_packet()
        for _ in range(30):
            self.step(packet_ok=False, strip='uncertain')
        self.assertTrue(self.flow.strip_seen)
        self.assertTrue(self.flow.all_packed)
        self.assertIn('retained', self.flow.message)
        self.flow.strip_seen = False
        for _ in range(8):
            self.step(strip='absent')
        self.assertFalse(self.flow.sealed)


if __name__ == '__main__':
    unittest.main()

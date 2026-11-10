import unittest
from standalone.flow import PackingFlow, Observation, Pickup, INSIDE, PICKED, UNVERIFIED, INTERACTING


class PackingTests(unittest.TestCase):
    def setUp(self):
        self.flow = PackingFlow([1, 2])
        self.t = 100.
        self.bench = Observation((20., 20.))
        self.bag = Observation((100., 20.), packet_overlap=.9)

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
        self.step({1: self.bag})
        self.assertFalse(self.flow.recording_required)
        self.assertIsNone(self.flow.jewels[1].started)
        self.assertFalse(self.flow.all_packed)

    def test_placement_does_not_wait_for_clear_hands_or_stationary_bag(self):
        self.pack()
        self.assertEqual(self.flow.jewels[1].ended, self.t)
        self.assertEqual(self.flow.jewels[1].placement_method, 'pickup_owned_packet_interaction')

    def test_near_packet_owned_hand_can_confirm_even_if_identity_hidden(self):
        self.take()
        self.step({1: Observation((90., 20.), identity_ok=False, near_packet=True)}, packet_ok=False, hands_clear=False)
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
        self.step({1: Observation((100., 20.), near_packet=True, identity_ok=False)},
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

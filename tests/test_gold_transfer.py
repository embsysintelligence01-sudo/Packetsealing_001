import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import cv2
import numpy as np
import packet_app
from standalone.gold_transfer import GoldTransfer, interaction_region
from standalone.app import PacketWorker
from standalone.flow import PackingFlow, Pickup, INSIDE


class GoldTransferTests(unittest.TestCase):
    shape = (400, 600, 3)

    def bag(self):
        mask = np.zeros(self.shape[:2], np.uint8)
        mask[100:300, 50:210] = 255
        return SimpleNamespace(mask=mask, box=(50, 100, 210, 300))

    def gold(self, x, y=200):
        return SimpleNamespace(c=np.array([float(x), float(y)]),
                               mask=np.full((20, 20), 255, np.uint8), origin=(x-10, y-10))

    def test_moving_gold_confirms_packet_entry_without_any_hand_model(self):
        path = GoldTransfer(2, (500, 200), 100.)
        result = None
        for i, x in enumerate((500, 450, 400, 350, 300, 250, 205)):
            result = path.observe(100.+i*.2, [self.gold(x)], self.bag(), self.shape)
        self.assertIsNotNone(result)
        self.assertEqual(path.owner, 2)
        self.assertEqual(result.method, 'pickup_owned_gold_packet_interaction')
        self.assertFalse(result.identity_ok)  # No inside identity classifier required.

    def test_gold_already_visible_inside_packet_does_not_complete_new_pickup(self):
        path = GoldTransfer(2, (500, 200), 100.)
        for i in range(8):
            self.assertIsNone(path.observe(100.+i*.2, [self.gold(120)], self.bag(), self.shape))

    def test_static_previous_jewel_cannot_become_current_transfer_at_packet_edge(self):
        path = GoldTransfer(2, (500, 200), 100.)
        # Previous jewel at x=180, normalized within the packet.
        path.blocked_packet_positions = ((130/160, .5),)
        for i, x in enumerate((500, 450, 400, 350, 300, 250)):
            self.assertIsNone(path.observe(100.+i*.2, [self.gold(x), self.gold(180)], self.bag(), self.shape))
        self.assertIsNone(path.observe(101.2, [self.gold(180)], self.bag(), self.shape))
        self.assertIsNone(path.last_interaction)

    def test_fast_current_transfer_rejects_older_packet_gold_after_detector_gap(self):
        # Geometry from the supplied video's second insertion: old gold at
        # (715,540), incoming gold at (547,708). The old jewel was missed in
        # the immediately preceding frames, but its packet-relative history
        # must still exclude it from the current path.
        shape = (1080, 1920, 3)
        mask = np.zeros(shape[:2], np.uint8)
        mask[294:874, 400:1008] = 255
        bag = SimpleNamespace(mask=mask)
        path = GoldTransfer(1, (1350, 253), 100.)
        path.blocked_packet_positions = ((.52, .34),)
        for i, (x, y) in enumerate(((1350, 253), (1402, 270), (1156, 643))):
            self.assertIsNone(path.observe(100.+i*.2, [self.gold(x, y)], bag, shape))
        incoming, old = self.gold(547, 708), self.gold(715, 540)
        result = path.observe(100.6, [old, incoming], bag, shape)
        self.assertIsNotNone(result)
        self.assertIs(path.last_detection, incoming)
        self.assertEqual(path.owner, 1)

    def test_old_gold_is_excluded_even_when_packet_mask_has_a_hole_at_its_center(self):
        shape = (1080, 1920, 3)
        mask = np.zeros(shape[:2], np.uint8)
        mask[322:910, 394:934] = 255
        mask[557, 762] = 0
        path = GoldTransfer(1, (1350, 253), 100.)
        path.blocked_packet_points = ((715., 540.),)
        bag = SimpleNamespace(mask=mask)
        for i, (x, y) in enumerate(((1350, 253), (1402, 270), (1156, 643))):
            path.observe(100.+i*.2, [self.gold(x, y)], bag, shape)
        self.assertIsNone(path.observe(100.6, [self.gold(762, 557)], bag, shape))

    def test_worker_hand_route_still_confirms_when_gold_is_missing(self):
        worker = PacketWorker(0, [])
        worker.flow = PackingFlow([2])
        worker.flow.update(100., {}, pickup=Pickup(True, 2), packet_ok=True, hands_clear=False, strip='present')
        worker.tracker = SimpleNamespace(bench_anchors={2: ((500., 200.), (490, 190, 510, 210))}, raw_detections=[])
        worker._observations = Mock(return_value={})
        worker.transfer.start(2, (500, 200), 100.)
        frame = np.zeros(self.shape, np.uint8)
        for i, x in enumerate((500, 450, 400, 350, 300, 250, 200)):
            t = 100.+i*.2
            hand = np.zeros(self.shape[:2], np.uint8)
            cv2.circle(hand, (x, 200), 30, 255, -1)
            observations = worker._transfer_observations(frame, t, self.bag(), hand, Pickup())
            if i:
                worker.flow.update(t, observations, packet_ok=True, hands_clear=False, strip='present')
        self.assertEqual(worker.flow.jewels[2].status, INSIDE)
        self.assertEqual(worker.flow.jewels[2].placement_method, 'pickup_owned_hand_packet_interaction')
        self.assertEqual(worker.gold_transfers[2].hits, 0)

    def test_duplicate_timestamp_cannot_build_a_gold_trajectory(self):
        path = GoldTransfer(1, (250, 200), 100.)
        path.observe(100., [self.gold(250)], self.bag(), self.shape)
        for _ in range(5):
            self.assertIsNone(path.observe(100., [self.gold(200)], self.bag(), self.shape))
        self.assertEqual(path.hits, 1)

    def test_long_blind_gap_cannot_jump_path_onto_packet_contents(self):
        path = GoldTransfer(1, (250, 200), 100.)
        path.observe(100., [self.gold(250)], self.bag(), self.shape)
        self.assertIsNone(path.observe(102., [self.gold(180)], self.bag(), self.shape))

    def test_path_can_recover_locally_outside_packet_after_long_gap(self):
        path = GoldTransfer(1, (500, 200), 100.)
        path.observe(100., [self.gold(500)], self.bag(), self.shape)
        self.assertIsNone(path.observe(103., [self.gold(500)], self.bag(), self.shape))
        result = None
        for i, x in enumerate((450, 400, 350, 300, 250, 205)):
            result = path.observe(103.2+i*.2, [self.gold(x)], self.bag(), self.shape)
        self.assertIsNotNone(result)
        self.assertEqual(result.method, 'pickup_owned_gold_packet_interaction')

    def test_ambiguous_gold_candidates_are_not_arbitrarily_assigned(self):
        path = GoldTransfer(1, (250, 200), 100.)
        self.assertIsNone(path.observe(100., [self.gold(240), self.gold(260)], self.bag(), self.shape))
        self.assertEqual(path.hits, 0)

    def test_worker_or_condition_uses_current_gold_transfer_with_blank_hand_masks(self):
        worker = PacketWorker(0, [])
        worker.flow = PackingFlow([2])
        worker.flow.update(100., {}, pickup=Pickup(True, 2), packet_ok=True, hands_clear=False, strip='present')
        worker.tracker = SimpleNamespace(bench_anchors={2: ((500., 200.), (490, 190, 510, 210))})
        worker._observations = Mock(return_value={})
        frame = np.zeros(self.shape, np.uint8)
        for i, x in enumerate((500, 450, 400, 350, 300, 250, 205)):
            t = 100.+i*.2
            worker.tracker.raw_detections = [self.gold(x)]
            worker.tracker.raw_timestamp = t
            observations = worker._transfer_observations(frame, t, self.bag(), np.zeros(self.shape[:2], np.uint8), Pickup())
            if i:
                worker.flow.update(t, observations, packet_ok=True, hands_clear=False, strip='present')
        self.assertEqual(worker.flow.jewels[2].status, INSIDE)
        self.assertEqual(worker.flow.jewels[2].placement_method, 'pickup_owned_gold_packet_interaction')

    def test_current_gold_reaching_open_packet_footprint_is_not_lost_in_mask_hole(self):
        mask = np.zeros(self.shape[:2], np.uint8)
        mask[100:300, 50:60] = mask[100:300, 200:210] = 255
        mask[290:300, 50:210] = 255  # Segmented film around an open centre.
        region = interaction_region(SimpleNamespace(mask=mask))
        path = GoldTransfer(1, (400, 200), 100.)
        result = None
        for i, x in enumerate((400, 350, 300, 250, 215)):
            result = path.observe(100.+i*.2, [self.gold(x)], region, self.shape)
        self.assertIsNotNone(result)
        self.assertEqual(result.method, 'pickup_owned_gold_packet_interaction')

    def test_stale_raw_detections_do_not_feed_gold_fallback(self):
        worker = PacketWorker(0, [])
        worker.flow = PackingFlow([1])
        worker.tracker = SimpleNamespace(bench_anchors={1: ((500, 200), (490, 190, 510, 210))},
                                         raw_timestamp=99., raw_detections=[self.gold(500)])
        worker._observations = Mock(return_value={})
        worker._transfer_observations(np.zeros(self.shape, np.uint8), 100., self.bag(), None, Pickup())
        self.assertEqual(worker.gold_transfers[1].hits, 0)

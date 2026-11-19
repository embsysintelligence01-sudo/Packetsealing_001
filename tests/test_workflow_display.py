import unittest
from types import SimpleNamespace
from standalone.workflow_display import sequence_rows

class WorkflowDisplayTests(unittest.TestCase):
    def test_verified_pickup_and_unverified_rubbing_have_distinct_columns(self):
        workflow=dict(pickup_status='VERIFIED',rubbing_status='UNVERIFIED',return_status='RETURNED',
            pickup_observed_at=2.,pickup_verified_at=3.,event_number=1)
        jewel=SimpleNamespace(id=1,workflow=workflow)
        row=sequence_rows([['J1','RETURNED_RUBBING_UNVERIFIED','VISIBLE','4','old','old','reason']],
            [jewel],[],lambda t:str(t))[0]
        self.assertEqual(['VERIFIED','UNVERIFIED','RETURNED'],row[1:4])
        self.assertEqual('3.0',row[7])
        self.assertTrue(row[8].startswith('E1 '))

    def test_anonymous_event_cannot_display_a_verified_pickup(self):
        workflow=dict(identity_verified=False,pickup_status='VERIFIED',event_number=1)
        track=dict(track='Gold-T1',workflow=workflow)
        row=sequence_rows([['Gold-T1','UNASSIGNED_RUBBING_PROVISIONAL','GOLD MASK','4','old','old','reason']],
            [],[track],lambda t:str(t))[0]
        self.assertEqual('UNASSIGNED',row[1])
        self.assertEqual('unverified',row[7])

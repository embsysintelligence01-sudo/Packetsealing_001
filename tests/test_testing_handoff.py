"""Keep testing identity/results linked to packet number and sealing exports."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import test_evidence as fixtures
from standalone.evidence import export_session, ffmpeg_path


class TestingHandoffReportTests(unittest.TestCase):
    def test_export_keeps_testing_result_and_confirmed_packet_number_together(self):
        with tempfile.TemporaryDirectory() as tmp:
            testing = Path(tmp)
            # A recognizable stand-in proves copying does not rewrite the original report.
            original_report = b'%PDF-1.4\noriginal testing evidence\n'
            (testing/'report.pdf').write_bytes(original_report)
            directory = testing/'packet_sealing'
            directory.mkdir()
            work, recorder, flow = fixtures.EvidenceTests().fixture(directory)
            flow.testing_session = dict(directory=str(testing), jewels=[dict(id=1, rubbed=True,
                acid_done=True, prediction=dict(prediction='gold_22k', confidence=.98))])
            flow.packet_number = dict(status='confirmed', text='PC123456', confidence=.99,
                                      confirmed_at=102.8)
            result = export_session(directory, work, recorder, flow, 102.9, 'Complete', ffmpeg_path())
            self.assertEqual(original_report, (testing/'report.pdf').read_bytes())
            self.assertEqual(original_report, (directory/'jewel_testing_report.pdf').read_bytes())
            saved = json.loads((directory/'evidence.json').read_text())
            self.assertEqual(1, saved['testing_session']['jewels'][0]['id'])
            self.assertEqual('gold_22k', saved['testing_session']['jewels'][0]['prediction']['prediction'])
            self.assertEqual('PC123456', saved['packet_number']['text'])
            self.assertTrue(result['sealed'])
            if shutil.which('pdftotext'):
                text = subprocess.check_output(['pdftotext', str(directory/'report.pdf'), '-'], text=True)
                self.assertIn('J1 | Jewel testing result', text)
                self.assertIn('GOLD_22K', text)
                self.assertIn('98.0%', text)
                self.assertIn('Packet number: PC123456', text)
                self.assertIn('SEALED (INFERRED)', text)


if __name__ == '__main__':
    unittest.main()

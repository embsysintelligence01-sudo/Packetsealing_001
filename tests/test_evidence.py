from pathlib import Path
import tempfile
import shutil
import subprocess
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from standalone.evidence import Recorder, export_session, ffmpeg_path, image_save
from standalone.flow import PackingFlow, INSIDE


class EvidenceTests(unittest.TestCase):
    def test_failed_recorder_with_full_queue_does_not_deadlock_on_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = Recorder(Path(tmp))
            recorder.enabled.set()
            recorder.offer(np.zeros((240, 320, 3), np.uint8), 100.)
            recorder.offer(np.zeros((200, 300, 3), np.uint8), 100.2)
            recorder.thread.join(timeout=2)
            self.assertFalse(recorder.thread.is_alive())
            while not recorder.queue.full():
                recorder.queue.put_nowait((101., np.zeros((20, 20, 3), np.uint8)))
            with self.assertRaises(OSError):
                recorder.close()

    def fixture(self, root):
        work = root / ".working"
        work.mkdir()
        recorder = Recorder(work)
        recorder.enabled.set()
        for i in range(30):
            frame = np.full((240, 320, 3), 220, np.uint8)
            cv2.circle(frame, (20 + i * 7, 120), 14, (0, 160, 210), -1)
            cv2.putText(frame, str(i), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, .6, (10, 10, 10), 1)
            recorder.offer(frame, 100 + i * .101)
        recorder.close()
        flow = PackingFlow([1])
        j = flow.jewels[1]
        j.started, j.entry, j.ended = 100.6, 101.5, 102.4
        j.status, j.note = INSIDE, "Synthetic observed entry"
        flow.strip_seen = flow.sealed = flow.strip_removed = flow.strip_removal_requested = True
        flow.bed_verified_at, flow.strip_removed_at = 101.5, 102.4
        flow.closure_status, flow.closure_confidence, flow.packet_stationary = "inferred_from_strip_removal", None, True
        flow.sealing_method = "strip_removal_inferred"
        image_save(work / "J1_initial.jpg", frame)
        image_save(work / "J1_entry.jpg", frame)
        image_save(work / "J1_after.jpg", frame)
        image_save(work / "final.jpg", frame)
        return work, recorder, flow

    def test_complete_export_decodes_and_meets_combined_budget(self):
        exe = ffmpeg_path()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, recorder, flow = self.fixture(root)
            result = export_session(root, work, recorder, flow, 102.9, "Synthetic test", exe, 102.4)
            self.assertTrue(result["sealed"])
            self.assertEqual(result["sealing_method"], "strip_removal_inferred")
            self.assertEqual(result["closure"]["status"], "inferred_from_strip_removal")
            if shutil.which("pdftotext"):
                report_text = subprocess.check_output(['pdftotext', str(root / 'report.pdf'), '-'], text=True)
                self.assertIn('SEALED (INFERRED)', report_text)
                self.assertIn('Sealing is inferred from verified strip removal', ' '.join(report_text.split()))
            self.assertTrue(result["all_budgets_met"])
            self.assertEqual(len(result["exports"]), 2)
            self.assertTrue(all(r["combined_bytes"] <= 2_000_000 for r in result["exports"]))
            self.assertFalse(work.exists())
            self.assertEqual([p.name for p in root.glob("*.pdf")], ["report.pdf"])
            self.assertTrue((root / "report.pdf").read_bytes().startswith(b"%PDF"))
            self.assertEqual(result["jewels"][0]["pdf"], "report.pdf")
            self.assertTrue((root / "report.pdf").is_file())
            self.assertTrue((root / "evidence.json").is_file())

    def test_packet_number_crop_and_confirmation_survive_work_cleanup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, recorder, flow = self.fixture(root)
            flow.packet_number = dict(status='confirmed', text='PC123456', confidence=.98,
                                      confirmed_at=102.7, crop='packet_number_crop.jpg',
                                      scene='packet_number_scene.jpg')
            for name in ('packet_number_crop.jpg', 'packet_number_scene.jpg'):
                image_save(work / name, np.full((60, 240, 3), 255, np.uint8))
            result = export_session(root, work, recorder, flow, 102.9, 'OCR complete', ffmpeg_path())
            self.assertTrue(result['sealed'])
            self.assertEqual(result['packet_number']['text'], 'PC123456')
            self.assertFalse(work.exists())
            self.assertTrue((root / 'packet_number_crop.jpg').is_file())
            self.assertTrue((root / 'packet_number_scene.jpg').is_file())
            if shutil.which('pdftotext'):
                text = subprocess.check_output(['pdftotext', str(root / 'report.pdf'), '-'], text=True)
                self.assertIn('Packet number: PC123456', text)

    def test_early_finish_keeps_sealing_but_does_not_confirm_candidate_number(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, recorder, flow = self.fixture(root)
            flow.packet_number = dict(status='reading', text='PC123456', confidence=.98, confirmed_at=None)
            result = export_session(root, work, recorder, flow, 102.9, 'Operator finished early', ffmpeg_path())
            self.assertTrue(result['sealed'])
            self.assertEqual(result['packet_number']['status'], 'reading')
            if shutil.which('pdftotext'):
                text = subprocess.check_output(['pdftotext', str(root / 'report.pdf'), '-'], text=True)
                self.assertIn('Packet number: Not confirmed', text)
                self.assertNotIn('Packet number: PC123456', text)

    @unittest.skipUnless(shutil.which("pdftotext"), "PDF text inspection needs pdftotext")
    def test_one_shared_report_contains_all_jewels_and_clip_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, recorder, flow = self.fixture(root)
            from standalone.flow import Jewel
            flow.jewels[2] = Jewel(2)
            result = export_session(root, work, recorder, flow, 102.9, "Session report test", ffmpeg_path())
            self.assertEqual([p.name for p in root.glob("*.pdf")], ["report.pdf"])
            self.assertTrue(all(r['pdf'] == 'report.pdf' for r in result['jewels']))
            text = subprocess.check_output(['pdftotext', str(root/'report.pdf'), '-'], text=True)
            for label in ('Last updated:', 'J1 | Packing evidence', 'J2 | Packing evidence',
                          'Pickup / recording started', 'Open J1 transfer video', 'Evidence not captured'):
                self.assertIn(label, text)
            self.assertIn(b'/URI (J1.avi)', (root/'report.pdf').read_bytes())
            pages = [p for p in text.split('\f') if p.strip()]
            self.assertEqual(len(pages), 4)
            self.assertIn('Sealed packet | Sealing evidence', pages[3])
            self.assertIn('Sealing started:', pages[3])
            self.assertIn('Sealing ended:', pages[3])
            self.assertIn('JEWEL PLACEMENT OUTCOMES', pages[0])
            self.assertIn('Open J1 transfer video', pages[0])
            for index, jid in enumerate((1, 2), 1):
                self.assertIn(f'J{jid} | Packing evidence', pages[index])
                for label in ('01 / Pickup', '02 / Packet interaction', '03 / Placement result retained'):
                    self.assertIn(label, pages[index])
            self.assertFalse(result['sealed'])

    def test_previous_jewel_clip_ends_at_next_pickup_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, recorder, flow = self.fixture(root)
            flow.jewels[1].clip_ended = 102.6
            result = export_session(root, work, recorder, flow, 102.9, "Handoff", ffmpeg_path())
            record = result['jewels'][0]
            self.assertEqual(record['clip_end'], 102.6)
            video = cv2.VideoCapture(str(root / 'J1.avi'))
            duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / video.get(cv2.CAP_PROP_FPS)
            video.release()
            # Available pre-roll begins at recorder.start=100, handoff at102.6.
            self.assertAlmostEqual(duration, 2.6, delta=.2)

    def test_overlapping_recording_gap_revokes_sealing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, recorder, flow = self.fixture(root)
            recorder.gaps.append((101., 101.7, "Synthetic gap"))
            result = export_session(root, work, recorder, flow, 102.9, "Synthetic test", ffmpeg_path())
            self.assertFalse(result["sealed"])
            self.assertEqual(result["jewels"][0]["status"], INSIDE)
            self.assertEqual(result["jewels"][0]["evidence_status"], "review_required")

    def test_failed_encoder_preserves_source_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            work, recorder, flow = self.fixture(root)
            with patch("standalone.evidence.compact_clip", side_effect=OSError("Synthetic encoder failure")):
                with self.assertRaises(OSError):
                    export_session(root, work, recorder, flow, 102.9, "Test", "ffmpeg")
            self.assertTrue(recorder.path.is_file())
            self.assertTrue((work / "J1_initial.jpg").is_file())


if __name__ == "__main__":
    unittest.main()

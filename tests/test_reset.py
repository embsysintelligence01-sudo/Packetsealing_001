"""Reset waits for save/release before replacing the session worker."""
import os
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import packet_app
from PyQt5 import QtWidgets
from standalone.app import MainWindow


class ResetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def fake_worker(self, running, sealed=False):
        return SimpleNamespace(isRunning=Mock(return_value=running), stop=Mock(), deleteLater=Mock(),
                               flow=SimpleNamespace(sealed=sealed), grabber=None, last_status="Original session")

    def test_active_reset_waits_for_export_before_starting_a_fresh_session(self):
        window = MainWindow()
        old = self.fake_worker(True)
        window.worker = old
        window.source.setText("2")
        window.roi_override, window.roi = True, [(0, 0), (100, 0), (100, 100)]
        window.regions_locked = True
        window.table.setRowCount(1)
        window.table.setItem(0, 0, QtWidgets.QTableWidgetItem("J1"))
        with patch.object(window, "connect_camera") as connect:
            window.reset_button.click()
            window.reset_process()  # A repeated request cannot queue another restart.
            old.stop.assert_called_once()
            self.assertIn("reset by operator", old.last_status)
            connect.assert_not_called()
            self.assertTrue(window.restart_pending)
            self.assertFalse(window.reset_button.isEnabled())
            self.assertEqual(window.table.rowCount(), 1)  # Old evidence remains visible while saving.
            old.isRunning.return_value = False
            window.worker_finished()
            connect.assert_not_called()
            self.app.processEvents()
            connect.assert_called_once()
            old.deleteLater.assert_called_once()
            self.assertFalse(window.restart_pending)
            self.assertTrue(window.reset_button.isEnabled())
            self.assertFalse(window.regions_locked)
            self.assertEqual(window.table.rowCount(), 0)
            self.assertEqual(window.source.text(), "2")
            self.assertEqual(window.roi, [(0, 0), (100, 0), (100, 100)])
        window.close()

    def test_reset_after_completed_session_preserves_its_result(self):
        window = MainWindow()
        old = self.fake_worker(False, sealed=True)
        window.worker = old
        window.table.setRowCount(2)
        with patch.object(window, "connect_camera") as connect:
            window.reset_process()
            connect.assert_called_once()
            old.stop.assert_not_called()
            self.assertTrue(old.flow.sealed)
            self.assertEqual(old.last_status, "Original session")
            self.assertEqual(window.table.rowCount(), 0)
            self.assertIsNone(window.last_image)
        window.close()

    def test_closing_cancels_the_pending_automatic_restart(self):
        window = MainWindow()
        old = self.fake_worker(True)
        window.worker = old
        with patch.object(window, "connect_camera") as connect:
            window.reset_process()
            window.close()
            self.assertFalse(window.restart_pending)
            old.isRunning.return_value = False
            window.worker_finished()
            self.app.processEvents()
            connect.assert_not_called()
        window.close()


if __name__ == "__main__":
    unittest.main()

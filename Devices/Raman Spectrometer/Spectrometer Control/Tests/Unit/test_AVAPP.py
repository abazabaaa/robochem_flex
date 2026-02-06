"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

TEST DESCR:
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tkinter as tk
from gui_app.RamaBerryAPP import MainApplication  # Replace with your actual import


class TestMainApplication(unittest.TestCase):
    @patch("gui_app.AVAPP.AVAPP_Backend", mock=MagicMock(), autospec=True)
    def setUp(self, mock_backend):
        # Mock the backend and queue to avoid actual operations
        self.app.backend = mock_backend
        self.app.command_queue = Mock()

        self.app.backend.stop_acq.side_effect = lambda: "stop_acq called"
        self.root = tk.Tk()
        self.app = MainApplication(self.root)

    def test_collect_spectrum_button(self):
        # Set dummy values for parameters
        for param, var in self.app.param_vars.items():
            var.set("1")

        # Set dummy value for path
        self.app.path_entry.insert(0, "/dummy/path")

        # Simulate clicking the "Collect Spectrum" button
        collect_spectrum_button = next(
            filter(
                lambda b: b.cget("text") == "Collect Spectrum",
                self.app.controls_container.winfo_children(),
            )
        )
        collect_spectrum_button.invoke()

        # Verify the backend method is called with correct parameters
        self.app.command_queue.put.assert_called_with(
            {
                "command": "set_parameters",
                "parameters": {
                    "integration_time": "1",
                    "n_averages": "1",
                    "n_scans": "1",
                    "save_each_n": "1",
                    "correct_dark": True,
                    "save_as": self.app.mode_var.get(),
                    "save_path": "/dummy/path",
                    "safety_threshold": 1000,
                },
            }
        )

    def test_stop_acquisition_button(self):
        # Simulate clicking the "Stop Acquisition" button
        stop_acquisition_button = next(
            filter(
                lambda b: b.cget("text") == "Stop Acquisition",
                self.app.controls_container.winfo_children(),
            )
        )
        stop_acquisition_button.invoke()

        # Verify the stop_acq method of the backend was called
        self.app.backend.stop_acq.assert_called_once()

    def tearDown(self):
        self.root.destroy()


if __name__ == "__main__":
    unittest.main()

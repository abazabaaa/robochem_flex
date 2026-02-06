"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: test file for the avalanche utils

"""

import unittest
from unittest.mock import patch, Mock, MagicMock
from unittest import TestCase
from gui_app.avalanche_utils import (
    thread_safe_updates,
    aesthetic_setup,
    update_display,
    on_close,
    submit_parameters,
)


class TestTkinterUtils(unittest.TestCase):
    def setUp(self):
        # Create mock objects for root, ax, backend, queue, etc.
        self.root = Mock()
        self.ax = Mock()
        self.backend = Mock()
        self.queue = Mock()

    def test_thread_safe_updates(self):
        func = Mock()
        thread_safe_updates(self.root, func)
        self.root.after_idle.assert_called_with(func)

    def test_aesthetic_setup(self):
        aesthetic_setup(self.ax)
        # Assert the expected calls on ax (set_xlabel, set_ylabel, etc.)
        self.ax.set_xlabel.assert_called_with("Raman Shift (cm$^{-1}$)")
        self.ax.set_ylabel.assert_called_with("Intensity (counts)")
        self.ax.set_xlim.assert_called_with(200, 3500)

    def test_update_display(self):
        # Mock data
        dat = ([], [])
        update_display(self.ax, dat, meas_method="single")
        # Test various assertions depending on the behavior of update_display

    def test_on_close(self):
        on_close(self.root, self.backend)
        self.backend.stop.assert_called()
        self.root.destroy.assert_called()

    def test_submit_parameters(self):
        submit_parameters(self.queue)
        # Test the parameters in the queue
        expected_command = {
            "command": "set_parameters",
            "parameters": {
                "integration_time": 1000,
                "n_averages": 1,
                "n_scans": 1,
                "save_each_n": 1,
                "correct_dark": True,
                "save_as": "single",
                "save_path": None,
                "safety_threshold": 1000,
            },
        }
        self.queue.put.assert_called_with(expected_command)

    # Add more tests as needed for different scenarios and edge cases


if __name__ == "__main__":
    unittest.main()

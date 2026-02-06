"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

TEST DESCR: Comprehensive test for the AVAPP backend. This test will run the backend
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from gui_app.AVAPP_backend import AVAPP_Backend as Backend


class TestBackend(unittest.TestCase):
    def setUp(self):
        self.update_gui_callback = Mock()
        self.parameter_queue = MagicMock()
        self.spec = Mock()
        with patch("gui_app.AVAPP_backend.AVAlanCHE", return_value=self.spec):
            self.backend = Backend(self.update_gui_callback, self.parameter_queue)

    def test_init(self):
        self.assertEqual(self.backend.spec, self.spec)
        self.assertEqual(self.backend.update_gui_callback, self.update_gui_callback)
        self.assertEqual(self.backend.data_queue, self.parameter_queue)
        self.assertIsNone(self.backend.acq_thread)
        self.assertIsNone(self.backend.poll_thread)
        self.assertTrue(self.backend.running)

    def test_gather_params(self):
        # Mock the parameter queue behavior and test gather_params

        # Case 1: Queue has a command
        command = {"command": "set_parameters", "parameters": {"param1": "value1"}}
        self.parameter_queue.get.return_value = command
        self.backend.gather_params()
        self.spec.set_parameters.assert_called_once_with(command["parameters"])
        import queue

        # Case 2: Queue is empty
        self.parameter_queue.get.side_effect = queue.Empty()
        self.backend.gather_params()
        # No exception should be raised and set_parameters should not be called again
        self.spec.set_parameters.assert_called_once()

    def test_start_acq(self):
        # Case 1: Threads have not been started yet
        with patch("threading.Thread") as mock_thread:
            self.backend.start_acq()
            self.assertEqual(mock_thread.call_count, 2)
            self.assertTrue(self.backend.acq_thread.is_alive())
            self.assertTrue(self.backend.poll_thread.is_alive())

        # Case 2: Threads are already running
        self.backend.start_acq()
        # Thread instantiation count should not increase
        self.assertEqual(mock_thread.call_count, 2)

    def test_stop_acq(self):
        # Setup active threads
        self.backend.acq_thread = MagicMock()
        self.backend.poll_thread = MagicMock()
        self.backend.acq_thread.is_alive.return_value = True
        self.backend.poll_thread.is_alive.return_value = True

        self.backend.stop_acq()

        # Check if spectrometer stop method was called
        self.spec.stop.assert_called_once()

        # Check if running flag is set to False
        self.assertFalse(self.backend.running)

        # Check if threads were set to None after stopping
        self.assertIsNone(self.backend.acq_thread)
        self.assertIsNone(self.backend.poll_thread)

    def test_acq_process_normal_operation(self):
        # Normal operation
        self.backend.running = True
        with patch.object(self.backend, "spec") as mock_spec:
            self.backend._acq_process()
            mock_spec.get_dark.assert_called_once()
            mock_spec.run_measurement.assert_called_once()

    def test_acq_process_with_exception(self):
        # Exception during the process
        self.backend.running = True
        with patch.object(self.backend, "spec") as mock_spec:
            mock_spec.run_measurement.side_effect = Exception(
                "Error during measurement"
            )
            with self.assertRaises(Exception):
                self.backend._acq_process()
            self.assertFalse(self.backend.running)

    def test_poll_process_normal_operation(self):
        def mock_update_gui_side_effect(dummy):
            self.backend.running = False

        # Normal operation
        self.backend.running = True
        self.spec.polltime = 1000  # 1 second for easier testing
        with patch("time.sleep", return_value=None) as mock_sleep:
            with patch.object(self.backend, "update_gui_callback") as mock_update_gui:
                # Simulate one iteration of the loop
                mock_update_gui.side_effect = mock_update_gui_side_effect
                self.backend._poll_process()
                mock_update_gui.assert_called_once_with(self.spec.data)
                mock_sleep.assert_called_once_with(1)

    def test_poll_process_with_exception(self):
        # Exception during the process
        self.backend.running = True
        with patch.object(self.backend, "update_gui_callback") as mock_update_gui:
            mock_update_gui.side_effect = Exception("Error during GUI update")
            with self.assertRaises(Exception):
                self.backend._poll_process()
            self.assertFalse(self.backend.running)

    def test_save_now(self):
        self.backend.save_now()
        self.spec.save_data.assert_called()

    def test_stop(self):
        self.backend.stop()
        self.assertFalse(self.backend.running)
        # ... other assertions ...

    # Add more tests as needed for different scenarios and edge cases


if __name__ == "__main__":
    unittest.main()

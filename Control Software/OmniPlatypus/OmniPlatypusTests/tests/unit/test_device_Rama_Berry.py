"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

TEST DESCR:
"""

import unittest
from omniplatypus.devices.nrg.rama_berry import RamaBerry
from omniplatypus.devices.base.device import DeviceParameter
import json
from unittest.mock import patch
import pandas as pd


class RamaBerryTest(unittest.TestCase):
    @patch("socket.socket")
    def setUp(self, mock_socket):
        """Setup runs before each test method."""
        self.device = RamaBerry()
        self.mock_socket = mock_socket.return_value

        # Setup mock to return a successful response
        self.mock_socket.recv.return_value = json.dumps({"status": "success"}).encode()

        # Test open method
        self.device.open("localhost", 8080)

        # Check if socket.connect was called
        self.mock_socket.connect.assert_called_with(("localhost", 8080))

    def tearDown(self):
        """Teardown runs after each test method."""
        self.mock_socket.recv.return_value = json.dumps(
            {"status": "success", "message": "Hasta La Vista, Baby!"}
        ).encode()
        self.device.close()

    # def test_open_connection_failure(self):
    #     """Test handling failure when opening the connection."""
    #     self.mock_socket.recv.return_value = json.dumps({"status": "fail", "message": "Error"}).encode()
    #
    #     with self.assertRaises(Exception) as context:
    #         self.device.open('localhost', 8080)
    #
    #     self.assertTrue("Failed to setup spectrometer" in str(context.exception))

    def test_send_command(self):
        """Test sending a command to the device."""
        # Setup mock to return a successful response for both calls
        self.mock_socket.recv.return_value = json.dumps({"status": "success"}).encode()

        # Assuming the 'spectrometer_setup' command is sent during initialization or another setup method
        # and 'test_command' is sent here
        response = self.device._send_command("test_command")

        # Check that sendall was called twice
        self.assertEqual(self.mock_socket.sendall.call_count, 2)

        # Check the arguments of the first call (spectrometer_setup)
        args1, kwargs1 = self.mock_socket.sendall.call_args_list[0]
        sent_command1 = json.loads(args1[0].decode())
        self.assertEqual(sent_command1["command"], "spectrometer_setup")

        # Check the arguments of the second call (test_command)
        args2, kwargs2 = self.mock_socket.sendall.call_args_list[1]
        sent_command2 = json.loads(args2[0].decode())
        self.assertEqual(sent_command2["command"], "test_command")

        # Verify the response from the _send_command method
        self.assertEqual(response, {"status": "success"})

    #
    def test_start_acquisition_success(self):
        """Test starting acquisition successfully."""
        self.mock_socket.recv.return_value = json.dumps({"status": "success"}).encode()

        self.device["start_acq"] = True

        # Verify parameter value is set to False after successful start
        self.assertFalse(self.device["start_acq"])

    #
    def test_stop_acquisition_success(self):
        """Test starting acquisition successfully."""
        self.mock_socket.recv.return_value = json.dumps({"status": "success"}).encode()

        self.device["stop_acq"] = True

        # Verify parameter value is set to False after successful start
        self.assertFalse(self.device["stop_acq"])

    #
    def test_stop_acquisition_failure(self):
        """Test handling failure when stopping acquisition."""
        self.mock_socket.recv.return_value = json.dumps(
            {"status": "error", "message": "Error"}
        ).encode()

        with self.assertRaises(Exception) as context:
            self.device["stop_acq"] = True

        self.assertTrue("Failed to stop acquisition" in str(context.exception))

    def test_start_acquisition_failure(self):
        """Test handling failure when stopping acquisition."""
        self.mock_socket.recv.return_value = json.dumps(
            {"status": "error", "message": "Error"}
        ).encode()

        with self.assertRaises(Exception) as context:
            self.device["start_acq"] = True

        self.assertTrue("Failed to " in str(context.exception))

    #
    def test_set_parameters(self):
        """Test setting all parameters sequentially."""
        self.mock_socket.recv.return_value = json.dumps({"status": "success"}).encode()
        parameters_to_set = [
            ("integration_time", 5000.0),
            ("n_averages", 5),
            ("delay", 2000.0),
            ("n_scans", 10),
            ("dark_correction", True),
        ]

        for param_name, value in parameters_to_set:
            self.device[param_name] = value
            self.mock_socket.sendall.assert_called_with(
                json.dumps(
                    {"command": "set_parameters", "data": {param_name: value}}
                ).encode()
            )

    #
    def test_poll_data_success(self):
        """Test successful data polling."""
        self.mock_socket.recv.return_value = json.dumps(
            {"status": "success", "data": {"sample_data": [1, 2, 3]}}
        ).encode()

        response = self.device._poll_data()
        self.assertIsNotNone(response)
        self.assertEqual(
            response, {"data": {"sample_data": [1, 2, 3]}, "status": "success"}
        )

    #
    def test_read_data(self):
        """Test the _read function for polling data."""
        self.mock_socket.recv.return_value = json.dumps(
            {"status": "success", "data": {"sample_data": [1, 2, 3]}}
        ).encode()

        response = self.device["data"]
        self.assertIsInstance(response, pd.DataFrame)
        self.assertEqual(list(response.columns), ["sample_data"])
        self.assertEqual(response.iloc[0]["sample_data"], 1)

    def test_extract(self):
        """tests all combinations of the _extract function"""
        self.assertIsNone(self.device._extract("dummy", None))
        with self.assertRaises(Exception):
            self.device._extract("dummy", Exception("dummy"))

        result = self.device._extract("dummy", {"status": "booleans", "value": True})
        self.assertTrue(result)
        result = self.device._extract(
            "dummy", {"status": "success", "data": {"col1": [1, 2], "col2": [3, 4]}}
        )
        self.assertTrue(isinstance(result, pd.DataFrame))
        self.assertEqual(list(result.columns), ["col1", "col2"])

        result = self.device._extract("dummy", {"status": "success", "data": "done"})
        self.assertEqual(result, "done")

        with self.assertRaises(Exception) as context:
            self.device._extract("dummy", {"status": "error", "message": "Test Error"})
        self.assertIn("Test Error", str(context.exception))
        with self.assertRaises(Exception) as context:
            self.device._extract("dummy", "unexpected format")
        self.assertIn("Data not in the right format", str(context.exception))

    # patch the device._poll_data method to return something
    @patch("devices.Rama_Berry.RamaBerry._poll_data")
    def test_read(self, mock_poll_data):
        """Test the _read method."""
        mock_poll_data.return_value = {"status": "success", "data": {"test": "value"}}
        response = self.device._read(
            DeviceParameter(
                name="data",
                access_level=3,
                value_type=pd.DataFrame,
                internal_id=207,
            )
        )
        self.assertEqual(response, {"status": "success", "data": {"test": "value"}})

        parameter = DeviceParameter(
            name="start_acq", access_level=3, value_type=pd.DataFrame, internal_id=207
        )
        parameter.last_known_value = False
        response = self.device._read(parameter)
        self.assertEqual(response, {"status": "booleans", "value": False})

        parameter = DeviceParameter(
            name="stop_acq", access_level=3, value_type=pd.DataFrame, internal_id=207
        )
        parameter.last_known_value = False
        response = self.device._read(parameter)
        self.assertEqual(response, {"status": "booleans", "value": False})


if __name__ == "__main__":
    unittest.main()

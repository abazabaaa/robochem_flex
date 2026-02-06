import unittest
from unittest.mock import MagicMock, patch
import json
import socket


# Mock external dependencies
class DeviceParameter:
    def __init__(self, name, access_level, value_type, internal_id):
        self.name = name
        self.access_level = access_level
        self.value_type = value_type
        self.internal_id = internal_id
        self.description = None
        self.timeout = 10


class ParameterAccess:
    R = "R"
    W = "W"
    RW = "RW"


class ParameterEnumValue:
    _raw_values = {}


class ParameterValueRun:
    pass


class SocketDevice:
    def __init__(self):
        self.parameters = {}
        self._socket = None

    def add_parameter(self, parameter):
        self.parameters[parameter.name] = parameter

    def log(self, message, level="info"):
        pass  # For testing, we can just pass


class DeviceTimeoutError(Exception):
    pass


class InvalidResponseError(Exception):
    pass


class NoResponseError(Exception):
    pass


class FailedWriteError(Exception):
    pass


# Now, import your classes
from omniplatypus.devices.nrg.chromtroller import (
    HPLCClient,
    HPLCProcessingSettings,
    ValvePosValue,
    SocketDeviceParameter,
)


class TestHPLCClient(unittest.TestCase):
    def setUp(self):
        self.client = HPLCClient()
        # Mock the socket
        self.mock_socket = MagicMock()
        self.client._socket = self.mock_socket
        # Mock the lock
        self.client._socket_lock = MagicMock()

    def test_initialization(self):
        self.assertEqual(self.client.generic_name, "Chromtroller HPLC Client")
        # Test that parameters are added correctly
        parameters = self.client.parameters
        self.assertIn("run_info", parameters)
        self.assertIn("analysis_parameters", parameters)
        self.assertIn("rt_target", parameters)
        self.assertIn("rt_tolerance", parameters)
        self.assertIn("valve_position", parameters)
        self.assertIn("acquisition", parameters)
        self.assertIn("analysis_result", parameters)

    def test_send_to_server(self):
        command = "test_command"
        data = {"key": "value"}

        # Mock '_check_and_clear_buffer' to do nothing
        self.client._check_and_clear_buffer = MagicMock()

        # Call the method
        self.client._send_to_server(command, data)

        # Now check that 'self.mock_socket.sendall' was called with the correct data
        expected_message_dict = {"command": command, "data": data}
        message = json.dumps(expected_message_dict).encode()
        message_length = len(message).to_bytes(4, byteorder="big")
        expected_send_data = message_length + message

        self.mock_socket.sendall.assert_called_once_with(expected_send_data)

    def test_receive_from_server(self):
        # Prepare the data that the mock socket will return
        message_dict = {"type": "reply", "data": "test_data"}
        message = json.dumps(message_dict).encode()
        message_length = len(message).to_bytes(4, byteorder="big")

        # When 'recv' is called, it should first return 'message_length', then 'message'
        self.mock_socket.recv = MagicMock(side_effect=[message_length, message])

        # Call the method
        response = self.client._receive_from_server()

        # Check that the response is correct
        self.assertEqual(response, message_dict)

    def test_get_reply_success(self):
        # Mock '_receive_from_server' to return a reply message
        self.client._receive_from_server = MagicMock(
            return_value={"type": "reply", "data": "test_data"}
        )

        # Call the method
        response = self.client._get_reply(timeout=5.0)

        # Check that the response is correct
        self.assertEqual(response, "test_data")

    def test_get_reply_error(self):
        # Mock '_receive_from_server' to return an error message
        self.client._receive_from_server = MagicMock(
            return_value={"type": "error", "data": "error_message"}
        )

        # Call the method and check that it raises an exception
        with self.assertRaises(Exception) as context:
            self.client._get_reply(timeout=5.0)
        self.assertEqual(str(context.exception), "error_message")

    def test_get_reply_no_response(self):
        # Mock '_receive_from_server' to return None
        self.client._receive_from_server = MagicMock(return_value=None)

        # Call the method and check that it raises NoResponseError
        with self.assertRaises(NoResponseError):
            self.client._get_reply(timeout=1.0)

    def test_read(self):
        # Prepare parameter
        parameter = SocketDeviceParameter(
            name="analysis_result",
            access_level=ParameterAccess.R,
            value_type=str,
            internal_id="run_data_analysis",
        )
        parameter.timeout = 60

        # Mock '_send_to_server' and '_get_reply'
        self.client._send_to_server = MagicMock()
        self.client._get_reply = MagicMock(return_value="analysis_data")

        # Call '_read'
        result = self.client._read(parameter)

        # Check that '_send_to_server' was called with correct command
        self.client._send_to_server.assert_called_once_with(
            command=parameter.internal_id
        )

        # Check that '_get_reply' was called with correct timeout
        self.client._get_reply.assert_called_once_with(timeout=parameter.timeout)

        # Check that the result is correct
        self.assertEqual(result, "analysis_data")

    def test_write_success(self):
        # Prepare parameter
        parameter = SocketDeviceParameter(
            name="rt_target",
            access_level=ParameterAccess.W,
            value_type=float,
            internal_id="set_analysis_target",
        )
        value = 5.0

        # Mock '_send_to_server' and '_get_reply'
        self.client._send_to_server = MagicMock()
        self.client._get_reply = MagicMock(return_value="OK")

        # Call '_write'
        self.client._write(parameter, value)

        # Check that '_send_to_server' was called with correct command and data
        self.client._send_to_server.assert_called_once_with(
            command=parameter.internal_id, data=value
        )

        # Check that '_get_reply' was called
        self.client._get_reply.assert_called_once()

    def test_write_error(self):
        # Prepare parameter
        parameter = SocketDeviceParameter(
            name="rt_target",
            access_level=ParameterAccess.W,
            value_type=float,
            internal_id="set_analysis_target",
        )
        value = 5.0

        # Mock '_send_to_server' and '_get_reply'
        self.client._send_to_server = MagicMock()
        self.client._get_reply = MagicMock(return_value="Error")

        # Call '_write' and expect exception
        with self.assertRaises(FailedWriteError):
            self.client._write(parameter, value)

    def test_write_hplc_processing_settings(self):
        # Prepare parameter
        parameter = SocketDeviceParameter(
            name="analysis_parameters",
            access_level=ParameterAccess.W,
            value_type=HPLCProcessingSettings,
            internal_id="set_analysis_parameters",
        )
        # Create an instance of HPLCProcessingSettings
        settings = HPLCProcessingSettings()

        # Mock '_send_to_server' and '_get_reply'
        self.client._send_to_server = MagicMock()
        self.client._get_reply = MagicMock(return_value="OK")

        # Call '_write'
        self.client._write(parameter, settings)

        # Check that '_send_to_server' was called with correct command and data
        expected_data = settings.to_dict()
        self.client._send_to_server.assert_called_once_with(
            command=parameter.internal_id, data=expected_data
        )

        # Check that '_get_reply' was called
        self.client._get_reply.assert_called_once()

    def test_check_and_clear_buffer_with_data(self):
        # Mock 'recv' to return some data
        residual_data = b"residual_data"
        self.mock_socket.recv = MagicMock(return_value=residual_data)

        # Mock 'setblocking' to do nothing
        self.mock_socket.setblocking = MagicMock()

        # Mock 'log' method
        self.client.log = MagicMock()

        # Call '_check_and_clear_buffer'
        self.client._check_and_clear_buffer()

        # Check that 'setblocking' was called with False and then True
        self.mock_socket.setblocking.assert_any_call(False)
        self.mock_socket.setblocking.assert_any_call(True)

        # Check that 'recv' was called
        self.mock_socket.recv.assert_called_once_with(4096)

        # Check that 'log' was called with the residual data
        self.client.log.assert_called_once_with(
            message=f"Residual data remaining in socket buffer: {residual_data}",
            level="warning",
        )

    def test_check_and_clear_buffer_no_data(self):
        # Mock 'recv' to raise BlockingIOError
        self.mock_socket.recv = MagicMock(side_effect=BlockingIOError)

        # Mock 'setblocking' to do nothing
        self.mock_socket.setblocking = MagicMock()

        # Mock 'log' method
        self.client.log = MagicMock()

        # Call '_check_and_clear_buffer'
        self.client._check_and_clear_buffer()

        # Check that 'setblocking' was called with False and then True
        self.mock_socket.setblocking.assert_any_call(False)
        self.mock_socket.setblocking.assert_any_call(True)

        # Check that 'recv' was called
        self.mock_socket.recv.assert_called_once_with(4096)

        # Check that 'log' was not called
        self.client.log.assert_not_called()

    def test_receive_from_server_timeout(self):
        # Mock 'settimeout' to do nothing
        self.mock_socket.settimeout = MagicMock()

        # Mock 'recv' to raise socket.timeout
        self.mock_socket.recv = MagicMock(side_effect=socket.timeout)

        # Call '_receive_from_server' and expect 'DeviceTimeoutError'
        with self.assertRaises(DeviceTimeoutError):
            self.client._receive_from_server(timeout=5.0)

        # Check that 'settimeout' was called with 5.0 and then None
        self.mock_socket.settimeout.assert_any_call(5.0)
        self.mock_socket.settimeout.assert_any_call(None)

    def test_receive_from_server_invalid_response(self):
        # Mock 'settimeout' to do nothing
        self.mock_socket.settimeout = MagicMock()

        # Mock 'recv' to raise an exception
        self.mock_socket.recv = MagicMock(side_effect=Exception("Test Exception"))

        # Mock 'exception_note' to return a string
        self.client.exception_note = MagicMock(return_value="Exception Note")

        # Call '_receive_from_server' and expect 'InvalidResponseError'
        with self.assertRaises(InvalidResponseError) as context:
            self.client._receive_from_server(timeout=5.0)

        # Check that the exception message contains 'Test Exception'
        self.assertIn("Test Exception", str(context.exception))


class TestHPLCProcessingSettings(unittest.TestCase):
    def test_to_dict(self):
        settings = HPLCProcessingSettings()
        settings_dict = settings.to_dict()
        expected_dict = {
            "baseline_model": "flatfit",
            "baseline_smoothness": 1.0,
            "min_rel_prominence": 0.01,
            "min_prominence": 1,
            "border_max_peak_cutoff": 0.1,
            "split_threshold": 0.05,
            "explained_threshold": 0.995,
            "peak_model": "Bemg",
            "max_peak_comps": 4,
            "max_peak_distance": 1.0,
            "min_spectrum_correl": 0.99,
            "min_elution_time": 0.4,
            "max_elution_time": 10.0,
            "min_wavelength": 210.0,
            "max_wavelength": 400.0,
            "min_rel_integral": 0.01,
            "relaxe_concs": False,
        }
        self.assertEqual(settings_dict, expected_dict)

    def test_from_dict(self):
        data = {
            "baseline_model": "asls",
            "baseline_smoothness": 2.0,
            "min_rel_prominence": 0.02,
            "min_prominence": 2,
            "border_max_peak_cutoff": 0.2,
            "split_threshold": 0.1,
            "explained_threshold": 0.99,
            "peak_model": "BiGaussian",
            "max_peak_comps": 3,
            "max_peak_distance": 0.5,
            "min_spectrum_correl": 0.98,
            "min_elution_time": 0.5,
            "max_elution_time": 9.0,
            "min_wavelength": 220.0,
            "max_wavelength": 380.0,
            "min_rel_integral": 0.02,
            "relaxe_concs": True,
        }
        settings = HPLCProcessingSettings.from_dict(data)
        self.assertEqual(settings.baseline_model, "asls")
        self.assertEqual(settings.baseline_smoothness, 2.0)
        self.assertEqual(settings.min_rel_prominence, 0.02)
        self.assertEqual(settings.min_prominence, 2)
        self.assertEqual(settings.border_max_peak_cutoff, 0.2)
        self.assertEqual(settings.split_threshold, 0.1)
        self.assertEqual(settings.explained_threshold, 0.99)
        self.assertEqual(settings.peak_model, "BiGaussian")
        self.assertEqual(settings.max_peak_comps, 3)
        self.assertEqual(settings.max_peak_distance, 0.5)
        self.assertEqual(settings.min_spectrum_correl, 0.98)
        self.assertEqual(settings.min_elution_time, 0.5)
        self.assertEqual(settings.max_elution_time, 9.0)
        self.assertEqual(settings.min_wavelength, 220.0)
        self.assertEqual(settings.max_wavelength, 380.0)
        self.assertEqual(settings.min_rel_integral, 0.02)
        self.assertEqual(settings.relaxe_concs, True)


# Run the tests
if __name__ == "__main__":
    unittest.main()

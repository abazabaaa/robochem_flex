"""
File: chromtroller.py
Author: Simone Pilon, Olly Bayley - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: Client to connect to UPLC-DAD-MS device via the Chromtroller server.

"""

from bidict import bidict
import socket
import json
import time
from dataclasses import dataclass, asdict
from threading import Lock
from typing import Any, Dict, Literal, Optional
from omniplatypus.devices.base.device_socket import SocketDevice
from omniplatypus.devices.errors import (
    DeviceTimeoutError,
    InvalidResponseError,
    NoResponseError,
    FailedWriteError,
    HPLCError,
)
from omniplatypus.devices.base.device import (
    DeviceParameter,
    ParameterAccess,
    ParameterEnumValue,
    ParameterValueRun,
)


class SocketDeviceParameter(DeviceParameter):
    """
    Enhanced DeviceParameter with timeout.
    """

    def __init__(
        self,
        name: str,
        access_level: ParameterAccess,
        value_type: type,
        internal_id: Any,
    ) -> None:
        DeviceParameter.__init__(self, name, access_level, value_type, internal_id)
        self.timeout = 10


class ValvePosValue(ParameterEnumValue):
    """Class containing the valid valve positions."""

    _raw_values = bidict({"FILL": "FILL", "INJECT": "INJECT"})

    def serialize(self) -> str:
        return self.value


class ParameterValueRunHplc(ParameterValueRun):
    """Serializable ParameterValueRun class for Parameters which trigger a command with no argument"""

    def serialize(self) -> str:
        return self.value


class HPLCClient(SocketDevice):
    """
    Connect to Chromtroller server via socket interface and access HPLC device.
    """

    def __init__(self):
        SocketDevice.__init__(self)
        self._socket_lock = Lock()
        self.generic_name = "Chromtroller HPLC Client"

        parameter = SocketDeviceParameter(
            name="run_info",
            access_level=ParameterAccess.W,
            value_type=dict,
            internal_id="add_run_info",
        )
        parameter.description = "Info about the run to be logged by the server"
        self.add_parameter(parameter)

        parameter = SocketDeviceParameter(
            name="analysis_parameters",
            access_level=ParameterAccess.W,
            value_type=HPLCProcessingSettings,
            internal_id="set_analysis_parameters",
        )
        parameter.description = "Settings to use for automatic chromatogram processing"
        self.add_parameter(parameter)

        parameter = SocketDeviceParameter(
            name="rt_target",
            access_level=ParameterAccess.W,
            value_type=float,
            internal_id="set_analysis_target",
        )
        parameter.description = (
            "Sets the target retention time for automatic chromatogram processing"
        )
        self.add_parameter(parameter)

        parameter = SocketDeviceParameter(
            name="rt_tolerance",
            access_level=ParameterAccess.W,
            value_type=float,
            internal_id="set_analysis_tolerance",
        )
        parameter.description = (
            "Sets the allowed tolerance in rt for automatic chromatogram processing"
        )
        self.add_parameter(parameter)

        parameter = SocketDeviceParameter(
            name="valve_position",
            access_level=ParameterAccess.W,
            value_type=ValvePosValue,
            internal_id="valve_position",
        )
        parameter.description = "Set valve position. Either 'FILL' or 'INJECT'"
        self.add_parameter(parameter)

        parameter = SocketDeviceParameter(
            name="acquisition",
            access_level=ParameterAccess.W,
            value_type=ParameterValueRunHplc,
            internal_id="start_hplc_run",
        )
        parameter.description = "Starts the acquisition procedure to analyse a sample stored in the sample loop"
        parameter.timeout = 10 * 60
        self.add_parameter(parameter)

        parameter = SocketDeviceParameter(
            name="analysis_result",
            access_level=ParameterAccess.R,
            value_type=dict,
            internal_id="run_data_analysis",
        )
        parameter.description = "Starts the analysis procedure and returns the results"
        parameter.timeout = 60
        self.add_parameter(parameter)

    @property
    def keep_open(self) -> bool:
        """
        Expresses a preference for opening the device for each operation, rather that opening at platform build-time
        and keeping it open (default).

        @return: bool
            True if the device should be kept in the open state when not in use.
        """
        return False

    def _read(self, parameter: SocketDeviceParameter) -> Any:
        """
        Low-level function to read data from the device.


        @param parameter: SocketDeviceParameter
            The parameter which will be accessed.
        @return:
            The value of the parameter read from the device.
        """
        with self._socket_lock:
            close_device = False
            if not self.is_open():
                self.reopen()
                close_device = True
            try:
                # Send the command to ask the server for the info
                self._send_to_server(command=parameter.internal_id)

                # Get the server response
                received_reply = self._get_reply(timeout=parameter.timeout)

                return received_reply

            finally:
                if close_device:
                    self.close()

    def _write(self, parameter: SocketDeviceParameter, value: Any) -> None:
        """
        Low-level function to write data to the device.

        @param parameter: SocketDeviceParameter
            The parameter which will be accessed.
        @param value:
            The value of the parameter to be written to the device.
        """
        with self._socket_lock:
            close_device = False
            if not self.is_open():
                self.reopen()
                close_device = True
            try:
                value = self.convert_to_serialisable(value)

                self._send_to_server(command=parameter.internal_id, data=value)

                # Get the server response
                received_reply = self._get_reply(timeout=parameter.timeout)

            finally:
                if close_device:
                    self.close()

    @staticmethod
    def convert_to_serialisable(value: Any) -> Any:
        """
        Checks the given parameter type and converts to serializable format for socket communication
        Args:
            value: Any
        Returns:
            Any
        """
        try:
            return value.serialize()
        except AttributeError:
            return value

    def _send_to_server(self, command: str, data: Optional[Any] = None) -> None:
        """
        Receive data from the server with a timeout.

        Sending format:
          - First 4 bytes are the message length.
          - Followed by a keyed dict encoded as a JSON
          - dict: {'command': str, 'data': Any}.
        """
        self._check_and_clear_buffer()
        message_dict = {"command": command, "data": data}
        message = json.dumps(message_dict).encode()
        message_length = len(message).to_bytes(4, byteorder="big")
        self._socket.sendall(message_length + message)

    def _check_and_clear_buffer(self) -> None:
        """
        Checks if the buffer is full and if so,logs it and flushes the buffer
        """
        self._socket.setblocking(False)  # Set socket to non-blocking mode
        try:
            residual_data = self._socket.recv(4096)  # Read data in 4KB chunks
            self.log(
                message=f"Residual data remaining in socket buffer: {residual_data}",
                level="warning",
            )
        except BlockingIOError:
            pass  # No data to read
        finally:
            self._socket.setblocking(True)  # Restore socket to blocking mode if needed

    def _receive_from_server(self, timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """
        Receive data from the server with a timeout.

        Receiving format:
          - First 4 bytes are the message length.
          - Followed by a keyed dict encoded as a JSON
          - dict: {'type': str, 'data': Any}.

        Supported message types ('type' key): 'reply', 'info', 'error'.

        :param timeout: Time in seconds to wait for a response before raising a timeout exception.
        :return: Parsed JSON response from the server, or None if no data is received.
        :raises:
            - InvalidReadException if no data is received for the message length.
            - socket.timeout if the server does not respond within the timeout period.
        """
        try:
            # Set the timeout for the socket operations
            self._socket.settimeout(timeout)

            # Read the length of the incoming message (4 bytes)
            message_length_bytes = self._socket.recv(4)
            if not message_length_bytes:
                raise InvalidResponseError(
                    "No data received while attempting to read message length."
                )
            message_length = int.from_bytes(message_length_bytes, byteorder="big")

            # Read the message
            server_byte_data = self._socket.recv(message_length)

            # Decode the message
            if server_byte_data:
                received_dict = json.loads(server_byte_data.decode())
                return received_dict

            return None

        except socket.timeout:
            raise DeviceTimeoutError(
                f"Server did not respond within {timeout} seconds."
            )
        except Exception as error:
            error.add_note(self.exception_note())
            raise InvalidResponseError(f"Error receiving data from server: {error}")
        finally:
            self._socket.settimeout(None)  # Restore to default blocking behavior

    def _get_reply(self, timeout: float = 5.0) -> Any:
        """"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            received_dict = self._receive_from_server(timeout=timeout)
            if received_dict.get("type") == "error":
                self.log(
                    message=f"Error received from server: {received_dict['data']}",
                    level="error",
                )
                raise HPLCError(received_dict["data"])
            elif received_dict.get("type") == "info":
                self.log(
                    message=f"Info received from server: {received_dict['data']}",
                    level="none",
                )
            elif received_dict.get("type") == "reply":
                return received_dict["data"]

        raise NoResponseError(
            f"No command reply received from server within {timeout} seconds."
        )


@dataclass(init=True)
class HPLCProcessingSettings:
    """Collection of all settings required for automatic chromatogram processing in MoccaDataset"""

    baseline_model: Literal["asls", "arpls", "flatfit"] = "flatfit"
    """Name of baseline estimator"""
    baseline_smoothness: float = 1.0
    """Smoothness penalty for baseline"""
    min_rel_prominence: float = 0.01
    """Minimal relative peak height"""
    min_prominence: float = 1
    """Minimal peak height"""
    border_max_peak_cutoff: float = 0.1
    """Maximum relative peak height for peak cutoff"""
    split_threshold: float = 0.05
    """Maximum relative height of minima between peaks to split them"""
    explained_threshold: float = 0.995
    """Minimal R2 to consider peak resolved"""
    peak_model: Literal[
        "BiGaussian", "BiGaussianTailing", "FraserSuzuki", "Bemg"
    ] = "Bemg"
    """Model that describes the peak shape"""
    max_peak_comps: int = 4
    """Maximum number of deconvolved components in single peak"""
    max_peak_distance: float = 1.0
    """Maximum peak distance deviation relative to peak width for one compound"""
    min_spectrum_correl: float = 0.99
    """Minimum correlation of spectra for one compound"""
    min_elution_time: float = 0.4
    """Peaks with maxima before this time will not be considered"""
    max_elution_time: float = 10.0
    """Peaks with maxima after this time will not be considered"""
    min_wavelength: float = 210.0
    """The data will be cropped such that lower wavelengths are not included"""
    max_wavelength: float = 400.0
    """The data will be cropped such that higher wavelengths are not included"""
    min_rel_integral: float = 0.01
    """Minimum integral relative to the largest peak"""
    relaxe_concs: bool = False
    """If True, the concentrations will be relaxed to fit the calibration curve without any peak model"""

    @classmethod
    def from_dict(cls, data: Dict[str, Any]):
        """Creates an HardwareSettings instance from a dictionary."""
        return cls(
            baseline_model=data["baseline_model"],
            baseline_smoothness=float(data["baseline_smoothness"]),
            min_rel_prominence=float(data["min_rel_prominence"]),
            min_prominence=float(data["min_prominence"]),
            border_max_peak_cutoff=float(data["border_max_peak_cutoff"]),
            split_threshold=float(data["split_threshold"]),
            explained_threshold=float(data["explained_threshold"]),
            peak_model=data["peak_model"],
            max_peak_comps=int(data["max_peak_comps"]),
            max_peak_distance=float(data["max_peak_distance"]),
            min_spectrum_correl=float(data["min_spectrum_correl"]),
            min_elution_time=float(data["min_elution_time"]),
            max_elution_time=float(data["max_elution_time"]),
            min_wavelength=float(data["min_wavelength"]),
            max_wavelength=float(data["max_wavelength"]),
            min_rel_integral=float(data["min_rel_integral"]),
            relaxe_concs=bool(data["relaxe_concs"]),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Converts the dataclass to a dictionary."""
        return asdict(self)

    serialize = to_dict  # Alias to allow a .serialize() call

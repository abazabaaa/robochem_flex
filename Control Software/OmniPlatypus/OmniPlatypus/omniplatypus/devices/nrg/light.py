"""
File: light.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: This device controls light intensity of sources compatible with 10v analog inputs. More information and
    schematics at https://github.com/Noel-Research-Group/Array_Devices_Omniplatypus .
"""

from omniplatypus.devices.base.device_arduino import (
    ArduinoParameter,
    ParameterCachingPolicy,
    ParameterAccess,
    ArduinoError,
    ArduinoValueRun,
    ArduinoValueOnOff,
)
from omniplatypus.devices.base.array_arduino import (
    ProxyArduinoDevice,
    ArrayArduinoDevice,
)
from omniplatypus.devices.errors import DeviceError


class LightArrayError(ArduinoError):
    """Lights control array specific error parameter value."""

    _error_types = {
        0: "No error",
        1: "Communication error",
        2: "Light modulation signal error",
    }

    _error_values = {
        2: {
            0: "No error",
        }
    }

    _error_exceptions = {
        2: DeviceError,
    }

    _parameter_error_id = 1
    _no_error_id = 0


class LightSource(ProxyArduinoDevice):
    """
    Control the modulation signal sent to a single light source.
    """

    _parent: "LightArray"

    def __init__(
        self,
        parent: "LightArray",
        serial_interface,
        serial_lock,
        array_index: int,
    ) -> None:
        ProxyArduinoDevice.__init__(
            self,
            parent=parent,
            serial_interface=serial_interface,
            serial_lock=serial_lock,
            array_index=array_index,
        )
        self._array_parameter_begin = 20
        self._array_parameter_period = 10
        self.generic_name = "Light Source"

        base_variable_number = self.base_variable_number()

        parameter = ArduinoParameter(
            name="intensity",
            access_level=ParameterAccess.RW,
            value_type=int,
            internal_id=base_variable_number,
        )
        parameter.description = (
            "Set the light intensity as a percentage of the source capacity"
        )
        parameter.units = "%"
        parameter.min_value = 0
        parameter.max_value = 100
        parameter.caching_policy = ParameterCachingPolicy.TRY
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="calibration",
            access_level=ParameterAccess.RW,
            value_type=float,
            internal_id=base_variable_number + 1,
        )
        parameter.caching_policy = ParameterCachingPolicy.TRY
        parameter.description = (
            "Proportional calibration value used for generating the signal"
        )
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="P",
            access_level=ParameterAccess.RW,
            value_type=float,
            internal_id=base_variable_number + 2,
        )
        parameter.description = (
            "Proportional component for the light control PID signal generator"
        )
        parameter.caching_policy = ParameterCachingPolicy.TRY
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="I",
            access_level=ParameterAccess.RW,
            value_type=float,
            internal_id=base_variable_number + 3,
        )
        parameter.description = (
            "Integral component for the light control PID signal generator"
        )
        parameter.caching_policy = ParameterCachingPolicy.TRY
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="D",
            access_level=ParameterAccess.RW,
            value_type=float,
            internal_id=base_variable_number + 4,
        )
        parameter.description = (
            "Derivative component for the light control PID signal generator"
        )
        parameter.caching_policy = ParameterCachingPolicy.TRY
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="error",
            access_level=ParameterAccess.R,
            value_type=LightArrayError,
            internal_id=9,
        )
        parameter.description = "Error register"
        self._error_parameter = parameter
        self.add_parameter(parameter)


class LightArray(ArrayArduinoDevice):
    """
    Arduino device controlling several analog 10v outputs, each modulating a light source power output.
    The device equips current and voltage monitoring functions to measure electrical power of the sources.
    """

    _proxy_types = {
        "light_source": LightSource,
    }
    _default_proxy = LightSource

    def __init__(self):
        ArrayArduinoDevice.__init__(self)
        self.generic_name = "Lights Array"
        self.default_timeout = 180.0
        self._max_proxies = 4

        parameter = ArduinoParameter(
            name="max_lights",
            access_level=ParameterAccess.R,
            value_type=int,
            internal_id=2,
        )
        parameter.description = (
            "Maximum number of light sources controlled by one array device"
        )
        parameter.caching_policy = ParameterCachingPolicy.TRY
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="enable",
            access_level=ParameterAccess.RW,
            value_type=ArduinoValueOnOff,
            internal_id=3,
        )
        parameter.description = "Enable power source for all light sources connected"
        parameter.caching_policy = ParameterCachingPolicy.TRY
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="current",
            access_level=ParameterAccess.R,
            value_type=int,
            internal_id=4,
        )
        parameter.description = (
            "Total measured electrical current delivered to the light sources"
        )
        parameter.units = " mA"
        parameter.min_value = 0
        parameter.max_value = 5000
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="current_q",
            access_level=ParameterAccess.RW,
            value_type=int,
            internal_id=5,
        )
        parameter.description = "Zero offset calibration value for the measured current"
        parameter.min_value = 0
        parameter.max_value = 1024
        parameter.caching_policy = ParameterCachingPolicy.TRY
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="current_m",
            access_level=ParameterAccess.RW,
            value_type=int,
            internal_id=6,
        )
        parameter.description = (
            "Proportional calibration factor for the measured current"
        )
        parameter.caching_policy = ParameterCachingPolicy.TRY
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="voltage",
            access_level=ParameterAccess.R,
            value_type=float,
            internal_id=7,
        )
        parameter.description = "Measured voltage delivered to the light sources"
        parameter.units = " V"
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="voltage_m",
            access_level=ParameterAccess.RW,
            value_type=float,
            internal_id=8,
        )
        parameter.description = (
            "Proportional calibration factor for the measured voltage"
        )
        parameter.caching_policy = ParameterCachingPolicy.TRY
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="error",
            access_level=ParameterAccess.R,
            value_type=LightArrayError,
            internal_id=9,
        )
        parameter.description = "Error register"
        self._error_parameter = parameter
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="save",
            access_level=ParameterAccess.W,
            value_type=ArduinoValueRun,
            internal_id=10,
        )
        parameter.description = "Save parameters as default"
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="reset",
            access_level=ParameterAccess.W,
            value_type=ArduinoValueRun,
            internal_id=11,
        )
        parameter.description = "Restore parameters to factory"
        self.add_parameter(parameter)

    def open(self, comport: str) -> None:
        """
        Connects to the device. The connection must open before data can be sent or received.

        @param comport: str
            Identifier of the serial port to which the device is connected.

        @raise: serial.SerialException, ValueError
            If connection fails.
        """
        ArrayArduinoDevice.open(self, comport=comport)
        # Suppress non-initialized warning.
        initialization = self._initialized
        self._initialized = True
        try:
            self._max_proxies = self.__getitem__("max_lights")
        finally:
            self._initialized = initialization

    def _turn_off_all(self) -> None:
        """
        Turn all lights off and disable power.
        """
        for source in self._proxies.values():
            source["intensity"] = 0
        self.__setitem__("enable", "OFF")

    def prepare_for_close(self) -> None:
        """
        Ensure the device is in a safe state before closing the connection (and losing control).
        Turns everything off.
        """
        self._turn_off_all()

    def initialize(self) -> None:
        """
        Prepare the device for operation.
        Time intensive operations (movement, zeroing, syncing) are performed here.
        Every subclass should implement this method differently.
        """
        ArrayArduinoDevice.initialize(self)
        self._turn_off_all()

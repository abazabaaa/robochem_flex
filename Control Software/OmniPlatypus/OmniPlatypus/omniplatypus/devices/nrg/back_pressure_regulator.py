"""
File: back_pressure_regulator.py
Author: Simone Pilon - Noël Research Group - 2023
GitHub: https://github.com/simone16

Description: Interface to the Arduino-based back pressure regulator.
"""

from omniplatypus.devices.base.device_arduino import (
    ArduinoDevice,
    ArduinoParameter,
    ArduinoValueOnOff,
    ArduinoValueRun,
)
from omniplatypus.devices.base.device import ParameterAccess


class BackPressureRegulator(ArduinoDevice):
    """
    Handles communication with an adjustable back pressure regulator device.
    This device is documented in detail at https://github.com/Noel-Research-Group/variable_BPR .

    Usage:
        First, connect to the device with
        `self.open('COM4')`

        Then, change parameters of the device with
        `self['parameter_name'] = xyz`

        or read parameters with
        `xyz = self['parameter_name']`

        For a detailed list of available parameters, call
        `self.parameters()`

        For a human-readable list of parameters, call
        `print(self.usage())`

        Finally, close the device with
        `self.close()`
    """

    def __init__(self):
        ArduinoDevice.__init__(self)

        self.generic_name = "Back Pressure Regulator"

        parameter = ArduinoParameter(
            name="enable",
            access_level=ParameterAccess.RW,
            value_type=ArduinoValueOnOff,
            internal_id=2,
        )
        parameter.description = "Enable automatic back pressure adjustment"
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="setpoint",
            access_level=ParameterAccess.RW,
            value_type=float,
            internal_id=3,
        )
        parameter.description = "Pressure setpoint"
        parameter.units = " Bar"
        parameter.max_value = 50.0
        parameter.min_value = 0.0
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="pressure",
            access_level=ParameterAccess.R,
            value_type=float,
            internal_id=4,
        )
        parameter.description = "Pressure measured"
        parameter.units = " Bar"
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="valve", access_level=ParameterAccess.R, value_type=int, internal_id=5
        )
        parameter.description = "Valve position [0-1024]"
        parameter.units = " Bar"
        parameter.max_value = 1024
        parameter.min_value = 0
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="calibration_1",
            access_level=ParameterAccess.W,
            value_type=float,
            internal_id=6,
        )
        parameter.description = "Begin calibration procedure"
        parameter.units = " Bar"
        parameter.max_value = 50.0
        parameter.min_value = 0.0
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="calibration_2",
            access_level=ParameterAccess.W,
            value_type=float,
            internal_id=7,
        )
        parameter.description = "Begin calibration procedure"
        parameter.units = " Bar"
        parameter.max_value = 50.0
        parameter.min_value = 0.0
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="minimum",
            access_level=ParameterAccess.W,
            value_type=ArduinoValueRun,
            internal_id=8,
        )
        parameter.description = "Set current valve position as minimum"
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="maximum",
            access_level=ParameterAccess.W,
            value_type=ArduinoValueRun,
            internal_id=9,
        )
        parameter.description = "Set current valve position as maximum"
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="move", access_level=ParameterAccess.W, value_type=int, internal_id=10
        )
        parameter.description = "Move valve actuator by this many steps"
        self.add_parameter(parameter)

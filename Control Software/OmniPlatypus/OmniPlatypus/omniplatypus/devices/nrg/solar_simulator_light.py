"""
File: mass_flow_controller.py
Author: Simone Pilon - Noël Research Group - 2023
GitHub: https://github.com/simone16

Description: Interface for controlling arduino-based light source system of the solar simulator setup.
"""

from omniplatypus.devices.base.device_arduino import (
    ArduinoDevice,
    ArduinoParameter,
    ArduinoValueOnOff,
)
from omniplatypus.devices.base.device import ParameterAccess


class SolarLightSource(ArduinoDevice):
    """
    Handles communication with the light source of the solar simulator setup.
    The device is documented in detail at https://github.com/Noel-Research-Group/solar_simulator_controller .

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

        self.generic_name = "Solar Light Source"

        parameter = ArduinoParameter(
            name="enable",
            access_level=ParameterAccess.RW,
            value_type=ArduinoValueOnOff,
            internal_id=2,
        )
        parameter.description = "Lights On/_Off state"
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="fan_on",
            access_level=ParameterAccess.RW,
            value_type=ArduinoValueOnOff,
            internal_id=3,
        )
        parameter.description = "Fans On/_Off state"
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="lock_on",
            access_level=ParameterAccess.RW,
            value_type=ArduinoValueOnOff,
            internal_id=4,
        )
        parameter.description = "Manual interface lock On/_Off state"
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="intensity",
            access_level=ParameterAccess.RW,
            value_type=int,
            internal_id=5,
        )
        parameter.description = "Light intensity"
        parameter.units = "%"
        parameter.max_value = 100
        parameter.min_value = 10
        self.add_parameter(parameter)

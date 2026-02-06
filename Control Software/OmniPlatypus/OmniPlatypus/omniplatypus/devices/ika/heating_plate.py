"""
File: heating_plate_IKA.py
Author: Miguel Claros - Noël Research Group - 2024
GitHub: https://github.com/MiguelClaros

Description: This device controls IKA heating plate RCT basic. More information and
    schematics at https://github.com/Noel-Research-Group/Array_Devices_Omniplatypus .
"""

from typing import Any

from omniplatypus.devices.base.device_arduino import (
    ArduinoDevice,
    ArduinoParameter,
    ArduinoValueOnOff,
    ArduinoValueRun,
    ArduinoValue,
)
from omniplatypus.devices.base.device import ParameterAccess


class HeatingPlate(ArduinoDevice):
    READ_DEVICE_NAME = "IN_NAME"
    READ_DEVICE_TYPE = "IN_TYPE"
    READ_ACTUAL_PROCESS_TEMP = "IN_PV_1"
    READ_ACTUAL_SURFACE_TEMP = "IN_PV_2"
    READ_ACTUAL_SPEED = "IN_PV_4"
    # READ_ACTUAL_FLUID_TEMP = "IN_PV_7"  # for double temp probe "PT1000"
    # READ_VISCOSITY_TREND_VALUE = "IN_PV_5"
    READ_PROCESS_TEMP_SETPOINT = "IN_SP_1"
    READ_SURFACE_TEMP_SETPOINT = "IN_SP_2"
    READ_TEMP_LIMIT = "IN_SP_3"  # find the set safe temperature of the plate, the target/set temperature the plate can go to is 50 degrees beneath this
    READ_SPEED_SETPOINT = "IN_SP_4"
    # READ_PROCESS_HEATER_STATUS = "STATUS_1"  # undocumented in manual
    # READ_SURFACE_HEATER_STATUS = "STATUS_2"  # undocumented in manual
    # READ_SHAKER_STATUS = "STATUS_4"  # undocumented in manual
    SET_PROCESS_TEMP_SETPOINT = "OUT_SP_1 "  # requires a value to be appended
    SET_SURFACE_TEMP_SETPOINT = "OUT_SP_2 "
    SET_SPEED_SETPOINT = "OUT_SP_4 "  # requires a value to be appended
    START_THE_HEATER = "START_1"
    STOP_THE_HEATER = "STOP_1"
    START_THE_STIRRING = "START_4"
    STOP_THE_STIRRING = "STOP_4"
    RESET = "RESET"
    SET_OPERATING_MODE_A = "SET_MODE_A"
    SET_OPERATING_MODE_B = "SET_MODE_B"
    SET_OPERATING_MODE_D = "SET_MODE_D"
    SET_WD_SAFETY_LIMIT_TEMPERATURE_WITH_SET_VALUE_ECHO = (
        "OUT_SP_12@"  # requires a value to be appended
    )
    SET_WD_SAFETY_LIMIT_SPEED_WITH_SET_VALUE_ECHO = (
        "OUT_SP_42@"  # requires a value to be appended
    )
    WATCHDOG_MODE_1 = (
        "OUT_WD1@"  # requires a watchdog time (20-1500 s) to be appended to the end
    )
    # This command launches the watchdog function and must be transmitted within the set time.
    # In watchdog mode 1, if event WD1 occurs, the heating and stirring functions are switched off
    #  and ER 2 is displayed
    WATCHDOG_MODE_2 = (
        "OUT_WD2@"  # requires a watchdog time (20-1500 s) to be appended to the end
    )
    # This command launches the watchdog function and must be transmitted within the set time.
    # In watchdog mode 2, if event WD2 occurs, the speed and temperature setpoint are set to their
    # watchdog setpoints.
    # the WD2 event can be reset with the command "OUT_WD2@0", which also stops the watchdog

    def __init__(self):
        ArduinoDevice.__init__(self)

        self.generic_name = "IKA Heating Plate"

        # This is Arduino, but has no custom ID.
        self._clear_parameters()  # remove 'ID' parameter.

        parameter = ArduinoParameter(
            name="name",
            access_level=ParameterAccess.R,
            value_type=str,
            internal_id="IN_NAME",
        )
        parameter.description = "Read device name"
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="actual_sensor_temperature",
            access_level=ParameterAccess.R,
            value_type=float,
            internal_id="IN_PV_1",
        )
        parameter.description = "Read actual external sensor value"
        parameter.float_rounding = 1
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="actual_hotplate_temperature",
            access_level=ParameterAccess.R,
            value_type=float,
            internal_id="IN_PV_2",
        )
        parameter.description = "Read actual hotplate value"
        parameter.float_rounding = 1
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="actual_stirring_speed",
            access_level=ParameterAccess.R,
            value_type=float,
            internal_id="IN_PV_4",
        )
        parameter.description = "Read stirring speed value "
        parameter.float_rounding = 1
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="read_rated_temperature",
            access_level=ParameterAccess.R,
            value_type=str,
            internal_id="IN_SP_1",
        )
        parameter.description = "Read rated temperature value"
        parameter.float_rounding = 1
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="read_rated_safety_temperature",
            access_level=ParameterAccess.R,
            value_type=str,
            internal_id="IN_SP_3",
        )
        parameter.description = "Read rated set safety temperature value"
        parameter.float_rounding = 2
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="read_rated_speed",
            access_level=ParameterAccess.R,
            value_type=float,
            internal_id="IN_SP_4",
        )
        parameter.description = "Read rated speed value"
        parameter.float_rounding = 1
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="set_temperature",
            access_level=ParameterAccess.W,
            value_type=float,
            internal_id="OUT_SP_1",
        )
        parameter.description = "Adjust the set temperature value"
        parameter.float_rounding = 1
        parameter.max_value = 310.0
        parameter.min_value = 0.0
        parameter.units = "°C"
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="set_stirring",
            access_level=ParameterAccess.W,
            value_type=float,
            internal_id="OUT_SP_4",
        )
        parameter.description = "Adjust the set speed value"
        parameter.float_rounding = 1
        parameter.max_value = 1500.0
        parameter.min_value = 0.0
        parameter.units = "rpm"
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="start_heating",
            access_level=ParameterAccess.W,
            value_type=str,
            internal_id="START_1",
        )
        parameter.description = "Start the heater"
        parameter.to_serial_string = lambda x: ""
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="stop_heating",
            access_level=ParameterAccess.W,
            value_type=str,
            internal_id="STOP_1",
        )
        parameter.description = "Stop the heater"
        parameter.to_serial_string = lambda x: ""
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="start_stirring",
            access_level=ParameterAccess.W,
            value_type=str,
            internal_id="START_4",
        )
        parameter.description = "Start the stirring"
        parameter.to_serial_string = lambda x: ""
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="stop_stirring",
            access_level=ParameterAccess.W,
            value_type=str,
            internal_id="STOP_4",
        )
        parameter.description = "Stop the stirring"
        parameter.to_serial_string = lambda x: ""
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="reset",
            access_level=ParameterAccess.R,
            value_type=str,
            internal_id="RESET",
        )
        parameter.description = "Switch to normal operating mode"
        parameter.to_serial_string = lambda x: ""
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="set_operating_mode",
            access_level=ParameterAccess.RW,
            value_type=str,
            internal_id="SET_MODE_",
        )
        parameter.description = "Set operating mode"
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="safety_temperature_echo",
            access_level=ParameterAccess.R,
            value_type=float,
            internal_id="OUT_SP_12@n",
        )
        parameter.description = (
            "Setting WD safety limit temperature with set value echo"
        )
        parameter.float_rounding = 2
        self.add_parameter(parameter)

        parameter = ArduinoParameter(
            name="safety_speed_echo",
            access_level=ParameterAccess.R,
            value_type=float,
            internal_id="OUT_SP_42@n",
        )
        parameter.description = "Setting WD safety limit speed with set value echo"
        parameter.float_rounding = 2
        self.add_parameter(parameter)

        # TODO
        parameter = ArduinoParameter(
            name="watchdog_mode1",
            access_level=ParameterAccess.R,
            value_type=float,
            internal_id="OUT_WD1@m",
        )
        parameter.description = "Watchdog mode 1:"
        parameter.float_rounding = 2
        self.add_parameter(parameter)
        """
        Watchdog mode 1: if event WD1 should occur, the heating and stirring functions are switched off and E 2 is displayed. Set watchdog time to m (20 - 1500)
        seconds, with watchdog time echo. This command launches the watchdog
        function and must be transmitted within the set watchdog time.
        """

        parameter = ArduinoParameter(
            name="watchdog_mode2",
            access_level=ParameterAccess.R,
            value_type=float,
            internal_id="OUT_WD2@m",
        )
        parameter.description = "Watchdog mode 2:"
        parameter.float_rounding = 2
        self.add_parameter(parameter)
        """
        Watchdog mode 2: if event WD2 should occur, the speed target value is
        changed to the WD safety speed limit and the temperature target value is
        changed to the WD safety temperature limit value. The warning WD is displayed. The WD2 event can be reset with OUT_WD2@0 - this also stops the
        watchdog function. Set watchdog time to m (20 - 1500) seconds, with watchdog time echo. This command launches the watchdog function and must be
        transmitted within the set watchdog time.
        """

    def prepare_for_close(self) -> None:
        """
        Ensure the device is in a safe state before closing the connection (and losing control).
        """
        self.__setitem__("stop_heating", True)  # same name as the command

    def initialize(self) -> None:
        """
        Prepare the device for operation.
        Time intensive operations (movement, zeroing, syncing) are performed here.
        """
        ArduinoDevice.initialize(self)

        # TODO put here a code to tun at the beginning.

    def _command_write(self, parameter: ArduinoParameter, value: Any) -> str:
        """
        Computes the string which needs to be sent to write a value to a parameter.

        @param parameter: ArduinoParameter
            The parameter which needs to be written.
        @param value: Any
            The value of the parameter.
        @return: str
            The command string to send via serial.
        """
        if parameter.to_serial_string is not None:
            return (
                parameter.internal_id + " " + parameter.to_serial_string(value) + "\n"
            )
        elif isinstance(value, ArduinoValue):
            return parameter.internal_id + " " + value.to_serial_string() + "\n"
        else:
            return parameter.internal_id + " " + str(value) + "\n"

    def _command_read(self, parameter: ArduinoParameter) -> str:
        """
        Computes the string which needs to be sent to read the value of a parameter.

        @param parameter: ArduinoParameter
            The parameter which needs to be read.
        @return: str
            The command string to send via serial.
        """
        return parameter.internal_id + "\n"

    def _read(self, parameter: ArduinoParameter) -> Any:
        """
        Low-level function to read data from the device.

        @param parameter: DeviceParameter
            The parameter which will be accessed.
        @return:
            The value of the parameter read from the device.
        @raise: serial.SerialException, ValueError
            In case of issues with serial communication.
        """
        response = super()._read(parameter)
        response = response.split()[0]
        return response

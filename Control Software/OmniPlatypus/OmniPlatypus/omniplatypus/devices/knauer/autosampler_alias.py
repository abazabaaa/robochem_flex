"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: This is the device class for the ALIAS autosampler, the one from this publication : DOI: 10.1002/anie.201805632.

"""

import numpy as np

from omniplatypus.devices.base.device import (
    BaseDevice,
    DeviceParameter,
    ParameterAccess,
    ParameterCachingPolicy,
    ParameterEnumValue,
    ParameterValue,
    DeviceSetupOption,
)
from omniplatypus.devices.nrg.sampler import VialHolder
from bidict import bidict
import serial
from typing import Any, List
import time
from threading import Lock


class ValuesTraySet(ParameterEnumValue):
    """
    Defines the values for the tray settings parameter
    """

    _raw_values = bidict(
        {
            "96L96L": "00",
            "96L96H": "01",
            "96L48V": "03",
            "96L12V": "04",
            "96H96L": "10",
            "96H96H": "11",
            "96H48V": "13",
            "96H12V": "14",
            "384384": "22",
            "38448V": "23",
            "38412V": "24",
            "48V96L": "30",
            "48V96H": "31",
            "48V384": "32",
            "48V48V": "33",
            "48V12V": "34",
            "12V96L": "40",
            "12V96H": "41",
            "12V384": "42",
            "12V48V": "43",
            "12V12V": "44",
        }
    )


class ValuesAirSegment(ParameterEnumValue):
    """
    Defines the values for the air segment parameter
    """

    _raw_values = bidict({"OFF": "0", "ON": "1"})


class ValuesPrepMode(ParameterEnumValue):
    """
    Defines the values for the prep mode parameter
    """

    _raw_values = bidict({"OFF": "0", "ON": "1"})


class ValuesTrayCoolingHeater(ParameterEnumValue):
    """
    Defines the values for the tray cooling/heater parameter
    """

    _raw_values = bidict({"OFF": "0", "ON": "1"})


class ValuesSyringeSpeed(ParameterEnumValue):
    """
    Defines the values for the syringe speed parameter
    """

    _raw_values = bidict({"SLOWEST": "1", "MEDIUM": "2", "FASTEST": "3"})


class ValuesInitialWash(ParameterEnumValue):
    """
    Defines the values for the initial wash parameter
    """

    _raw_values = bidict({"START": "1", "STOP": "0"})


class ValuesSyringeValveSwitching(ParameterEnumValue):
    """
    Defines the values for the syringe valve switching parameter
    """

    _raw_values = bidict({"WASH": "1", "NEEDLE": "2", "WASTE": "3"})


class ValuesInjectionValveSwitching(ParameterEnumValue):
    """
    Defines the values for the injection valve switching parameter
    """

    _raw_values = bidict({"LOAD": "1", "INJECT": "0"})


class ValuesNeedleVerticalMovement(ParameterEnumValue):
    """
    Defines the values for the needle vertical movement parameter
    """

    _raw_values = bidict({"UP": "0", "DOWN": "1", "DOWN HARD": "2"})


class ValuesMoveSyringe(ParameterEnumValue):
    """
    Defines the values for the move syringe parameter
    """

    _raw_values = bidict({"HOME": "0", "END": "1", "CHANGE": "2"})


class AliasPosition(ParameterValue):
    """
    Holds the position of the autosampler in the X and Y coordinates (not the Z axis).

    The autosampler needs to move first in the X axis and then in the Y axis to reach a specific position.
    the position is not a value in mm but a value that is respective of the sample holder used.
    position is then something like X = 01L, Y= 01L which means the autosampler is in the first position of the left tray.
    the class also holds the T value which is the tray side, left or right.
    Attributes:
        x: float | None  X axis position.
        y: float | None  Y axis position.
        T: str | None  Tray position.
    """

    coordinate_names = ["X", "Y", "T"]

    value: list[int | None, int | None, str | None]

    @property
    def x(self) -> int | None:
        """X coordinate or None for no change."""
        return self.value[0]

    @x.setter
    def x(self, value: int | None) -> None:
        """Set the value for the x coordinate."""
        self.value[0] = value

    @property
    def y(self) -> int | None:
        """Y coordinate or None for no change."""
        return self.value[1]

    @y.setter
    def y(self, value: int | None) -> None:
        """Set the value for the y coordinate."""
        self.value[1] = value

    @property
    def t(self) -> str | None:
        """Tray side or None for no change."""
        return self.value[2]

    @t.setter
    def t(self, value: str | None) -> None:
        """Set the value for the tray side."""
        self.value[2] = value

    def __init__(
        self,
        X: int | None = None,
        Y: int | None = None,
        T: str | None = None,
        special: str | None = None,
    ):
        """
        creates a position object, specify the X, Y and T values to set the position. All three required,
        alternatively use the special parameter to set one of the following special positions:
        - HOME_X (Wash)
        - HOME_Y
        - HOME_ALL
        - WASTE

        param: X int | None: X axis position. as number of the selected column
        param: Y int | None: Y axis position. as number of the selected row
        param: T str | None: Tray side. 'L' for left, 'R' for right
        param: special str | None: Special position to set. ['HOME_X', 'HOME_Y', 'HOME_ALL', 'WASH', 'WASTE']
        """
        ParameterValue.__init__(self)

        if special is not None:
            match special:
                case "HOME_X":
                    self.value = [-1, None, None]
                case "HOME_Y":
                    self.value = [None, -1, None]
                case "HOME_ALL":
                    self.value = [-1, -1, None]
                case "WASTE":
                    self.value = [None, None, "W"]
                case _:
                    raise ValueError(f"Invalid special position '{special}'.")
        else:
            # check if the values are valid
            if X is None or Y is None or T is None:
                raise ValueError("X, Y and T values must be set.")
            assert isinstance(X, (int, np.integer)), "X must be an integer."
            assert isinstance(Y, (int, np.integer)), "Y must be an integer."
            assert isinstance(T, str), "T must be a string."
            assert T in ["L", "R"], "T must be 'L' or 'R'."
            assert X >= 0, "X must be greater or equal to 0."
            assert Y >= 0, "Y must be greater or equal to 0."

            self.value = [None, None, None]

            for i in range(len(self.value)):
                name = self.coordinate_names[i].lower()
                match name:
                    case "x":
                        self.value[i] = X
                    case "y":
                        self.value[i] = Y
                    case "t":
                        self.value[i] = T

    def to_serial_string(self) -> str:
        """
        Generate a string to transmit the value of this object over serial to the Alias device.

        depending on the command given this may take 1 or 2 commands to set the position. if it's a special command
        it's only 1 (except home_all which is two) if it's a movement command it's always 2.

        @return: tuple(str | None, str | None)
            The tuple of strings to be sent to serial (with necessary command / decorators).
            The first element is the command to move the X axis.
            The second element is the command to move the Y axis.
        """
        base_string = "0xabcd"
        com_code_x = "5136"
        com_code_y = "5111"

        commands = [None, None]

        # Handle X axis movement
        if self.x is not None:
            if self.x == -1:
                commands[0] = base_string.replace("x", "1").replace("abcd", "0000")
            else:
                x_pos = f"{self.x:02}"
                if self.t == "L":
                    commands[0] = (
                        base_string.replace("x", "3")
                        .replace("ab", "01")
                        .replace("cd", x_pos)
                    )
                else:
                    commands[0] = (
                        base_string.replace("x", "3")
                        .replace("ab", "02")
                        .replace("cd", x_pos)
                    )
            commands[0] = com_code_x + commands[0]
        # Handle Y axis movement
        if self.y is not None:
            if self.y == -1:
                commands[1] = base_string.replace("xa", "00").replace("bcd", "107")
            else:
                y_pos = f"{self.y:02}"
                if self.t == "L":
                    commands[1] = (
                        base_string.replace("x", "1")
                        .replace("ab", "00")
                        .replace("cd", y_pos)
                    )
                else:
                    commands[1] = (
                        base_string.replace("x", "2")
                        .replace("ab", "00")
                        .replace("cd", y_pos)
                    )
            commands[1] = com_code_y + commands[1]
        # Handle special positions
        if self.t == "W":
            commands[0] = com_code_x + base_string.replace("x", "2").replace(
                "abcd", "0000"
            )
            commands[1] = None

        # make sure that the if a none value is present it's at the back
        if commands[0] is None:
            commands[0], commands[1] = commands[1], commands[0]
        return tuple(commands)


class AliasVialHolder:
    type = str | None
    name = str | None

    def __init__(self, type: str | None = None, name: str | None = None):
        self.type = type
        self.name = name


class AutosamplerAlias(BaseDevice):
    """
    Class to handle communication with the ALIAS autosampler, a standard serial port communication device. This class provides methods to send specific commands and check responses.

    Usage:
        1. Connect to the device:
           self.open('COM4')

        2. Change parameters:
           self['parameter_name'] = value

        3. Read parameters:
           value = self['parameter_name']

        4. List available parameters:
           self.parameters()

        5. Print a human-readable list of parameters:
           print(self.usage())

        6. Close the device:
           self.close()

    Commands and Responses:
        The autosampler uses a command-response protocol with ASCII strings.

        Command Structure (PC -> Autosampler):
            STXSSAACCCCVVVVVVETX
            - STX: Start of text, ASCII 0x02 (chr(2))
            - SS: Destination address (e.g., 61)
            - AA: Added info code, two-digit hex code (01 to EF), usually set to 01
            - CCCC: Command code (4 digits)
            - VVVVVV: Value (6 digits, padded with zeros)
            - ETX: End of text, ASCII 0x03 (chr(3))

        Response Structure (Autosampler -> PC):
            - ACK: Acknowledge, ASCII 0x06 (chr(6))
            - NACK: Not Acknowledge, ASCII 0x15 (chr(21))
            - NACK0: Not Acknowledge (busy/not ready), ASCII 0x30 (chr(48))

    Follow-up Commands:
        Some commands require follow-up commands:
        - Set Program (SP): STXSSAA100000CCCCETX
        - Set Actual (SA): STXSSAA100100CCCCETX
        - Response: STXSSAACCCCVVVVVVETX (success), NACK/NACK0 (failure)

    Parameters and Commands:
        Parameter Setting (all SP):
            - 0100: Analysis time (0HMMSS)
            - 0107: Loop volume (00xxxx, min 0000, max 5000 µL)
            - 0125: Syringe volume (00xxxx, available values: 00250, 00500, 01000 µL, currently mounted is 500 µL)
            - 0126: Tubing volume (000xxx, min 000, max 999 µL)
            - 0200: Tray settings (0000xy)
                - x: Left side tray
                - y: Right side tray
                - 0: 96 well plate low profile
                - 1: 96 well plate high profile
                - 2: 384 well plate low profile
                - 3: 48 vials
                - 4: 12 vials
                - Response: 0000xy (verify the settings)
            - 0131: Syringe speed (00000x, 1=slowest, 3=fastest)
            - 0130: Needle plunge height (0000xy, value as x.y mm, y can be 0 or 5, min 2.0, max 6.0 mm)
            - 0134: Buffer tube volume (00xxxx, min 0000, max 9999 µL)
            - 0198: Prep Mode (00000x, 0=off, 1=on, turn this off at the start)
            - 0122: Tray cooling/heater (00000x, 0=off, 1=on)
            - 0151: Tray temperature (0000xx, temperature in °C, min 04, max 40)

        Status Checks (no follow-up, sent directly as SA commands):
            - 0152: Status (00xeee)
                - x: Status of the sampler (0: normal, 1: error)
                - eee: Status code (000 -> idle, ready; others -> error or busy)
            - 0154: Errors (000eee)
                - eee: Error code (000 -> no errors; others -> specific error)
            - 0156: Reset (000001, reset the autosampler to homing position, sent as regular command, no follow up)

    Actions (no follow-up, immediate execution):
        - 5105: Injection Valve switching (00000x, 1=load, 0=inject)
            Valve diagram:
                Load (1): connections a-b, c-d, e-f
                      _______________
                     |               |
                     |    b     c    |
                     |   /       \   |
                     | a     X    d  |
                     |               |
                     |    f  -  e    |
                     |_______________|

                Inject (0): connections a-f, b-c, d-e
                      _______________
                     |               |
                     |    b  -   c   |
                     |               |
                     | a           d |
                     |  \         /  |
                     |    f     e    |
                     |_______________|

        - 5130: Initial wash (00000x, 1=start wash, 0=stop wash)
        - 5137: Syringe valve switching (00000x, 1=wash, 2=needle, 3=waste)
            Valve diagram:
                1: Wash
                    A   N   W
                          /
                       S
                2: Needle
                    A   N   W
                        |
                        S
                3: Waste
                    A   N   W
                     \\
                       S
            Note: Typically connect A to the solvent reservoir, N to the needle, and W to a waste bottle.

        - 5111: Move tray (Y axis, 0xabcd)
            Special positions (x=0):
                - 0107: Home (back)
                - 0108: Exchange needle position
                - 0109: Tray front
            Plate positions (x=1 for left, x=2 for right):
                ab = 00
                - 384 well plate: 01-24
                - 96 well plate: 01-12
                - 48 Vial plate: 01-08
                - 12 Vial plate: 01-04

        - 5135: Needle vertical movement (Z axis, 00000x, 0=up, 1=down, 2=down hard)
        - 5136: Needle horizontal movement (X axis, 0xabcd)
            Positions:
                - 1: Wash port
                - 2: Waste port
                - 3: Plate
                    ab
                    - 01: Left
                    - 02: Right
                    cd:
                    - 384 well plate: 01-16
                    - 96 well plate: 01-08
                    - 48 Vial plate: 01-06
                    - 12 Vial plate: 01-03
                - 4: Exchange position

        - 5140: Move syringe (00000x, 0=home, 1=end, 2=change)
        - 5138: Aspirate (000xxx, volume in µL, min 000, max 999)
        - 5139: Dispense (000xxx, volume in µL, min 000, max 999)

        Notes:
        - The autosampler keeps track of the volume in the syringe and will return NACK if the volume is too high or too low.


        """

    parameters = [
        {
            "name": "loop_volume",
            "access_level": ParameterAccess.RW,
            "value_type": int,
            "internal_id": 202,
            "description": "Volume of the sample loop",
            "units": "uL",
            "max_value": 5000,
            "min_value": 0,
            "com_code": "0107",
            "follow_up": "SP",
            "format": "00xxxx",
        },
        {
            "name": "air_segment",
            "access_level": ParameterAccess.RW,
            "value_type": ValuesAirSegment,
            "internal_id": 201,
            "description": "Use air segment before injection",
            "com_code": "0192",
            "follow_up": "SP",
            "format": "00000x",
        },
        {
            "name": "syringe_volume",
            "access_level": ParameterAccess.RW,
            "value_type": int,
            "internal_id": 203,
            "description": "Syringe volume",
            "units": "uL",
            "max_value": 1000,
            "min_value": 250,
            "com_code": "0125",
            "follow_up": "SP",
            "format": "00xxxx",
        },
        {
            "name": "tubing_volume",
            "access_level": ParameterAccess.RW,
            "value_type": int,
            "internal_id": 204,
            "description": "Tubing volume",
            "units": "uL",
            "max_value": 999,
            "min_value": 0,
            "com_code": "0126",
            "follow_up": "SP",
            "format": "000xxx",
        },
        {
            "name": "tray_settings",
            "access_level": ParameterAccess.RW,
            "value_type": ValuesTraySet,
            "internal_id": 205,
            "description": "Tray settings",
            "com_code": "0200",
            "follow_up": "SP",
            "format": "0000xx",
        },
        {
            "name": "syringe_speed",
            "access_level": ParameterAccess.RW,
            "value_type": ValuesSyringeSpeed,
            "internal_id": 206,
            "description": "Syringe speed",
            "com_code": "0131",
            "follow_up": "SP",
            "format": "00000x",
        },
        {
            "name": "needle_plunge_height",
            "access_level": ParameterAccess.RW,
            "value_type": float,
            "internal_id": 207,
            "description": "Needle plunge height",
            "units": "mm",
            "max_value": 6.0,
            "min_value": 2.0,
            "com_code": "0130",
            "follow_up": "SP",
            "format": "0000xx",
        },
        {
            "name": "buffer_tube_volume",
            "access_level": ParameterAccess.RW,
            "value_type": int,
            "internal_id": 208,
            "description": "Buffer tube volume",
            "units": "uL",
            "max_value": 9999,
            "min_value": 0,
            "com_code": "0134",
            "follow_up": "SP",
            "format": "00xxxx",
        },
        {
            "name": "prep_mode",
            "access_level": ParameterAccess.RW,
            "value_type": ValuesPrepMode,
            "internal_id": 209,
            "description": "Prep mode",
            "com_code": "0198",
            "follow_up": "SP",
            "format": "00000x",
        },
        {
            "name": "tray_cooling_heater",
            "access_level": ParameterAccess.RW,
            "value_type": ValuesTrayCoolingHeater,
            "internal_id": 210,
            "description": "Tray cooling/heater",
            "com_code": "0122",
            "follow_up": "SP",
            "format": "00000x",
        },
        {
            "name": "tray_temperature",
            "access_level": ParameterAccess.RW,
            "value_type": int,
            "internal_id": 211,
            "description": "Tray temperature",
            "units": "°C",
            "max_value": 40,
            "min_value": 4,
            "com_code": "0151",
            "follow_up": "SP",
            "format": "0000xx",
        },
        {
            "name": "status",
            "access_level": ParameterAccess.R,
            "value_type": int,
            "internal_id": 301,
            "description": "Status",
            "com_code": "0152",
            "follow_up": None,
            "format": "00cccc",
        },
        {
            "name": "errors",
            "access_level": ParameterAccess.R,
            "value_type": int,
            "internal_id": 302,
            "description": "Errors",
            "com_code": "0155",
            "follow_up": None,
            "format": "00cccc",
        },
        {
            "name": "reset",
            "access_level": ParameterAccess.W,
            "value_type": int,
            "internal_id": 303,
            "description": "Reset",
            "com_code": "0156",
            "follow_up": None,
            "format": "000001",
        },
        {
            "name": "injection_valve_switching",
            "access_level": ParameterAccess.RW,
            "value_type": ValuesInjectionValveSwitching,
            "internal_id": 501,
            "description": "Injection Valve switching",
            "com_code": "5105",
            "follow_up": None,
            "format": "00000x",
        },
        {
            "name": "initial_wash",
            "access_level": ParameterAccess.W,
            "value_type": ValuesInitialWash,
            "internal_id": 502,
            "description": "Initial wash",
            "com_code": "5130",
            "follow_up": None,
            "format": "00000x",
        },
        {
            "name": "syringe_valve_switching",
            "access_level": ParameterAccess.W,
            "value_type": ValuesSyringeValveSwitching,
            "internal_id": 503,
            "description": "Syringe valve switching",
            "com_code": "5137",
            "follow_up": None,
            "format": "00000x",
        },
        {
            "name": "position",
            "access_level": ParameterAccess.RW,
            "value_type": AliasPosition,
            "internal_id": 504,
            "description": "Position",
            "units": None,
            "com_code": "5111",
            "follow_up": None,
            "format": ("0xabcd", "0xabcd"),
        },
        {
            "name": "needle_vertical_movement",
            "access_level": ParameterAccess.W,
            "value_type": ValuesNeedleVerticalMovement,
            "internal_id": 505,
            "description": "Needle vertical movement",
            "com_code": "5135",
            "follow_up": None,
            "format": "00000x",
        },
        {
            "name": "move_syringe",
            "access_level": ParameterAccess.W,
            "value_type": ValuesMoveSyringe,
            "internal_id": 506,
            "description": "Move syringe",
            "com_code": "5140",
            "follow_up": None,
            "format": "00000x",
        },
        {
            "name": "aspirate",
            "access_level": ParameterAccess.W,
            "value_type": int,
            "internal_id": 507,
            "description": "Aspirate",
            "units": "uL",
            "max_value": 999,
            "min_value": 0,
            "com_code": "5138",
            "follow_up": None,
            "format": "000xxx",
        },
        {
            "name": "dispense",
            "access_level": ParameterAccess.W,
            "value_type": int,
            "internal_id": 508,
            "description": "Dispense",
            "units": "uL",
            "max_value": 999,
            "min_value": 0,
            "com_code": "5139",
            "follow_up": None,
            "format": "000xxx",
        },
    ]
    _actions = [
        "injection_valve_switching",
        "initial_wash",
        "syringe_valve_switching",
        "position",
        "needle_vertical_movement",
        "move_syringe",
        "aspirate",
        "dispense",
    ]
    locations: dict[str, [VialHolder]]
    _internal_buffer: str = ""

    def __init__(self):
        super().__init__()

        self.generic_name = "Autosampler"
        self.meters = {}
        for param in self.parameters:
            meter = DeviceParameter(
                name=param["name"],
                access_level=param["access_level"],
                value_type=param["value_type"],
                internal_id=param["internal_id"],
            )
            meter.description = param["description"]
            if param["access_level"] == ParameterAccess.RW:
                meter.caching_policy = ParameterCachingPolicy.ALWAYS
            elif param["access_level"] == ParameterAccess.R:
                meter.caching_policy = ParameterCachingPolicy.NEVER
            elif param["access_level"] == ParameterAccess.W:
                meter.caching_policy = ParameterCachingPolicy.NEVER

            if "max_value" in param:
                meter.max_value = param["max_value"]
            if "min_value" in param:
                meter.min_value = param["min_value"]

            # add extra values:
            if "format" in param:
                meter.format = param["format"]
            if "follow_up" in param:
                meter.follow_up = param["follow_up"]
            if "com_code" in param:
                meter.com_code = param["com_code"]
            self.meters[meter.name] = meter
            self.add_parameter(meter)

        self._serial_iface = serial.Serial()
        self._serial_iface.baudrate = 9600
        self._serial_iface.timeout = 1
        self._timeout = 2
        self.locations = {}
        locations_option = DeviceSetupOption(
            name="locations", handler=self.setup_locations
        )
        locations_option.value_type = dict
        locations_option.description = """
                    Locations specify special positions for the handler, such as vial holders and injection ports.
                    This option must be specified in the config file as a dict of location_name -> data pairs.
                    Data is a dictionary itself, which must specify:
                     - type: "type"
                        At the moment,"holder" is supported.
                     - position: is not used
                     - shape: "shape"
                        This is only for holders. It must match a holder type from the 'sample_holder_types' config file.
                    Example:
                    "setup": {
                      "locations": {
                        "holder_right": {
                          "type": "holder",
                          "shape": "vial_GC_4ml_4x4"
                        },
                    """
        self.add_setup_option(locations_option)

    def setup_locations(self, locations: dict) -> None:
        """
        Setup handler for 'locations'.
        Enables injection of samples into a flow system by specifying the injection port location.

        @param locations: dict
            Dict of location_name -> data pairs.
            data is a dictionary itself, which must specify:
             - type: "type"
                At the moment, "injection_port" and "holder" are supported.
             - position: {"x": "10.0mm", "y": "10.0mm" (, "z": "10.0mm")}
                Use SI units: mm, cm ecc...
                z position is only used for injection port, it indicates the depth to reach for injection.
             - shape: "shape"
                This is only for holders. It must match a holder type from the 'sample_holder_types' config file.
             - injection_valve_position: "OFF"
                If specified, this is the position that the auxiliary switchvalve of the pump connected to the sampler
                will be set to when injecting in the injection port.
                This is only used by the injection port locations.
                By default, the switchvalve is not moved.
        """
        try:
            for location_name, location_data in locations.items():
                location = None
                if location_data["type"] == "holder":
                    type = location_data["shape"]
                    location = AliasVialHolder(name=location_name, type=type)
                self.locations[location_name] = location
        except (KeyError, ValueError) as e:
            e.add_note(
                "Make sure the configuration files correctly specify the locations."
            )
            e.add_note(self.exception_note())
            self.log(e)
            raise

    def initialize(self) -> None:
        """Set the autosampler to default values"""

        super().initialize()

        default_parameters = {
            "prep_mode": "OFF",
            "air_segment": "OFF",
            "loop_volume": 800,
            "syringe_volume": 500,
            "tubing_volume": 15,
            "tray_settings": "48V48V",
            "needle_plunge_height": 2.0,
            "buffer_tube_volume": 2000,
        }

        for param, value in default_parameters.items():
            self[param] = value

    def open(self, comport: str) -> None:
        """Opens connection to the serial port"""
        if self.is_open():
            self.close()

            time.sleep(1)
        self._serial_iface.port = comport
        self._serial_iface.open()
        if self._serial_iface.is_open:
            self.log("Connection to the autosampler opened", level="ok")
        else:
            self.log("Failed to open connection to the autosampler", level="error")
            raise ConnectionError("Failed to open connection to the autosampler")

    def is_open(self) -> bool:
        """Returns True if the serial port is open"""
        return self._serial_iface.isOpen()

    def close(self) -> None:
        """Closes the serial port"""
        self._serial_iface.close()
        self._serial_iface = serial.Serial()
        self._serial_iface.baudrate = 9600
        self._serial_iface.timeout = 1
        self.log("Connection to the autosampler closed", level="ok")

    def _construct_message(
        self, parameter: DeviceParameter, value
    ) -> tuple[bytes, bytes]:
        """Constructs the message to send to the autosampler following a specific protocol."""
        special_cases = {
            "position": self._construct_position_message,
            "reset": lambda p, v: self._construct_special_message(p, "000001"),
            "errors": lambda p, v: self._construct_special_message(p, "000155", "1001"),
            "status": lambda p, v: self._construct_special_message(p, "000152", "1001"),
        }

        if parameter.name in special_cases:
            return special_cases[parameter.name](parameter, value)

        com_code = parameter.com_code
        message = list(parameter.format)

        value_str = str(value).zfill(len(message))

        for i, char in enumerate(message):
            if char == "x":
                message[i] = value_str[i]
        if parameter.name in self._actions:
            # no follow up for actions:
            return (chr(2) + "6101" + com_code + "".join(message) + chr(3)).encode(
                "ascii"
            ), None

        follow_up_message = (
            "6101" + ("100000" if parameter.follow_up == "SP" else "100100") + com_code
        )
        message = "".join(message)
        return (chr(2) + "6101" + com_code + message + chr(3)).encode("ascii"), (
            chr(2) + follow_up_message + chr(3)
        ).encode("ascii")

    def _construct_position_message(
        self, parameter: DeviceParameter, value
    ) -> tuple[bytes, bytes]:
        message = list(value.to_serial_string())
        M1 = None
        M2 = None
        if message[0] is not None:
            M1 = str.encode(chr(2) + "6101" + message[0] + chr(3))
        if message[1] is not None:
            M2 = str.encode(chr(2) + "6101" + message[1] + chr(3))

        return M1, M2

    def _construct_special_message(
        self, parameter: DeviceParameter, value: str, com_code: str = None
    ) -> tuple[bytes, None]:
        com_code = com_code or parameter.com_code
        return str.encode(chr(2) + "6101" + com_code + value + chr(3)), None

    def _decode_response(self, response: str) -> str:
        response_map = {chr(6): "Ack", chr(21): "Nack", chr(8): "Nack0"}
        decoded_response = response_map.get(response, response)
        self.log(f"Decoded response: {decoded_response}", level="ok")
        return decoded_response

    def _send_command(self, message: bytes) -> str:
        """Sends the command to the autosampler"""
        time.sleep(0.1)
        self._serial_iface.write(message)
        time.sleep(0.1)
        self.log(f"Sent command: {message}", level="ok")
        return self._wait_for_answer()

    def _flush_buffer_in(self, whole_line: bool = False) -> str:
        """
        Read everything from the incoming buffer and return it as a string.
        If whole_line is True, only one line is returned and the rest is stored internally and returned at the
        next call. If less than one line is available, nothing will be returned until one line is complete or the
        function is called with whole_line=False.

        @param whole_line: bool = False
            If True, returns the buffer only once a whole line has been read.
            Requires multiple calls until the end of line character is received.
        @return: str
            The contents of the incoming data buffer.
        """
        bytes_to_read = self._serial_iface.in_waiting
        if bytes_to_read > 0:
            self._internal_buffer += self._serial_iface.read(bytes_to_read).decode(
                "ascii"
            )
        result = ""
        if whole_line:
            if "\n" in self._internal_buffer:
                result, self._internal_buffer = self._internal_buffer.split("\n", 1)
                result = result.rstrip("\r")
        else:
            result = self._internal_buffer
            self._internal_buffer = ""
        return result

    def _wait_for_answer(self) -> str:
        """Waits for the answer from the autosampler"""
        start_time = time.time()
        while True:
            answer = self._flush_buffer_in(whole_line=False)
            if answer:
                return self._decode_response(answer)
            if time.time() - start_time > self._timeout:
                self.log("Nothing returned from the autosampler.", level="warning")
                return "nothing returned"
            time.sleep(0.1)

    def _check_status(self, ret: bool = False) -> bool:
        """Check the status of the autosampler
        parameter ret: bool: if True returns the status code
        """
        message, _ = self._construct_message(self.meters["status"], None)
        answer = self._send_command(message)
        if "000000" in answer:
            self.log(f"Status code: {answer}", level="ok")
            return True if not ret else answer
        elif "nothing returned" in answer:
            self.log("No status returned", level="error")
            raise ConnectionError("No status returned")
        self.log(f"Status code: {answer}", level="error")
        return False if not ret else answer

    def _check_errors(self, ret: bool = False) -> bool:
        """Check the errors of the autosampler"""
        message, _ = self._construct_message(self.meters["errors"], None)
        answer = self._send_command(message)
        if "000000" in answer:
            self.log("No errors", level="ok")
            return False if not ret else answer
        self.log(f"Error code: {answer}", level="error")
        return True if not ret else answer

    def _reset(self):
        """Resets the autosampler to the home position"""
        message, _ = self._construct_message(self.meters["reset"], None)
        answer = self._send_command(message)
        self.log("Resetting the autosampler", level="ok")

    def _check_answer(self, answer: str, follow_up: bool = False) -> None:
        """Check the answer from the autosampler"""
        if follow_up and (not answer or answer == "nothing returned"):
            self.log("Follow-up command not acknowledged", level="error")
            raise ConnectionError("Follow-up command not acknowledged")
        elif not follow_up and (not answer or answer == "nothing returned"):
            self.log("Nothing returned from the autosampler", level="warning")
            raise ConnectionError("Nothing returned from the autosampler")

        if answer in ["Nack", "Nack0"]:
            self.log("Command not acknowledged", level="error")
        elif answer == "Ack":
            self.log("Command acknowledged", level="ok")
        else:
            self.log("Unknown response", level="error")

    def _reset_serial(self):
        """since we have seen that the serial comm is being a little bitch, if we get the nothing returned
        error, we reset the comm"""
        comport = self._serial_iface.port
        self.open(comport=comport)

    def _process_message(self, messages: List[str], parameter: DeviceParameter):
        while messages:
            message = messages.pop(0)
            if message is None:
                continue

            retries = 5
            while retries > 0:
                try:
                    with Lock():
                        answer = self._send_command(message)
                        self._check_answer(answer)

                        if parameter.name in self._actions:
                            time.sleep(5)
                            while not self._check_status():
                                time.sleep(5)
                        break  # Break the retry loop if successful

                except ConnectionError:
                    self.log("Connection error, retrying the command", level="error")
                    retries -= 1
                    if retries == 0:
                        raise ConnectionError(
                            "Maximum retries reached. Command failed."
                        )
                    time.sleep(10)  # Wait before retrying

                except Exception as e:
                    self.log(f"Error: {e}", level="error")
                    raise e

    def _write(self, parameter: DeviceParameter, value) -> None:
        """Writes the command to the serial port"""
        time.sleep(0.5)
        try:
            while not self._check_status():
                time.sleep(1)
        except ConnectionError:
            self.log("Connection error, retrying to check status", level="error")
            retries = 5
            while retries > 0:
                self._reset_serial()
                time.sleep(5)
                retries -= 1
                try:
                    if self._check_status():
                        break
                except ConnectionError:
                    if retries == 0:
                        raise BrokenPipeError(
                            "Even after resetting, communication is not working."
                        )

        # Check the type of the value, if it's one of the bidict values we need to convert it to the raw value
        if isinstance(value, ParameterEnumValue):
            value = value.value

        if isinstance(value, float):
            value = int(value * 10)

        # If the parameter name is position check if the needle is down or up, and if it's down pull it up
        if parameter.name == "position" and self.meters[
            "needle_vertical_movement"
        ].last_known_value not in ["UP", "1"]:
            self.log("Needle is down, pulling it up", level="ok")
            self["needle_vertical_movement"] = "UP"

        message_tuple = self._construct_message(parameter, value)
        messages = list(message_tuple)
        self._process_message(messages=messages, parameter=parameter)

    def _read(self, parameter: DeviceParameter) -> Any:
        """Reads the value of a paramter, only valid for the status and errors parameters"""
        if parameter.name == "status":
            status = self._check_status(ret=True)
            return status
        if parameter.name == "errors":
            errors = self._check_errors(ret=True)
            return errors

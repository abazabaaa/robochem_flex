"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: code for the hplc pump (find make and model)

"""

from omniplatypus.devices.base.device import (
    BaseDevice,
    DeviceParameter,
    ParameterAccess,
    ParameterCachingPolicy,
    ParameterEnumValue,
    ParameterValue,
)

from bidict import bidict
import serial
from typing import Any
import time


class pump_onoff_state(ParameterEnumValue):
    _raw_values = bidict({"ON": "ON", "OFF": "ON"})


class AzuraHPLC(BaseDevice):
    """HPLC pump from azura. Very minimal interface, you can set flowrate and turn it on and off

    for a specfic amount of volume refer to pump_volume_azura unit task.

    """

    parameters = [
        {
            "name": "flowrate",
            "access": ParameterAccess.RW,
            "value_type": int,
            "unit": "ul/min",
            "description": "flowrate of the pump",
            "max_value": 5000,
            "min_value": -1,
            "internal_id": 201,
        },
        {
            "name": "state",
            "access": ParameterAccess.RW,
            "value_type": pump_onoff_state,
            "description": "turn the pump on or off",
            "internal_id": 202,
        },
    ]

    def __init__(self):
        super().__init__()
        self.generic_name = "AzuraHPLC"
        for param in self.parameters:
            meter = DeviceParameter(
                name=param["name"],
                access_level=param["access"],
                value_type=param["value_type"],
                internal_id=param["internal_id"],
            )
            meter.description = param["description"]
            meter.caching_policy = ParameterCachingPolicy.ALWAYS

            self.add_parameter(meter)

        self._serial_iface = serial.Serial()
        self._serial_iface.baudrate = 9600
        self._serial_iface.timeout = 1
        self._timeout = 10

    def initialize(self) -> None:
        """initialises the pump to default values"""
        self._send_command(str.encode("REMOTE\r"))
        self["flowrate"] = 1000
        self["state"] = "OFF"

    def open(self, comport: str) -> None:
        """opens the connection to the serial port"""
        if self.is_open():
            self.close()

        self._serial_iface.port = comport
        self._serial_iface.open()
        if self._serial_iface.is_open:
            self.log("HPLC pump connected", level="ok")
        else:
            self.log("HPLC pump not connected", level="error")
            raise ConnectionError("HPLC pump not connected")

    def is_open(self) -> bool:
        """checks if the serial port is open"""
        return self._serial_iface.is_open

    def close(self) -> None:
        """closes the serial port"""
        if self.is_open():
            self._serial_iface.close()
            self.log("HPLC pump disconnected", level="ok")

    def _construct_message(
        self, parameter: DeviceParameter, value: Any
    ) -> tuple(str, str):
        """constructs the message to send to the pump"""
        if parameter.internal_id == 201:
            message = f"FLOW {value}.00\r"
        elif parameter.internal_id == 202:
            message = f"{value}\r"

        return str.encode(message)

    def _send_command(self, message: bytes) -> str:
        """sends the command to the pump"""
        self._serial_iface.write(message)
        time.sleep(0.1)
        return self._wait_for_answer()

    def _wait_for_answer(self) -> str:
        """waits for the pump to answer"""
        start_time = time.time()
        while True:
            answer = self._serial_iface.readline()
            if len(answer) > 0:
                answer_decoded = answer.decode()
                return self._decode_response(answer_decoded)
            if time.time() - start_time > self._timeout:
                self.log("Nothing returned from the HPLC pump.", level="warning")
                return "nothing returned"
            time.sleep(0.1)
            self.log("No answer from the HPLC pump yet.", level="warning")

    def _decode_response(self, response: str) -> str:
        """decondes the pump's response"""
        if "OK" in response:
            decoded_response = "Ack"
        else:
            decoded_response = response

        return decoded_response

    def _check_answer(self, answer: str) -> bool:
        """checks if the pump answered correctly"""
        if "Ack" in answer:
            self.log("Command executed successfully", level="ok")
        else:
            self.log("Command not executed", level="error")

    def _write(self, parameter: DeviceParameter, value: Any) -> None:
        """writes the value to the pump"""
        message = self._construct_message(parameter, value)
        answer = self._send_command(message)
        self._check_answer(answer)

    def _read(self, parameter: DeviceParameter) -> Any:
        pass

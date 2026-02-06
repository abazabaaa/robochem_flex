"""
File: platform.py
Author: Simone Pilon - Noël Research Group - 2023
        Ronald Kortekaas - University of Amsterdam (TC) - 2023
GitHub: https://github.com/simone16

Description:  Extends device.py with additional behaviour to control devices which use the Modbus communication
    protocol.
"""

import struct
from enum import Enum, Flag
from pymodbus.client import ModbusSerialClient
from pymodbus.framer.rtu_framer import ModbusRtuFramer
from pymodbus.pdu import ModbusResponse
from omniplatypus.devices.base.device import (
    BaseDevice,
    DeviceParameter,
    ParameterAccess,
)
from threading import Lock


# Enable debugging
# pymodbus_apply_logging_config('DEBUG')


class DataSize(Enum):
    BYTE = 8
    WORD = 16


class StructSizeUnsigned(Enum):
    B = 1
    H = 2
    L = 4
    Q = 8


class RegisterType(Enum):
    HOLDING = 1
    COMMAND = 2
    COMMAND_ACTION = 3


class ModbusParameter(DeviceParameter):
    """
    Holds information on a single parameter monitored or affected by a Modbus device.
    This extends DeviceParameter with specific Modbus data.

    Attributes:
        register_type   Type of Modbus register.

        internal_id     identifier used internally by the device class to access the parameter.
                        For type 'HOLDING' this is the address.
                        For type 'COMMAND(_ACTION)' this is the command_code.
        internal_type   type used internally for the parameter.
        data_length     number of units transmitted for this parameter (bytes/2).
    """

    internal_type: type
    data_length: int
    register_type: RegisterType

    def __init__(
        self,
        name: str,
        access_level: ParameterAccess,
        value_type: type,
        internal_type: type,
        register_type: RegisterType,
        internal_id: int,
    ):
        DeviceParameter.__init__(self, name, access_level, value_type, internal_id)

        self.internal_type = internal_type
        self.data_length = 1
        self.register_type = register_type


class ModbusDevice(BaseDevice):
    """Handles communication with a device via the Modbus protocol."""

    _parameter_type = ModbusParameter

    def __init__(self, slave: int = 0) -> None:
        BaseDevice.__init__(self)

        self.slave = slave
        self._client = None
        self._connected = False

        self._thread_lock = Lock()

    def open(self, comport: str) -> None:
        """
        Connects to the device. The connection must open before data can be sent or received.

        @param comport: str
            Identifier of the serial port to which the device is connected.
        """
        self.close()
        with self._thread_lock:
            self._client = ModbusSerialClient(
                framer=ModbusRtuFramer,
                port=comport,
                baudrate=9200,
                timeout=10,
                parity="E",
                stopbits=1,
                bytesize=8,
            )
            self._connected = self._client.connect()
            if self._connected:
                self.log(f"Connected on port '{comport}'.", level="ok")
            else:
                self.log(f"Failed to connect on port '{comport}'.", level="error")

    def is_open(self) -> bool:
        """
        Check whether the connection is open.

        @return: bool
            True if the connection is open.
        """
        return self._connected

    def close(self) -> None:
        """Disconnects the device."""
        with self._thread_lock:
            self._client.close()
            self._connected = False
        self.log("Disconnected.", level="ok")

    @staticmethod
    def _hton(words: list, insize: int, outsize: int) -> int:
        val = [0] * (insize - len(words)) + words
        val = struct.pack(f"<{len(val)}{StructSizeUnsigned(insize).name}", *val)
        val = struct.unpack(f">{StructSizeUnsigned(outsize * 2).name}", val)[0]
        return val

    @staticmethod
    def _ntoh(words: list, insize: int, outsize: int) -> int:
        val = [0] * (insize - len(words)) + words
        val = struct.pack(f"<{len(val)}{StructSizeUnsigned(insize).name}", *val)
        val = struct.unpack(f"<{StructSizeUnsigned(outsize * 2).name}", val)[0]
        return val

    def _read_modbus(self, addr: int, count: int = 1) -> ModbusResponse | None:
        return self._client.read_holding_registers(
            address=addr, count=count, slave=self.slave
        )

    def _read(self, parameter: ModbusParameter):
        """
        Low-level function to read data from the device.

        @param parameter: ModbusParameter
            The parameter which will be accessed.
        @return:
            The value of the parameter read from the device.
        """
        modbus_response = None
        value = None

        if parameter.internal_type is int:
            if parameter.data_length == 1:
                modbus_response = self._read_modbus(parameter.internal_id)
                value = modbus_response.registers[0]
            elif parameter.data_length == 2:
                modbus_response = self._read_modbus(
                    parameter.internal_id, parameter.data_length
                )
                value = ModbusDevice._ntoh(
                    modbus_response.registers, parameter.data_length, 1
                )
        elif parameter.internal_type is bin:
            # I don't think this is actually used or needed.
            if parameter.data_length == 1:
                modbus_response = self._read_modbus(parameter.internal_id)
            value = bin(modbus_response.registers[0])
        elif parameter.internal_type is Flag:
            modbus_response = self._read_modbus(
                parameter.internal_id, parameter.data_length
            )
            value = ModbusDevice._ntoh(
                modbus_response.registers, 2, parameter.data_length
            )
        return value

    def _write(self, parameter: ModbusParameter, value):
        """
        Low-level function to write data to the device.

        @param parameter: DeviceParameter
            The parameter which will be accessed.
        @param value:
            The value of the parameter to be written to the device.
        """
        if parameter.register_type == RegisterType.HOLDING:
            modbus_response = self._client.write_registers(
                address=parameter.internal_id, values=value, slave=self.slave
            )
        else:
            if parameter.register_type == RegisterType.COMMAND_ACTION:
                # In this case the value is ignored (no value is sent)
                # however, the calling __setitem__ function will check that a valid value is provided
                # A dummy value should be set to ensure the call is not a mistake.
                value = ModbusDevice._hton([parameter.internal_id, 0], 1, 1)
            else:
                value = ModbusDevice._hton([parameter.internal_id, value], 1, 1)
            modbus_response = self._client.write_register(
                address=0x0000, value=value, slave=self.slave
            )
        if modbus_response.isError():
            raise modbus_response

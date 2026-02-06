"""
File: test_device_phase_sensor.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: unit test for Phase sensor array using mock interface.
"""

import unittest

from omniplatypus.devices.platform import Platform
from omniplatypus.devices.nrg.phase_sensor import (
    PhaseSensorArray,
    PhaseSensor,
)
from omniplatypus.devices import testing


class MockArray(PhaseSensorArray):
    @property
    def serial_iface(self):
        return self._serial_iface

    def open(self, comport: str) -> None:
        dummy_serial_iface = testing.DummySerial()
        self._serial_iface = dummy_serial_iface
        self._serial_iface.open()
        dummy_serial_iface.reply("", "4", "", "0-0")
        self._max_proxies = self.__getitem__("max_sensors")


class TestPhaseArray(unittest.TestCase):
    def setUp(self):
        self.platform = Platform()
        self.platform.device_constructors["phase_sensor_array"] = MockArray
        self.platform._known_devices = {"phase_array_op1": "COM1"}
        self.platform.build(
            platform_name="Perry",
            devices=["Phase_Sensor_Array_1"],
            update_docs=False,
            open_gui=False,
        )
        self.device = self.platform["Phase_Sensor_Array_1"]
        self.subdevice = self.platform["ps_handler_out"]
        self.device: MockArray
        self.dummy_serial_iface = self.device.serial_iface

    def tearDown(self):
        pass

    def test_write(self):
        self.dummy_serial_iface.reply("", "", "0-0\r\n")
        self.device["save"] = "run"

    def test_read(self):
        self.dummy_serial_iface.reply("", "0-0\r\n")
        d = self.device["error"]

    def test_subdevice(self):
        self.assertTrue(isinstance(self.subdevice, PhaseSensor))

        self.dummy_serial_iface.reply("", "128\r\n", "", "0-0\r\n")
        a = self.subdevice["analog"]

        self.dummy_serial_iface.reply("", "", "0-0\r\n")
        self.subdevice["calibrate"] = "run"


if __name__ == "__main__":
    unittest.main()

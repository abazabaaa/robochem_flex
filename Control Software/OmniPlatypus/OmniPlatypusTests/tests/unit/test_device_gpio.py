"""
File: test_device_gpio.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: unit test for Gpio array using mock interface.
"""

import unittest

from omniplatypus.devices.platform import Platform
from omniplatypus.devices.nrg.gpio import GpioArray, SolenoidValve
from omniplatypus.devices import testing


class MockArray(GpioArray):
    @property
    def serial_iface(self):
        return self._serial_iface

    def open(self, comport: str) -> None:
        dummy_serial_iface = testing.DummySerial()
        dummy_serial_iface.read_delay = 0.2
        self._serial_iface = dummy_serial_iface
        self._serial_iface.open()
        self._initialized = True

        dummy_serial_iface.reply("", "12", "", "0-0")
        dummy_serial_iface.reply("", "6", "", "0-0")
        dummy_serial_iface.reply("", "", "0-0\r\n")
        dummy_serial_iface.reply("", "", "0-0\r\n")
        self._max_digital_ports = self.__getitem__("max_digital")
        self._max_analog_ports = self.__getitem__("max_analog")
        self._max_proxies = self._max_digital_ports + self._max_analog_ports


class TestGpioArray(unittest.TestCase):
    def setUp(self):
        self.platform = Platform()
        self.platform.device_constructors["gpio_array"] = MockArray
        self.platform._known_devices = {"gpio_array_op1": "COM1"}
        self.platform.build(
            platform_name="Perry",
            devices=["Gpio_Array_1"],
            update_docs=False,
            open_gui=True,
        )
        self.device = self.platform["Gpio_Array_1"]
        self.solenoid = self.platform["sv_injection_n2"]
        self.device: MockArray
        self.dummy_serial_iface = self.device.serial_iface

    def tearDown(self):
        self.dummy_serial_iface.reply("", "", "0-0\r\n")
        self.platform.clear()

    @unittest.skip("no")
    def test_write(self):
        self.dummy_serial_iface.reply("", "", "0-0\r\n")
        self.device["save"] = "run"

    @unittest.skip("no")
    def test_read(self):
        self.dummy_serial_iface.reply("", "0-0\r\n")
        d = self.device["error"]

    def test_subdevice(self):
        self.assertTrue(isinstance(self.solenoid, SolenoidValve))

        self.dummy_serial_iface.reply("", "", "0-0\r\n")
        self.solenoid["valve"] = "open"

    @unittest.skip("no")
    def test_module(self):
        from omniplatypus.utilities.general import path_to_configuration_folder

        print(path_to_configuration_folder())


if __name__ == "__main__":
    unittest.main()

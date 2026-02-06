"""
File: test_device_gpio.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: unit test for the gpio array device.
"""

import unittest

import time
from omniplatypus.devices.platform import Platform
from omniplatypus.devices.nrg.gpio import GpioArray, DigitalPort, AnalogPort


class GpioArrayTest(unittest.TestCase):
    device: GpioArray = None
    platform = None

    @classmethod
    def setUpClass(cls):
        cls.platform = Platform()
        cls.platform.build(
            platform_name="dummy",
            devices=["gpio"],
        )
        cls.device = cls.platform["gpio"]

    @classmethod
    def tearDownClass(cls):
        cls.device.close()

    def test_class(self):
        self.assertTrue(isinstance(self.device, GpioArray))

    def test_id(self):
        devid = self.device["ID"]
        self.device["ID"] = "UNIT_TEST"
        self.device.parameter_by_name("ID").last_known_value = None  # Force read
        self.assertEqual("UNIT_TEST", self.device["ID"])
        self.device["ID"] = devid

    def test_error(self):
        self.assertFalse(self.device["error"].is_error)

    def test_subdevices(self):
        all_devices = self.platform.devices.keys()
        proxies = self.device.proxies()
        self.assertNotEqual(0, len(proxies))
        for name in proxies:
            self.assertTrue(name in all_devices)
            self.assertTrue(
                isinstance(self.device.proxy(name), (DigitalPort, AnalogPort))
            )

    def test_digital(self):
        port: DigitalPort = self.device.proxy("led")
        port["mode"] = "output"
        for i in range(10):
            time.sleep(0.1)
            port["write"] = "on"
            time.sleep(0.1)
            port["write"] = "off"

    def test_pwm(self):
        port: DigitalPort = self.device.proxy("led")
        port["mode"] = "pwm"
        for i in range(25):
            time.sleep(0.1)
            port["pwm"] = i * 10

    def test_analog(self):
        port: AnalogPort = self.device.proxy("pot")
        for i in range(10):
            time.sleep(0.1)
            print("reading: " + str(port["read"]))


if __name__ == "__main__":
    unittest.main()

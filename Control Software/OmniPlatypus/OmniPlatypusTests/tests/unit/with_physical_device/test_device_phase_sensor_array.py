"""
File: test_device_phase_sensor_array.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: unit test for the phase sensor array device.
"""

import unittest

import time

from omniplatypus.devices import Platform
from omniplatypus.devices.nrg.phase_sensor import PhaseSensorArray, PhaseSensor


class PhaseSensorArrayTest(unittest.TestCase):
    device: PhaseSensorArray = None
    platform = None

    @classmethod
    def setUpClass(cls):
        cls.platform = Platform()
        cls.platform.build(
            platform_name="dummy",
            devices=["Phase_Sensor_Array"],
        )
        cls.device = cls.platform["Phase_Sensor_Array"]
        for name in cls.device.proxies():
            sensor: PhaseSensor = cls.device.proxy(name)
            sensor["calibrate"] = "run"
        time.sleep(5)

    @classmethod
    def tearDownClass(cls):
        cls.device.close()

    def test_class(self):
        self.assertTrue(isinstance(self.device, PhaseSensorArray))

    def test_id(self):
        devid = self.device["ID"]
        self.device["ID"] = "UNIT_TEST"
        self.device.parameter_by_name("ID").last_known_value = None  # Force read
        self.assertEqual("UNIT_TEST", self.device["ID"])
        self.device["ID"] = devid

    def test_error(self):
        erro = self.device["error"].is_error()
        self.assertFalse(erro)

    def test_subdevices(self):
        all_devices = self.platform.devices.keys()
        proxies = self.device.proxies()
        self.assertNotEqual(0, len(proxies))
        for name in proxies:
            self.assertTrue(name in all_devices)
            self.assertTrue(isinstance(self.device.proxy(name), PhaseSensor))

    def test_array_parameters(self):
        all_values = self.device["read_all"]
        self.assertEqual(self.device.max_proxies, self.device["max_sensors"])

    def test_each_sensor(self):
        for name in self.device.proxies():
            sensor: PhaseSensor = self.device.proxy(name)
            phase = sensor["phase"]
            analog = sensor["analog"]
            sensor["monitor"] = "once"
            print(f"Please cause a phase change on '{name}'.")
            new_phase = sensor.wait(60.0)
            new_analog = sensor["analog"]
            self.assertFalse(phase.is_liquid() == new_phase.is_liquid())
            self.assertTrue(
                abs(analog - new_analog) >= 300,
                f"Detected low difference between phases, calibrate sensor {name}!",
            )


if __name__ == "__main__":
    unittest.main()

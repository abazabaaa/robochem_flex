"""
File: test_device_syringe_pump.py
Author: Simone Pilon - Noël Research Group - 2023
GitHub: https://github.com/simone16

Description: unit test for the self-made syringe pumps.
"""

import unittest

from omniplatypus.devices import SyringePump
from omniplatypus.devices import Platform
import pause


class SyringePumpNRGTest(unittest.TestCase):
    device: SyringePump = None

    @classmethod
    def setUpClass(cls):
        test_platform = Platform()
        test_platform.build(
            platform_name="Perry",
            devices=["Sampler_pump"],
        )
        cls.device = test_platform["Sampler_pump"]
        print(cls.device.usage())

    @classmethod
    def tearDownClass(cls):
        cls.device.close()

    def test_id(self):
        devid = self.device["ID"]
        self.device["ID"] = "UNIT_TEST"
        self.device.parameter_by_name("ID").last_known_value = None  # Force read
        self.assertEqual("UNIT_TEST", self.device["ID"])
        self.device["ID"] = devid

    def test_ack(self):
        self.assertEqual(1, self.device["ack_pump"])

    def test_noack(self):
        self.device["ack_pump"] = "OFF"
        self.device["enable"] = "ON"
        pump_volume = 100.0
        if self.device["volume"] <= pump_volume:
            pump_volume = -pump_volume
        self.device["pump"] = pump_volume
        while True:
            pause.seconds(0.1)
            if self.device["volume_left"] == 0.0:
                break
        self.device["enable"] = "OFF"
        self.device["ack_pump"] = "ON"
        self.assertEqual(1, self.device["ack_pump"])

    def test_valve(self):
        self.device["valve_setpoint"] = "OFF"
        self.assertTrue(self.device["valve_actual"] == "OFF")
        self.device["valve_setpoint"] = "ON"
        self.assertTrue(self.device["valve_actual"] == "ON")

    def test_diameter(self):
        diameter = self.device["diameter"]
        self.device["diameter"] = 4.3
        self.assertEqual(4.3, self.device["diameter"])
        self.device["diameter"] = diameter

    def test_stepsml(self):
        stepsml = self.device["steps_per_ml"]
        self.device["steps_per_ml"] = 80000
        self.assertAlmostEqual(80000, self.device["steps_per_ml"], 1)
        self.device["steps_per_ml"] = stepsml

    def test_flowrate(self):
        # error = self.device["error"]
        flowrate = self.device["flowrate"]
        self.device["flowrate"] = 4.5
        self.assertAlmostEqual(4.5, self.device["flowrate"], 1)
        self.device["flowrate"] = flowrate

    def test_zero(self):
        self.device["zero"] = "IN_PLACE"
        self.assertEqual(0.0, self.device["volume"])
        self.device["enable"] = "ON"
        self.device["zero"] = "FILL"
        self.assertAlmostEqual(self.device.syringe_volume, self.device["volume"], 0)
        self.device["zero"] = "EMPTY"
        self.assertEqual(0.0, self.device["volume"])
        self.device["enable"] = "OFF"

    def test_pump(self):
        self.device["enable"] = "ON"
        self.device["zero"] = "EMPTY"
        self.assertAlmostEqual(0.0, self.device["volume"], 1)
        self.device["pump"] = -100.0
        self.assertAlmostEqual(0.0, self.device["volume_left"], 1)
        self.assertAlmostEqual(100.0, self.device["volume"], 1)
        self.device["pump"] = 50.0
        self.assertAlmostEqual(0.0, self.device["volume_left"], 1)
        self.assertAlmostEqual(50.0, self.device["volume"], 1)
        # does not work with specific exception. Why??
        with self.assertRaises(Exception):
            self.device["pump"] = 100.0
        # self.assertAlmostEqual(0.0, self.device['volume'], 0)
        self.device["enable"] = "OFF"

    def test_enable(self):
        self.device["enable"] = "OFF"
        self.assertTrue(self.device["enable"] == "OFF")
        # does not work with specific exception. Why??
        with self.assertRaises(Exception):
            self.device["pump"] = 100.0

    def test_prime(self):
        self.device["enable"] = "ON"
        for i in range(2):
            self.device["valve_setpoint"] = "OFF"
            self.device["zero"] = "FILL"
            self.assertAlmostEqual(
                self.device.syringe_volume, self.device["volume"], places=0
            )
            self.device["valve_setpoint"] = "ON"
            self.device["zero"] = "EMPTY"
            self.assertAlmostEqual(0.0, self.device["volume"], places=2)
        self.device["enable"] = "OFF"


if __name__ == "__main__":
    unittest.main()

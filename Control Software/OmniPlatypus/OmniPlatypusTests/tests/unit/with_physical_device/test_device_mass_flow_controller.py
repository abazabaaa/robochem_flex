"""
File: test_device_mass_flow_controller.py
Author: Simone Pilon - Noël Research Group - 2023
GitHub: https://github.com/simone16

Description: unit test for the Bronkhorst mass flow controller devices.
"""

import unittest

from omniplatypus.devices import Platform
from time import sleep


class MassFlowControllerTest(unittest.TestCase):
    device = None

    @classmethod
    def setUpClass(cls):
        test_platform = Platform()
        test_platform.build(
            platform_name="Perry",
            devices=["Main_mfc"],
        )
        cls.device = test_platform["Main_mfc"]

    @classmethod
    def tearDownClass(cls):
        cls.device["flow_setpoint"] = 0.0
        cls.device.close()

    def test_read_flow(self):
        print(self.device["flow"])

    def test_set_flow(self):
        for index in range(5):
            flow = 0.2 * index
            self.device["flow_setpoint"] = flow
            timeout = 0
            for timeout in range(100):
                sleep(0.1)
                if abs(flow - self.device["flow"]) < 0.05:
                    break
            self.assertLess(timeout, 100)
            print(f"Achieved flow in {timeout/10} S.")
        self.device["flow_setpoint"] = 0.0

    def test_setpoint(self):
        self.device["flow_setpoint"] = 0.5
        self.assertEqual(0.5, self.device["flow_setpoint"])

    def test_units(self):
        self.assertEqual("mln/min", self.device["flow_units"])

    def test_tag(self):
        tag = self.device["tag"]
        self.device["tag"] = "test"
        self.assertEqual("test", self.device["tag"])
        self.device["tag"] = tag

    def testT_temperature(self):
        temperature = self.device["temperature"]
        self.assertLess(temperature, 40.0)
        self.assertGreater(temperature, 10.0)


if __name__ == "__main__":
    unittest.main()

"""
File: test_device_syringe_pump.py
Author: Simone Pilon - Noël Research Group - 2023
GitHub: https://github.com/simone16

Description: unit test for the liquid handler syringe pump.
"""

import unittest

from omniplatypus.devices.platform import Platform
from time import sleep


class LiquidHanlerPumpTest(unittest.TestCase):
    device = None

    @classmethod
    def setUpClass(cls):
        test_platform = Platform()
        device_data = {
            "known_device": "LH_pump",
            "tags": ["liquid", "flow", "sampler", "3w_valve", "serial"],
            "class": "Liquid_Handler_Pump",
        }
        test_platform.build_device("Sampler_pump", device_data, True)
        cls.device = test_platform["Sampler_pump"]

    @classmethod
    def tearDownClass(cls):
        cls.device["enable"] = "OFF"
        cls.device.close()

    def test_no_reverse(self):
        self.device["valve_setpoint"] = "OFF"  # connect to reservoir
        self.device["enable"] = "ON"
        self.device["zero"] = "EMPTY"
        self.device["flowrate"] = 1.0
        self.device["pump"] = -310.0
        self.device["valve_setpoint"] = "ON"  # connect to vials
        self.device["pump"] = 300.0
        self.device["valve_setpoint"] = "OFF"  # connect to reservoir
        self.device["pump"] = -300.0
        self.device["valve_setpoint"] = "ON"  # connect to vials
        for i in range(10):
            sleep(2)
            self.device["pump"] = 150.0
            self.device["valve_setpoint"] = "OFF"  # connect to reservoir
            self.device["pump"] = -150.0
            self.device["valve_setpoint"] = "ON"  # connect to vials
        self.device["enable"] = "OFF"

    # def test_flush(self):
    #     self.device['enable'] = 'ON'
    #     self.device['zero'] = 'EMPTY'
    #     self.device['flowrate'] = 2.0
    #     self.device['pump'] = -10.0
    #     for i in range(5):
    #         self.device['valve_setpoint'] = 'OFF'  # connect to reservoir
    #         self.device['pump'] = -700.0
    #         self.device['valve_setpoint'] = 'ON' # connect to vials
    #         self.device['pump'] = +700.0
    #     self.device['enable'] = 'OFF'

    # def test_nothing(self):
    #     self.device['enable'] = 'OFF'


if __name__ == "__main__":
    unittest.main()

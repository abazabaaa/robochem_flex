"""
File: test_device_syringe_pump_nrg_timing.py
Author: Simone Pilon - Noël Research Group - 2023
GitHub: https://github.com/simone16

Description: unit test for verifying the timing of the self-made syringe pumps.
    Note: this is meant to run on an arduino without the actual pump. If ran on a physical pump **it will destroy it**.
"""

import unittest

from omniplatypus.devices.platform import Platform
from timeit import default_timer as timer


class SyringePumpNRGTest(unittest.TestCase):
    device = None
    platform = None

    @classmethod
    def setUpClass(cls):
        test_platform = Platform()
        test_platform.build(
            platform_name="dummy",
            devices=["pump"],
        )
        cls.platform = test_platform
        cls.device = test_platform["pump"]

    @classmethod
    def tearDownClass(cls):
        cls.platform.clear()

    def test_pump_time(self):
        data = {"diameter": [], "flow": [], "time": [], "theo": [], "error": []}
        try:
            self.device._timeout_multiplier = 10.0
            self.device["encoder"] = "OFF"
            self.device["enable"] = "ON"
            for diameter in [5.0, 10.0, 15.0]:
                self.device["diameter"] = diameter
                for flowrate in [3.0, 5.0]:
                    self.device["flowrate"] = flowrate
                    for pump_volume in [50.0, 100.0, 500.0, 1000.0]:
                        self.device["volume"] = round(pump_volume * 1.1, 2)
                        start = timer()
                        self.device["pump"] = pump_volume
                        end = timer()
                        actual_time = end - start
                        theoretical_time = self.device.pumping_time(pump_volume)
                        data["diameter"].append(diameter)
                        data["flow"].append(flowrate)
                        data["time"].append(round(actual_time, 3))
                        data["theo"].append(round(theoretical_time, 3))
                        data["error"].append(
                            round(
                                (actual_time - theoretical_time)
                                * 100
                                / theoretical_time,
                                1,
                            )
                        )
        finally:
            for i in range(len(data["diameter"])):
                print(
                    f"D: {data['diameter'][i]}, Flow: {data['flow'][i]}, T: {data['time'][i]} ({data['theo'][i]}), E: {data['error'][i]}%"
                )
            for i in range(len(data["diameter"])):
                print(
                    f"{data['diameter'][i]}, {data['flow'][i]}, {data['time'][i]}, {data['theo'][i]}, {data['error'][i]}"
                )
            for i in range(len(data["diameter"])):
                self.assertLess(data["error"][i], 5.0)


if __name__ == "__main__":
    unittest.main()

"""
File: test_device_light_array.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: Unit test for the array light control device.
"""

import unittest
from time import sleep

from omniplatypus.devices.platform import Platform


class LightsTest(unittest.TestCase):
    device = None
    platform = None

    @classmethod
    def setUpClass(cls):
        test_platform = Platform()
        test_platform.build(
            platform_name="Perry",
            devices=["Light_Array"],
        )
        cls.platform = test_platform
        cls.device = test_platform["Light_Array"]
        cls.subdevice = test_platform["uflow_kessil"]

    @classmethod
    def tearDownClass(cls):
        cls.platform.clear()

    def test_enable(self):
        self.device["enable"] = "ON"
        sleep(2)
        self.device["enable"] = "OFF"

    def test_intensity(self):
        self.device["enable"] = "ON"
        self.subdevice["intensity"] = 100
        print(self.subdevice["intensity"])
        sleep(5)
        self.subdevice["intensity"] = 0
        self.device["enable"] = "OFF"

    def test_calibrate(self):
        prompt = ""
        self.device["enable"] = "ON"
        try:
            while not prompt.lower() == "exit":
                prompt = input("Intensity [%]:")
                intensity = int(prompt)
                self.subdevice["intensity"] = intensity
                sleep(5)
                average = 0
                for i in range(5):
                    current = self.device["current"]
                    print(f"current: {current} mA.")
                    average += current
                average /= 5
                print(f"average of 5: {average} mA.")
        finally:
            self.device["enable"] = "OFF"


if __name__ == "__main__":
    unittest.main()

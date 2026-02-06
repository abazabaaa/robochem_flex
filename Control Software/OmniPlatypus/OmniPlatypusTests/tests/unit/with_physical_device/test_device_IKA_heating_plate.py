"""
File: test_device_liquid_handler_sampler.py
Author: Simone Pilon - Noël Research Group - 2023
GitHub: https://github.com/simone16

Description: unit test for the liquid handler CNC sampler.
"""

import unittest
import time


from omniplatypus.devices.platform import Platform
from omniplatypus.devices.ika.heating_plate import HeatingPlate


class HeatingPlateTest(unittest.TestCase):
    device = HeatingPlate
    platform = None

    @classmethod
    def setUpClass(cls):
        test_platform = Platform()
        test_platform.build(
            platform_name="Perry",
            devices=["heating_plate_IKA"],
        )
        cls.platform = test_platform
        cls.device = test_platform["heating_plate_IKA"]

    @classmethod
    def tearDownClass(cls):
        cls.platform.clear()

    # Read the device name
    def test_name(self):  # Read the device name
        device_name = self.device["name"]
        print(f"The device name is {device_name}")

    def test_read_sensor_temperature(self):  # Read actual external sensor value
        sensor_temperature = self.device["actual_sensor_temperature"]
        print(f"The temperature is {sensor_temperature}°C")

    def test_read_hotplate_temperature(self):  # Read actual hotplate sensor value
        hotplate_temperature = self.device["actual_hotplate_temperature"]
        print(f"The HotPlate temperature is {hotplate_temperature}°C")

    def test_rated_temperature(self):  # Read rated temperature value
        t_rated = self.device["read_rated_temperature"]
        print(f"The rated temperature is {t_rated}°C")

    def test_read_safety_temperature(self):  # Read rated set safety temperature value
        t_safety = self.device["read_rated_safety_temperature"]
        # t_safety_cleaned = t_safety.split()[0]
        print(f"The safety temperature is {t_safety}°C")

    def test_heating(self):
        temperature_value = 40.00
        self.device["set_stirring"] = 900
        self.device["start_stirring"] = "run"
        self.device["set_temperature"] = temperature_value
        print(f"The stirring speed is set to {temperature_value}°C.")

        # Start the heater
        self.device["start_heating"] = "run"
        print("The heating has started.")

        while True:
            # Read the current temperature from the device
            current_temperature = float(self.device["actual_sensor_temperature"])
            print(f"Current temperature: {current_temperature}°C")

            # Check if the target temperature is reached
            if current_temperature >= temperature_value:
                print("Target temperature reached. Maintaining temperature.")
                break

            # Short delay to prevent too frequent polling
            time.sleep(1)

            # Maintain temperature: the heater will keep running to maintain the set temperature.
            # This could also involve switching the heating on/off if temperature control is too strict
        print("Maintaining target temperature.")
        # time.sleep(10)
        #
        # # Stop the heater if needed
        # self.device["stop_heating"] = "run"
        # self.device["stop_stirring"] = "run"
        # print("The heating and stirring have stopped.")

    def test_stirring(self):
        stirring_speed = 1000.0
        self.device["set_stirring"] = stirring_speed
        print(f"The stirring speed is set to {stirring_speed} rpm.")

        # Start the motor
        self.device["start_stirring"] = "run"
        print("The stirring has started.")
        stirring_speed_actual = self.device["read_rated_speed"]
        time.sleep(10)
        print(f"The actual stirring speed is {stirring_speed_actual} rpm")
        time.sleep(5)

        # Stop the motor if needed
        self.device["stop_stirring"] = "run"
        print("The stirring has stopped.")

    def test_actual_stirring_speed(self):  # Read stirring speed value
        speed = self.device["actual_stirring_speed"]
        print(f"The rated stirring speed is {speed} rpm")

    def test_read_rated_speed(self):  # Read rated speed value
        speed = self.device["read_rated_speed"]
        print(f"The rated stirring speed is {speed} rpm")

    def test_set_operating_mode(self):  # Set operating mode
        operation_mode = self.device["set_operating_mode"]
        print(f"The operating mode is {operation_mode}")


if __name__ == "__main__":
    unittest.main()

"""
File: temperature.py
Author: Miguel Claros - Noël Research Group - 2024
GitHub: https://github.com/MiguelClaros

Description: Unit tasks to manipulate the reactor temperature.
"""

import time

from omniplatypus.devices.ika.heating_plate import HeatingPlate
from omniplatypus.procedures.unit_tasks.base_unit_task import BaseUnitTaskTemplate


class SetReactorTemperature(BaseUnitTaskTemplate):
    """
    Set the target temperature of a IKA RCT basic Heating Plate.

    Usage:
    call cls.run() to execute this task.
    """

    @classmethod
    def run(cls, heater: HeatingPlate, temperature: float, stirring: float = 0) -> None:
        """
        Set the target temperature for a heating plate.

        @param heater: HeatingPlate
            The heating plate device.
        @param temperature: float | int
            The temperature is a value from 0 to 360°C.
        @param stirring: float | int
            The stirring is a value from 0 to 1500 rpm.
        """
        return super().run(heater=heater, temperature=temperature, stirring=stirring)

    @classmethod
    def _validate_input(
        cls, heater: HeatingPlate, temperature: float, stirring: float = 0
    ) -> None:
        """Validate input arguments."""
        cls._validate_input_log(
            heater, HeatingPlate, "Heating Plate must be a 'HeatingPlate' object."
        )
        # temperature is validated by heating_plate class.

    @classmethod
    def _execute(
        cls, heater: HeatingPlate, temperature: float, stirring: float = 0
    ) -> None:
        if temperature > 0:
            heater["start_heating"] = "run"
        else:
            heater["stop_heating"] = "run"
        if stirring > 0:
            heater["start_stirring"] = "run"
        else:
            heater["stop_stirring"] = "run"
        heater["set_temperature"] = temperature
        heater["set_stirring"] = stirring


class CheckReactorTemperature(BaseUnitTaskTemplate):
    """
    Check that the target temperature of a IKA RCT basic Heating Plate is stable

    Usage:
    call cls.run() to execute this task.
    """

    @classmethod
    def run(
        cls,
        heater: HeatingPlate,
        temperature: float,
        stability_duration: float = 30.0,
        tolerance: float = 0.5,
    ) -> None:
        """
        Check that the heating plate termperature is stable and within the set range.

        @param heater: HeatingPlate
            The heating plate device.
        @param temperature: float
            The temperature is a value from 0 to 360°C [°C].
        @param stability_duration: float
            The time when the temperature need to be stable before starting the experiment [S].
        @param tolerance: float
            The tolerance is the acceptable range of variation [°C].
        """
        return super().run(
            heater=heater,
            temperature=temperature,
            stability_duration=stability_duration,
            tolerance=tolerance,
        )

    @classmethod
    def _validate_input(
        cls,
        heater: HeatingPlate,
        temperature: float,
        stability_duration: float = 30.0,
        tolerance: float = 0.5,
    ) -> None:
        """Validate input arguments."""
        cls._validate_input_log(
            heater, HeatingPlate, "Heating Plate must be a 'HeatingPlate' object."
        )
        # temperature is validated by heating_plate class.

    @classmethod
    def _execute(
        cls,
        heater: HeatingPlate,
        temperature: float,
        stability_duration: float = 30,
        tolerance: float = 0.5,
    ) -> None:
        # Stability tracking
        start_time = None
        while True:
            # Read the current temperature from the device
            current_temperature = heater["actual_sensor_temperature"]
            time.sleep(5)

            if abs(current_temperature - temperature) <= tolerance:
                # Start the stability timer if not already started
                if start_time is None:
                    start_time = time.time()
                elif time.time() - start_time >= stability_duration:
                    break  # Temperature has been stable; start the experiment
            else:
                # Reset the timer if the temperature goes out of range
                start_time = None

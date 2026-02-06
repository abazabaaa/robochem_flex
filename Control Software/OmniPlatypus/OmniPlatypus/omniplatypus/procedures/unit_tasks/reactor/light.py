"""
File: light.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: Unit tasks to manipulate the reactor light source.
"""

from omniplatypus.procedures.unit_tasks.base_unit_task import BaseUnitTaskTemplate
from omniplatypus.devices.nrg.light import LightSource, LightArray


class SetLightSourceIntensity(BaseUnitTaskTemplate):
    """
    Set the target intensity of a light source.

    Usage:
    call cls.run() to execute this task.
    """

    @classmethod
    def run(cls, source: LightSource, intensity: float | int) -> None:
        """
        Set the target intensity for a light source.

        @param source: LightSource
            The light source device.
        @param intensity: float | int
            The intensity required as a percentage of the maximum.
        """
        return super().run(source=source, intensity=intensity)

    @classmethod
    def _validate_input(cls, source: LightSource, intensity: float | int) -> None:
        """Validate input arguments."""
        cls._validate_input_log(
            source, LightSource, "Light Source must be a 'LightSource' object."
        )
        # intensity is validated by lightsource class.

    @classmethod
    def _execute(cls, source: LightSource, intensity: float | int) -> None:
        # todo notify the platform daemon (which does not exist yet) that this request was made, so that
        # it can ensure the condition is met when the reaction starts.
        intensity = round(intensity)
        source_parent: LightArray = source.parent
        if intensity == 0:
            # try to disable whole array
            source["intensity"] = intensity
            all_zero = True
            for source_name in source_parent.proxies():
                if not source_parent.proxy(source_name)["intensity"] == 0:
                    all_zero = False
                    break
            if all_zero:
                source_parent["enable"] = "OFF"
        else:
            if source_parent["enable"] == "OFF":
                source_parent["enable"] = "ON"
            source["intensity"] = intensity

"""
File: thermochemistry.py
Author: Simone Pilon - Noël Research Group - 2025
GitHub: https://github.com/simone16

Description: Thermochemical experiments procedures.
"""

from threading import Thread, Event
import time

from omniplatypus.devices.ika.heating_plate import HeatingPlate
from omniplatypus.devices.nrg.gpio import Fan
from omniplatypus.procedures.experiments.chemistry import ChemicalReaction
from omniplatypus.procedures.experiments.experiment_parameters import (
    ExperimentalParameter,
    NumericalParameter,
)
from omniplatypus.procedures.unit_tasks.reactor.temperature import (
    SetReactorTemperature,
    CheckReactorTemperature,
)
from omniplatypus.procedures.unit_tasks.sensing.monitoring import (
    ContinuousMonitoring,
    MonitoredValue,
)


class ThermochemicalReaction(ChemicalReaction):
    """
    Basic procedure for thermally activated reaction experiments on platform Perry.
    In addition to the base ChemicalReaction requirements, this experiment sets temperature via an IKA stirring plate
    device.
    """

    _required_parameters: list[ExperimentalParameter] = [
        NumericalParameter("temperature", 15.0, "°C", min_value=15.0, max_value=100.0),
    ] + ChemicalReaction.get_required_parameters()

    _optional_parameters: list[ExperimentalParameter] = [
        NumericalParameter("stirring", 300.0, "rpm", min_value=0.0, max_value=1500.0),
    ] + ChemicalReaction.get_optional_parameters()

    _required_devices: set[str] = ChemicalReaction._required_devices.union(
        {
            "heating_plate_IKA",
        }
    )

    _heater: HeatingPlate
    _cooler: Fan | None
    _stop_monitoring: Event
    _monitoring_thread: Thread | None

    def __init__(
        self,
        analytical_method: str | None = None,
    ) -> None:
        """
        Constructor.

        @param analytical_method: str | None = None
            The analytical method to build the platform with.
            Must match exactly one of the keys of cls._analytical_methods (or None).
        """
        super().__init__(analytical_method=analytical_method)

        self._stop_monitoring = Event()
        self._monitoring_thread = None

        self._cooler = None

    # noinspection PyTypeChecker
    def _procedure_build(self) -> None:
        ChemicalReaction._procedure_build(self)
        self._heater = self._platform["heating_plate_IKA"]
        try:
            self._cooler = self._platform["fan_ika_cooling"]
            self._cooler["fan"] = "ON"
        except KeyError:
            self._cooler = None
            self._log(
                "Thermochemical experiment is missing a cooling device for fast cooling.",
                level="warning",
            )

        # We keep checking the temperature of the reactor
        # At the moment nothing about this is checked, but the readings are logged.
        temperature_monitoring = MonitoredValue(
            device=self._heater,
            parameter_name="actual_sensor_temperature",
            checks=None,
            store_values=4,
        )
        self._stop_monitoring.clear()
        self._monitoring_thread = ContinuousMonitoring.get_thread(
            values_to_monitor=[temperature_monitoring],
            stop_event=self._stop_monitoring,
            update_delay=30.0,
        )
        self._monitoring_thread.start()

    def _reactor_prepare(
        self, conditions: dict[str, ExperimentalParameter | NumericalParameter]
    ) -> None:
        """
        Prepare the reactor for the reaction. This is called before the start of the run, to allow maximum time for
        the reactor to be ready.
        Sets the light intensity of the reactor.

        @param conditions: dict[str, ExperimentalParameter]
            Physical conditions for the run.
        """
        temperature = conditions["temperature"].set_units("°C", 1)
        stirring = int(conditions["stirring"].set_units("rpm", 0))
        self._cooler["fan"] = "OFF"
        SetReactorTemperature.run(
            heater=self._heater, temperature=temperature, stirring=stirring
        )

    def _reactor_wait_until_slug_can_be_prepared(
        self,
        conditions: dict[str, ExperimentalParameter | NumericalParameter],
    ) -> None:
        """
        This is called before starting the reaction and pumping the slug in the reactor.
        It should only return once the reactor parameters are such that we can expect the reactor to be ready after the
        slug has been prepared (5-10min). This prevents the prepared slug from sitting too long waiting for the reactor
        to be ready (avoiding prolonged background reactivity).

        @param conditions: dict[str, ExperimentalParameter]
            Physical conditions for the run.
        """
        target_temperature = conditions["temperature"].with_units("°C")
        current_temperature = self._heater["actual_sensor_temperature"]

        # If reactor is too hot, turn on the fan until in range.
        # While making the slug the reactor will cool further.
        if current_temperature > target_temperature + 5.0:
            self._cooler["fan"] = "ON"
            while True:
                time.sleep(30)
                if (
                    self._heater["actual_sensor_temperature"]
                    <= target_temperature + 5.0
                ):
                    break
            self._cooler["fan"] = "OFF"
        # If reactor is too cold, give it some time.
        # While making the slug the reactor can warm up the rest of the way.
        elif current_temperature < target_temperature - 20.0:
            while True:
                time.sleep(30)
                if (
                    self._heater["actual_sensor_temperature"]
                    >= target_temperature - 20.0
                ):
                    break

    def _reactor_wait_until_ready(
        self,
        conditions: dict[str, ExperimentalParameter | NumericalParameter],
    ) -> None:
        """
        This is called before starting the reaction and pumping the slug in the reactor.
        It should only return once the reactor parameters are stable and at the desired levels.

        @param conditions: dict[str, ExperimentalParameter]
            Physical conditions for the run.
        """
        temperature = conditions["temperature"].with_units("°C")
        CheckReactorTemperature.run(
            heater=self._heater,
            temperature=temperature,
            stability_duration=30.0,
            tolerance=2.5,
        )

    def _reactor_standby(
        self, conditions: dict[str, ExperimentalParameter | NumericalParameter]
    ) -> None:
        """
        Set the reactor in the standby configuration. This is called after the slug has exited the reactor.
        Turns off the heating.

        @param conditions: dict[str, ExperimentalParameter]
            Physical conditions for the run.
        """
        stirring = int(conditions["stirring"].with_units("rpm"))
        SetReactorTemperature.run(
            heater=self._heater, temperature=0.0, stirring=stirring
        )
        self._cooler["fan"] = "ON"

    def _procedure_shutdown(self) -> None:
        """
        Shut down procedure of the experimental platform, ensuring all devices are properly
        turned off or reset.
        """
        self._stop_monitoring.set()
        self._monitoring_thread.join(timeout=300)
        ChemicalReaction._procedure_shutdown(self)

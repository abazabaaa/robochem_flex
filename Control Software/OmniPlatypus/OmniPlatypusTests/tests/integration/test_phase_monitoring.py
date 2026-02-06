"""
File: test_phase_monitoring.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: unit test for the phase monitoring task.
"""

import unittest

from time import sleep
from threading import Thread, Event

from omniplatypus.devices.platform import Platform
from omniplatypus.devices.nrg.phase_sensor import PhaseSensor

from omniplatypus.procedures.unit_tasks.sensing.phase_sensors import (
    MonitorPhase,
)


class DrivingTest(unittest.TestCase):
    platform: Platform = None
    phase_sensor: PhaseSensor = None

    # noinspection PyTypeChecker
    @classmethod
    def setUpClass(cls):
        test_platform = Platform()
        test_platform.build(
            platform_name="dummy",
            devices=[
                "Phase_Sensor_Array",
            ],
        )
        cls.platform = test_platform
        cls.phase_sensor = test_platform["ps_sampler_out"]
        # cls.phase_sensor["calibrate"] = "run"
        # sleep(10.0)

    @classmethod
    def tearDownClass(cls):
        cls.platform.clear()

    def _reactor_monitoring_target(self) -> None:
        delta_t = 0.5
        phase_data = MonitorPhase.run(
            phase_sensor=self.phase_sensor,
            delta_t=delta_t,
            stop_event=self._reactor_monitoring_stop_event,
        )
        phase_data["Duration"] = (
            phase_data["Time"]
            .shift(periods=-1, fill_value=phase_data.iloc[-1].at["Time"])
            .sub(phase_data["Time"])
        )
        print(f"Reactor monitoring data:\n{phase_data}")
        measured_residence_time = 0.0
        for index, row in phase_data.iterrows():
            if not row["Phase"] == 0 and row["Duration"] >= 3 * delta_t:
                measured_residence_time = row["Time"]
                break
        print(
            f"Measured reactor residence time: {measured_residence_time:.1f} S ({measured_residence_time/60.0:.2f} min)"
        )

    def _start_reactor_monitoring(self) -> None:
        self._reactor_monitoring_stop_event = Event()
        self._reactor_monitoring_thread = Thread(
            target=self._reactor_monitoring_target, name="Reactor Monitoring Thread"
        )
        self._reactor_monitoring_thread.start()

    def _stop_reactor_monitoring(self) -> None:
        self._reactor_monitoring_stop_event.set()
        self._reactor_monitoring_thread.join(timeout=60.0)

    def test_ps(self):
        self._start_reactor_monitoring()
        monitoring_time = 60.0
        print(f"Monitoring for {monitoring_time} S.")
        sleep(monitoring_time)
        print(f"Done")
        self._stop_reactor_monitoring()


if __name__ == "__main__":
    unittest.main()

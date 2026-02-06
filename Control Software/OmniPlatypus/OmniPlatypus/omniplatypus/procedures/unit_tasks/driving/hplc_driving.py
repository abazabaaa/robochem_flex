"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

import time
from threading import Lock, Thread, Event
from omniplatypus.devices.knauer.HPLC_pump_azura import AzuraHPLC
from omniplatypus.procedures.unit_tasks.base_unit_task import BaseUnitTaskTemplate
from omniplatypus.devices.platform import Platform


class pump_azura(BaseUnitTaskTemplate):
    """
    This class is a unit task for the azura pump.
    since the azura pump has a very simple interface.
    The idea is you can either turn the pump on or off, set a flowrate and pump a certain
    amount of volume by staring a timer on a different thread and stopping it when the volume is reached.

    """

    @classmethod
    def run(cls, pump: AzuraHPLC, flowrate: int = None, volume: int = 0) -> None:
        """
        this function handles the pumping for the azura pump.
        :param flowrate: the flowrate of the pump in ul/min if None the flowrate will not be changed
        :param volume: the volume to pump in ul, if 0 the pump will be stopped, if -1 the pump will run indefinitely
        otherwise the pump will pump the volume
        """
        return cls._run(pump=pump, flowrate=flowrate, volume=volume)

    @classmethod
    def validate_input(cls, pump: AzuraHPLC, flowrate: int, volume: int) -> bool:
        """
        this function validates the input for the run function
        :param flowrate: the flowrate of the pump in ul/min if None the flowrate will not be changed
        :param volume: the volume to pump in ul, if 0 the pump will be stopped, if -1 the pump will run indefinitely
        otherwise the pump will pump the volume.
        """
        cls._validate_input_log(pump, AzuraHPLC)

        if flowrate is not None:
            cls._validate_input_log(flowrate, int)

        cls._validate_input_log(volume, lambda x: isinstance(x, int) and x >= -1)

    @classmethod
    def execute(cls, pump: AzuraHPLC, flowrate: int, volume: int) -> None:
        """
        this function executes the run function
        :param flowrate: the flowrate of the pump in ul/min if None the flowrate will not be changed
        :param volume: the volume to pump in ul, if 0 the pump will be stopped, if -1 the pump will run indefinitely
        otherwise the pump will pump the volume.
        """

        # if the flowrate is there set it:
        if flowrate is not None:
            pump["flowrate"] = flowrate

        # if the volume is 0 stop the pump
        if volume == 0:
            pump["state"] = "OFF"
            return

        # if the volume is -1 run the pump indefinitely
        if volume == -1:
            pump["state"] = "ON"
            return

        # if the volume is positive we need to do a cool thing
        time = (volume / pump["flowrate"]) * 60
        lock = Lock()

        def _daemon_stop_pump():
            with lock:
                pump["state"] = "ON"
            time.sleep(time)
            with lock:
                pump["state"] = "OFF"

        t = Thread(target=_daemon_stop_pump, daemon=True)
        t.start()

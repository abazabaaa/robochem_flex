"""
File: platform_calibration.py
Author: Simone Pilon - Noël Research Group - 2024
GitH0ub: https://github.com/simone16

Description: Interactive script to run calibration routines.
"""

from time import sleep

import pandas as pd
from timeit import default_timer as timer

from omniplatypus.utilities.general import run_all
from omniplatypus.devices.platform import Platform
from omniplatypus.devices.nrg.sampler import Sampler, GrblPosition, InjectionPort
from omniplatypus.devices.nrg.phase_sensor import PhaseSensor

from omniplatypus.procedures.unit_tasks.driving.driving_pumps import (
    FillPump,
    PumpVolume,
    PrimePump,
)
from omniplatypus.procedures.unit_tasks.sensing.phase_sensors import (
    PumpUntilPhaseChange,
)
from omniplatypus.procedures.unit_tasks.sampling.liquid_handler_injecting import (
    ConnectInjectionPort,
    Inject,
)
from omniplatypus.procedures.unit_tasks.sampling.liquid_handler_sampling import (
    GenerateSampleDataframe,
    PumpSample,
)


def yes_no(question: str) -> bool:
    """
    Returns true if the question was answered positively.

    @param question: str
        The question to prompt the used for.
    @return: bool
        True is the answer was yes.
    """
    question += " (y/N):"
    answer = input(question)
    return answer.lower() in ("y", "yes", "sure", "yeah")


def purge(platform):
    """
    Purge the platform with gas.
    """
    ConnectInjectionPort.run(
        platform=platform,
        sampler=platform["Sampler_cnc"],
        port_name="injection_flow",
        inject=False,
    )
    Inject.run(
        platform=platform,
        sampler=platform["Collector_cnc"],
        injection_port_name="injection_waste",
        move_speed=3000.0,
        plunge_speed=200.0,
        volume=0.0,
        flowrate=1.0,
        retract=False,
    )
    platform["sv_injection_n2"]["valve"] = "open"
    input("Press enter when the whole line is clear of liquid")
    platform["sv_injection_n2"]["valve"] = "close"


def calibrate_all_ps(platform):
    """
    Calibrate all phase sensors.
    """
    phase_sensors = [
        device
        for device in platform.devices.values()
        if isinstance(device, PhaseSensor)
    ]
    for ps in phase_sensors:
        print(f"Calibrating {ps}...")
        ps["calibrate"] = "run"
        sleep(2.0)
    sleep(10.0)
    print("calibration done.")


def assess_volumes(platform: Platform):
    """
    Pump liquid to measure volumes between phase sensors.

    @param platform: Platform
        The platform
    """
    # phase_sensors = [
    #     device
    #     for device in platform.devices.values()
    #     if isinstance(device, PhaseSensor)
    # ]
    phase_sensor_names = [
        "ps_handler_out",
        "ps_reactor_in",
        "ps_reactor_out",
        "ps_raman_in",
        "ps_out",
    ]
    phase_sensors = [
        platform[phase_sensor_name] for phase_sensor_name in phase_sensor_names
    ]
    print("Phase sensors will be checked in this order:")
    for ps in phase_sensors:
        print(ps)
    print("")

    ConnectInjectionPort.run(
        platform=platform,
        sampler=platform["Sampler_cnc"],
        port_name="injection_flow",
        inject=False,
    )

    for ps in phase_sensors:
        if not yes_no(f"Checking phase sensor {ps}, continue?"):
            return
        FillPump.run(pump=platform["Main_Pump_1"])
        volume = PumpUntilPhaseChange.run(
            pump=platform["Main_Pump_1"],
            phase_sensor=ps,
            flowrate=2.0,
            max_volume=8000.0,
            wait_for_gas=False,
        )
        print(f"measured volume: {volume:.2f} uL")
        answer = ""
        while not answer.lower() == "no":
            answer = input("Pump more (uL or 'no')?")
            try:
                pump = float(answer)
                PumpVolume.run(pump=platform["Main_Pump_1"], volume=pump, flowrate=1.0)
                print(f"pumped {pump} uL.")
            except ValueError:
                pass


def calibrate_injection_ports(platform):
    samplers = [
        device for device in platform.devices.values() if isinstance(device, Sampler)
    ]
    for sampler in samplers:
        if not yes_no(f"Calibrate {sampler}?"):
            continue

        sampler["feed"] = 2000.0
        injection_ports = {
            name: location
            for name, location in sampler.locations.items()
            if isinstance(location, InjectionPort)
        }
        positions = {}
        for name, port in injection_ports.items():
            print(f"Aligning '{name}':")
            destination = sampler.locations[name].move
            sampler["move"] = destination
            needle_down = GrblPosition(z=sampler.vials_top_z)
            needle_down.z -= 0.5
            needle_up = sampler.travel_position
            sampler["feed"] = 400
            position = ""
            while not position == "exit":
                sampler["move"] = needle_down
                print(sampler["position"])
                position = input("adjust?>")
                sampler["move"] = needle_up
                if position == "exit":
                    break
                position = position.split()
                coords = {}
                for p in position:
                    coords[p[0].lower()] = float(p[1:])
                destination.x += coords.get("x", 0.0)
                destination.y += coords.get("y", 0.0)
                sampler["move"] = destination
            if yes_no("Plunge needle?"):
                destination_2 = sampler.locations[name].plunge
                prompt = ""
                while not prompt.lower() == "exit":
                    sampler["move"] = destination_2
                    prompt = input("adjust z (or exit)>")
                    try:
                        destination_2.z += float(prompt)
                    except ValueError:
                        pass
                destination.z = destination_2.z
                positions[name] = str(destination)
            sampler["move"] = needle_up
            sampler["feed"] = 2000.0
        print(f"Calibrated positions for {sampler}:")
        for name, position in positions.items():
            print(f"{name}: {position}")
        print("Note: these positions have not been updated in the config file!")


def calibrate_sample_holders(platform):
    samplers = [
        device for device in platform.devices.values() if isinstance(device, Sampler)
    ]
    for sampler in samplers:
        if not yes_no(f"Calibrate {sampler}?"):
            continue

        sampler_name = None
        for name, item in platform.devices.items():
            # also this should be available, maybe from platform.
            if item is sampler:
                sampler_name = name
                break
        while True:
            print("Go to...")
            holder_name = input("Holder:")
            if holder_name.lower() == "exit":
                break
            vial_name = input("Vial:")
            if vial_name.lower() == "exit":
                break

            samples = pd.DataFrame(
                {
                    "VialID": ["0x01"],
                    "VialName": ["test"],
                    "Volume": [0.0],
                    "Type": ["Sample"],
                    "Sampler": [sampler_name],
                    "Holder": [holder_name],
                    "Position": [vial_name],
                }
            )
            samples.set_index("VialID", inplace=True, verify_integrity=True)
            GenerateSampleDataframe.run(platform=platform, samples=samples)
            PumpSample.run(
                platform=platform,
                sample_id="0x01",
                volume=0.0,
                needle_position="outside",
                auxiliary_needle=False,
            )
            if yes_no(f"Plunge needle?"):
                PumpSample.run(
                    platform=platform,
                    sample_id="0x01",
                    volume=0.0,
                    needle_position="top",
                    auxiliary_needle=False,
                )


def change_lh_solvent(platform):
    sampler = platform["Sampler_cnc"]
    pump = platform["Sampler_pump"]
    Inject.run(
        platform=platform,
        sampler=sampler,
        injection_port_name="injection_waste",
        volume=0.0,
        flowrate=1.0,
        retract=False,
    )

    while True:
        user_prompt = input("run purge cycle (y/N)?")
        if user_prompt.lower() not in ("y", "yes"):
            break
        FillPump.run(pump=pump, fill=True)
        FillPump.run(pump=pump, fill=False, reverse=True)


def change_carrier_solvent(platform):
    pump = platform["Main_Pump_1"]
    ConnectInjectionPort.run(
        platform=platform,
        sampler=platform["Sampler_cnc"],
        port_name="injection_flow",
        inject=False,
    )
    pump["aux_valve_setpoint"] = "OFF"  # redirect to waste

    while True:
        user_prompt = input("run purge cycle (y/N)?")
        if user_prompt.lower() not in ("y", "yes"):
            break
        FillPump.run(pump=pump, fill=True)
        FillPump.run(pump=pump, fill=False, reverse=True, flowrate=4.0)

    if input("purge with N2?") in ("y", "yes"):
        platform["sv_injection_n2"]["valve"] = "open"
        input("Press enter when the whole line is clear of liquid")
        platform["sv_injection_n2"]["valve"] = "close"

    pump["aux_valve_setpoint"] = "ON"  # redirect to waste


def prime_pumps(platform):
    try:
        cycles = int(input("Cycles?"))
    except Exception:
        return

    run_all(
        PrimePump.get_thread(pump=platform["Main_Pump_1"], cycles=cycles),
        PrimePump.get_thread(pump=platform["Sampler_pump"], cycles=cycles),
    )


def infuse_manually(platform):
    """
    Set up so that the platform tubing can be disconnected and filled with a cleaning solution (manually).
    """
    run_all(
        Inject.get_thread(
            platform=platform,
            sampler=platform["Sampler_cnc"],
            injection_port_name="injection_flow",
            move_speed=3000.0,
            plunge_speed=200.0,
            volume=0.0,
            flowrate=1.0,
            retract=False,
        ),
        Inject.get_thread(
            platform=platform,
            sampler=platform["Collector_cnc"],
            injection_port_name="injection_waste",
            move_speed=3000.0,
            plunge_speed=200.0,
            volume=0.0,
            flowrate=1.0,
            retract=False,
        ),
    )
    ConnectInjectionPort.run(
        platform=platform,
        sampler=platform["Sampler_cnc"],
        port_name="injection_flow",
        inject=True,
    )
    input(
        "Fill with cleaning solution from the connection above the sampler loop. Press enter when done."
    )
    platform["Sampler_cnc"]["move"] = platform["Sampler_cnc"].travel_position
    platform["Collector_cnc"]["move"] = platform["Collector_cnc"].travel_position


def characterize_light_source(platform):
    """
    Measure electrical power of source at different power levels.
    """
    device = platform["Light_Array"]
    subdevice = platform["uflow_kessil"]
    results = pd.DataFrame({"set": [], "voltage": [], "current": []})

    device["enable"] = "ON"

    for intensity in range(101):
        subdevice["intensity"] = intensity
        sleep(5)
        voltage = 0.0
        current = 0.0
        for i in range(5):
            voltage += device["voltage"]
            current += device["current"]
        voltage = voltage / 5
        current = current / 5
        results.loc[intensity] = [intensity, voltage, current]
        print(results.to_string())

    results.to_csv("light_calibration.csv")

    subdevice["intensity"] = 0
    device["enable"] = "OFF"


def main_pump_calibration(platform: Platform):
    """
    Calibrate volumes and flowrates of main pump.

    @param platform: Platform
        The platform
    """
    pump = platform["Main_Pump_1"]
    FillPump.run(pump)
    PumpVolume.run(pump, 1500.0, flowrate=2.0)

    flowrates = [0.8, 0.4, 0.2, 0.1, 0.05]
    volumes = [3.2, 3.2, 3.2, 1.6, 1.6]
    repeat = 3
    data_dict = {"Volume": [], "Time": [], "Flowrate": []}
    for i in range(len(flowrates)):
        for j in range(repeat):
            input("ready?")
            volume = volumes[i] * 1000.0
            flowrate = flowrates[i]
            time_start = timer()
            PumpVolume.run(pump, volume=volume, flowrate=flowrate)
            time_end = timer()
            delta_time = time_end - time_start
            data_dict["Volume"].append(volume)
            data_dict["Time"].append(delta_time)
            data_dict["Flowrate"].append(flowrate)
            print(pd.DataFrame(data_dict).to_string())
            FillPump.run(pump)
    pd.DataFrame(data_dict).to_csv("calibration_results.csv")


if __name__ == "__main__":
    actions = [
        purge,
        calibrate_all_ps,
        assess_volumes,
        calibrate_injection_ports,
        calibrate_sample_holders,
        change_lh_solvent,
        change_carrier_solvent,
        prime_pumps,
        infuse_manually,
        characterize_light_source,
        main_pump_calibration,
    ]

    print("Welcome to the platform calibration and maintenance script!")

    print("Building platform...")
    platform = Platform()
    platform_name = "Perry"
    devices = [
        "Main_Pump_1",
        "Sampler_cnc",
        "Sampler_pump",
        "Collector_cnc",
        "Gpio_Array_1",
        "Phase_Sensor_Array_1",
        "Phase_Sensor_Array_2",
        # "Light_Array",
    ]
    platform.build(
        platform_name=platform_name,
        devices=devices,
        open_gui=True,
    )
    print("ready!")

    while True:
        print("\nSelect one of the actions below:")
        index = 0
        for action in actions:
            print(f"{index}\t{action.__name__.replace('_', ' ')}")
            index += 1
        response = input("Action index (type 'exit' to finish):")
        if response.lower() == "exit":
            break
        try:
            response = int(response)
            action = actions[response]
        except:
            print("Invalid index!")
            continue
        action(platform)

    print("Clearing platform...")
    platform.clear()
    print("Done")

"""
File: manual_control.py
Author: Simone Pilon - Noël Research Group - 2024
GitH0ub: https://github.com/simone16

Description: Interactive script to control automation platform.
"""

from omniplatypus.devices.platform import Platform
from omniplatypus.devices.nrg.sampler import GrblPosition


if __name__ == "__main__":
    platform = Platform()
    platform_name = input("Platform name:")
    devices = input(
        "Which devices do you need? (separate names with comma, leave blank for all devices): "
    )
    if not devices == "":
        devices = devices.split(",")
        devices = [device.lstrip().rstrip() for device in devices]
    else:
        devices = None
    platform.build(platform_name=platform_name, devices=devices)
    print(
        "Activate a device with '$device_name', send commands with 'command_name: value'."
    )
    command = ""
    device = None
    if len(devices) == 1:
        device = platform[devices[0]]
    while not command == "exit":
        command = input(">").lstrip()
        if command.lower() == "exit":
            break
        if command[0] == "$":
            try:
                device = platform[command[1:].rstrip()]
            except Exception as e:
                print(e)
                device = None
        else:
            if device is not None:
                if ":" in command:
                    command, value = command.split(":", 1)
                    try:
                        parameter = device.parameter_by_name(command)
                        if parameter.value_type is GrblPosition:
                            value = value.split()
                            value = {v[0].lower(): float(v[1:]) for v in value}
                            value = GrblPosition(**value)
                        else:
                            value = parameter.value_type(value)
                        device[command] = value
                    except Exception as e:
                        print(e)
                else:
                    try:
                        print(device[command])
                    except Exception as e:
                        print(e)
    platform.clear()

"""
File: add_serial_device.py
Author: Simone Pilon - Noël Research Group - 2024
GitH0ub: https://github.com/simone16

Description: Interactive script to add known serial devices.
"""

from omniplatypus.devices.serial_id import KnownDevices


if __name__ == "__main__":
    known_devices = KnownDevices()
    known_devices.interactive_serial_scan()

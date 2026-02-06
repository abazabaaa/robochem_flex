"""
File: update_device_docs.py
Author: Simone Pilon - Noël Research Group - 2024
GitH0ub: https://github.com/simone16

Description: This script generates general documentation for all devices supported by this module.
"""

import os.path
from omniplatypus.devices.platform import Platform


if __name__ == "__main__":
    # Project assumes running directory to be project root, refuses to run if not.
    path_to_main = os.path.split(
        os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
    )[0]
    current_directory = os.getcwd()
    if not path_to_main == current_directory:
        print(
            "omniplatypus must run from project root.\nMake sure "
            "to cd or set the working directory properly and re-run."
        )
        exit(1)
    Platform.generate_device_docs(os.path.join("", "doc", "devices"))

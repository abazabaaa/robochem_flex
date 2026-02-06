"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from setuptools import setup, find_packages


setup(
    name="omniplatypus",
    version="1.0",
    packages=find_packages(),
    scripts=["omniplatypus/scripts/add_serial_device.py", "omniplatypus/scripts/manual_control.py", "omniplatypus/scripts/platform_calibration.py"],
    install_requires=[
        "bidict~=0.21.4",
        "pyserial~=3.5",
        "pandas~=2.2.1",
        "pause~=0.3",
        "pymodbus~=3.5.4",
        "colorama~=0.4.6",
        "slack_sdk~=3.27.1",
        "numpy~=1.26.4",
        "Pillow~=10.0.1",
        "matplotlib~=3.8.2",
        "scipy~=1.12.0",
        "bronkhorst-propar~=1.1.0",
        "paramiko~=4.0.0",
    ],
    data_files=[
        ("platform_config", ["omniplatypus/config/platform_config.json"]),
        ("sample_holder_types", ["omniplatypus/config/sample_holder_types.json"]),
        ("known_devices", ["omniplatypus/config/known_devices.json"]),
    ],
    description="The NRG's new and improved platform for Automated Chemistry.",
    author="dr. Simone Pilon, dr. Oliver Bayley and ir. Elia Savino",
)

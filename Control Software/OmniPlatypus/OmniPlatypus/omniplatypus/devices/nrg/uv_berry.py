"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

import os

import pandas as pd
import paramiko

from omniplatypus.devices.base.device import (
    DeviceParameter,
    ParameterAccess,
    ParameterCachingPolicy,
)
from omniplatypus.devices.base.device_socket import SocketDevice
from omniplatypus.devices.nrg.rama_berry import DataPollerStyle, RamaBerry


class UVBerry(RamaBerry):
    """
    Handles communication with RamaBerry device.
    Uses the RamaBerry library, wrapping the instrument class to implement the BaseDevice interface.

    Usage:
        First, connect to the device with
        `self.open()'

        then set the spectrometer parameters with:
        `self['parameter_name'] = xyz`

        or read the spectrometer parameters with:
        `xyz = self['parameter_name']`

        For a detailed list of available parameters, call
        `self.parameters()`

        For a human-readable list of parameters, call
        `print(self.usage())`
        (amongst parameters there is data as well so be careful)


        Finally, close the device with
        `self.close()`
    """

    def __init__(self):
        SocketDevice.__init__(self)

        self.generic_name = "UV-Vis Spectrometer"
        parameter = DeviceParameter(
            name="integration_time",
            access_level=ParameterAccess.W,
            value_type=float,
            internal_id=201,
        )
        parameter.description = "integration time of the spectrometer(ms)"
        parameter.units = "ms"
        parameter.caching_policy = ParameterCachingPolicy.ALWAYS
        parameter.max_value = 50000.0
        parameter.min_value = 100.0
        self.add_parameter(parameter)

        parameter = DeviceParameter(
            name="n_averages",
            access_level=ParameterAccess.RW,
            value_type=int,
            internal_id=202,
        )
        parameter.description = "number of averages"
        parameter.units = "int"
        parameter.caching_policy = ParameterCachingPolicy.ALWAYS
        parameter.max_value = 100
        parameter.min_value = 1
        self.add_parameter(parameter)

        parameter = DeviceParameter(
            name="delay",
            access_level=ParameterAccess.RW,
            value_type=float,
            internal_id=203,
        )
        parameter.description = "delay between measurements (ms)"
        parameter.units = "ms"
        parameter.caching_policy = ParameterCachingPolicy.ALWAYS
        parameter.max_value = 500000.0
        parameter.min_value = 0.0
        self.add_parameter(parameter)

        parameter = DeviceParameter(
            name="n_scans",
            access_level=ParameterAccess.RW,
            value_type=int,
            internal_id=204,
        )
        parameter.description = "number of scans"
        parameter.units = "int"
        parameter.caching_policy = ParameterCachingPolicy.ALWAYS
        parameter.max_value = 1000
        parameter.min_value = 1
        self.add_parameter(parameter)

        parameter = DeviceParameter(
            name="save_as",
            access_level=ParameterAccess.RW,
            value_type=DataPollerStyle,
            internal_id=205,
        )
        parameter.description = "return type defines shape of return data values (server_kinetic, server_single)"
        parameter.caching_policy = ParameterCachingPolicy.ALWAYS
        self.add_parameter(parameter)

        parameter = DeviceParameter(
            name="absorbance",
            access_level=ParameterAccess.RW,
            value_type=bool,
            internal_id=206,
        )
        parameter.description = "If return data should be in absorbance. If True, you will need to run blank measurement"
        parameter.caching_policy = ParameterCachingPolicy.ALWAYS
        self.add_parameter(parameter)

        parameter = DeviceParameter(
            name="data",
            access_level=ParameterAccess.R,
            value_type=pd.DataFrame,
            internal_id=207,
        )
        parameter.description = "spectral data"
        parameter.caching_policy = ParameterCachingPolicy.NEVER
        self.add_parameter(parameter)

        parameter = DeviceParameter(
            name="start_acq",
            access_level=ParameterAccess.RW,
            value_type=bool,
            internal_id=208,
        )
        parameter.description = "start acquisition, set to true to start, will go back to false when acq is started"
        parameter.caching_policy = ParameterCachingPolicy.ALWAYS
        self.add_parameter(parameter)

        parameter = DeviceParameter(
            name="stop_acq",
            access_level=ParameterAccess.RW,
            value_type=bool,
            internal_id=209,
        )
        parameter.description = "stop acquisition, set to true to stop, will go back to false when acq is stopped"
        parameter.caching_policy = ParameterCachingPolicy.ALWAYS
        self.add_parameter(parameter)

        parameter = DeviceParameter(
            name="take_blank",
            access_level=ParameterAccess.RW,
            value_type=bool,
            internal_id=210,
        )
        parameter.description = (
            "take blank measurement, set to true to take blank measurement"
        )
        parameter.caching_policy = ParameterCachingPolicy.ALWAYS
        self.add_parameter(parameter)

        parameter = DeviceParameter(
            name="save_moniker",
            access_level=ParameterAccess.RW,
            value_type=str,
            internal_id=211,
        )
        parameter.description = (
            "Save name addition, things are saved as Moniker_YYYY_MM_DD_HHMMSSsss in the folder"
            "robochem_spectra/YYYYMMDD"
        )
        parameter.caching_policy = ParameterCachingPolicy.ALWAYS
        self.add_parameter(parameter)
        parameter = DeviceParameter(
            name="save_path",
            access_level=ParameterAccess.RW,
            value_type=str,
            internal_id=212,
        )
        parameter.description = (
            "Data folder to save stuff on, default is in the cloud place"
        )
        parameter.caching_policy = ParameterCachingPolicy.ALWAYS
        self.add_parameter(parameter)

    def _connect_socket(self, host: str, port: int):
        """Connects to the server once it's up and running"""
        SocketDevice.open(self, host, port)
        # send the spectrometer setup command to the server
        """setup the spectrometer"""
        response = self._send_command("spectrometer_setup", data="uv")
        if response["status"] != "success":
            raise Exception(f"Failed to setup spectrometer: {response['message']}")

    def _start_server(self, host: str, username: str):
        """runs the ssh command to start the server on the spectrometer
        BIG BOY REQUIREMENT: you have to have an id_rsa and or a ed25519 key in your .ssh folder
        that allows to connect to the raspberry pi without a password!!!
        """
        self.log("Starting the server on the device", indent="enter")
        try:
            key_filename = os.path.expanduser("~/.ssh/id_rsa")
            self.log(f"Using key file: {key_filename}")
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            private_key = paramiko.RSAKey.from_private_key_file(key_filename)
            self.log("Client created and key loaded", level="ok")
        except Exception as e:
            self.log(f"Failed to load the private key: {e}", level="error")
            raise Exception(f"Failed to load the private key: {e}")

        try:
            ssh.connect(host, 22, username=username, pkey=private_key)
            self.log("Connected to the device", level="ok")
            command = """
            mkdir -p /mnt/Raman_Data/logs/log/Ramaberry_server/$(date +'%Y%m%d') &&
            cd Documents/GitHub/Spectrometer Interface &&
            source /home/Raman/miniforge3/etc/profile.d/conda.sh &&
            conda activate ramaberry_server_env &&
            nohup python server_app/RamaBerry_listener.py > /mnt/Raman_Data/logs/log/Ramaberry_server/$(date +'%Y%m%d')/$(date +'%H%M%S')-log.out 2>&1 &
            """
            stdin, stdout, stderr = ssh.exec_command(command, timeout=10)
            self.log("Ran the startup command:", indent="enter")
            exit_status = stdout.channel.recv_exit_status()

            if exit_status == 0:
                self.log("Command executed successfully", level="ok")
            else:
                self.log(
                    f"Command failed with exit status {exit_status}", level="error"
                )
                raise Exception(f"Command failed with exit status {exit_status}")
        except Exception as e:
            self.log(f"Failed to start the server: {e}", level="error")
            raise Exception(f"Failed to start the server: {e}")
        finally:
            ssh.close()

        self.log("Server started", indent="exit")

    def open(self, host: str, port: int):
        """Opens the connection to the device."""
        self._start_server(host, "Raman")
        self._connect_socket(host, port)
        self._initialized = True

    def _write(self, parameter: DeviceParameter, value):
        """
        writes the given parameter to the device. needs to check which parameter
        is being written and then send the appropriate command to the server.
        """
        if parameter.name == "start_acq" and value is True:
            self._start_acq(parameter)
        elif parameter.name == "stop_acq" and value is True:
            self._stop_acq(parameter)
        elif parameter.name == "take_blank" and value is True:
            self._take_blank(parameter)
        elif parameter.name == "save_as":
            self._set_parameter(parameter, value.value.lower())
        else:
            self._set_parameter(parameter, value)

        self.log("write completed", indent="exit")

    def _take_blank(self, parameter: DeviceParameter):
        response = self._send_command("take_blank")
        if response["status"] != "success":
            self.log(
                f"Unable to take blank measurement: {response['message']}",
                level="error",
            )
            raise Exception(f"Failed to take blank measurement: {response['message']}")
        else:
            parameter.value = False
            parameter.last_known_value = False

    def _read(self, parameter: DeviceParameter):
        """reads the given parameter from the device"""
        if parameter.name == "data":
            return self._poll_data()
        elif (
            parameter.name == "start_acq"
            or parameter.name == "stop_acq"
            or parameter.name == "take_blank"
        ):
            return {"status": "booleans", "value": parameter.last_known_value}

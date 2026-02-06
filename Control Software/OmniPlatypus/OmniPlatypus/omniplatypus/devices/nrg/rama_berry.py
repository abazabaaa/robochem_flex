"""
Author: Elia Savino
GitHub: github.com/EliaSavino

Happy Hacking!

Descr: interface to Raman spectrometer through socket and RamaBerryPy server app
"""

import json
import os
import time
from typing import Any

import pandas as pd
import paramiko
from bidict import bidict

from omniplatypus.devices.base.device import (
    DeviceParameter,
    ParameterAccess,
    ParameterCachingPolicy,
    ParameterEnumValue,
)
from omniplatypus.devices.base.device_socket import SocketDevice


class DataPollerStyle(ParameterEnumValue):
    """
    Describes the possible values for getting the data out of the spectrometer
    server_single -> one spectrum at a time
    server_kinetic -> a series of spectra at a time
    """

    _raw_values = bidict(
        {"SERVER_SINGLE": "SERVER_SINGLE", "SERVER_KINETIC": "SERVER_KINETIC"}
    )


class RamaBerry(SocketDevice):
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

        self.generic_name = "Raman Spectrometer"

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
            name="dark_correction",
            access_level=ParameterAccess.RW,
            value_type=bool,
            internal_id=206,
        )
        parameter.description = "dark correction pre spectrum"
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
            name="save_moniker",
            access_level=ParameterAccess.RW,
            value_type=str,
            internal_id=210,
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
            internal_id=211,
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
        response = self._send_command("spectrometer_setup", data="raman")
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

    def close(self):
        """Closes the connection to the device."""
        # send the shutdown command
        self.log("Sending shutdown command", indent="enter")
        response = self._send_command("shutdown")
        if response["status"] != "success":
            self.log(
                f"Shutdown unsuccessful response: {response['message']}. Better go get a hammer!",
                level="error",
            )

            raise Exception(f"Failed to shutdown: {response['message']}")
        else:
            self.log(response["message"], level="ok", indent="exit")
        SocketDevice.close(self)

    def _send_command(self, command: str, data=None):
        """Sends a command to the device."""
        self.log(f"Sending command {command}", indent="enter")
        payload = {"command": command}
        if data:
            self.log(f"Command payload: {data}")
            payload["data"] = data

        send_command = json.dumps(payload)
        self._socket.sendall(send_command.encode())
        data_len = int.from_bytes(self._socket.recv(4), "big")
        data = b""
        while len(data) < data_len:
            chunk = self._socket.recv(1024)
            if not chunk:
                break
            data += chunk

        response = data.decode()
        try:
            self.log(f"Response returned: {response}")
            response_dict = json.loads(response)
            self.log(f"Decoded response: {response}", indent="exit")
            return response_dict
        except json.JSONDecodeError:
            self.log("Error decoding response", level="error")
            raise Exception(f"Failed to decode response: {response}")

    def _write(self, parameter: DeviceParameter, value):
        """
        writes the given parameter to the device. needs to check which parameter
        is being written and then send the appropriate command to the server.
        """
        if parameter.name == "start_acq" and value is True:
            self._start_acq(parameter)

        elif parameter.name == "stop_acq" and value is True:
            self._stop_acq(parameter)
        elif parameter.name == "save_as":
            self._set_parameter(parameter, value.value.lower())
        else:
            self._set_parameter(parameter, value)

        self.log("write completed", indent="exit")

    def _start_acq(self, parameter: DeviceParameter):
        response = self._send_command("start_acq")
        if response["status"] != "success":
            self.log(
                f"Unable to start acquisition: {response['message']}", level="error"
            )
            raise Exception(f"Failed to start acquisition: {response['message']}")
        else:
            parameter.value = False
            parameter.last_known_value = False

    def _stop_acq(self, parameter: DeviceParameter):
        response = self._send_command("stop_acq")
        if response["status"] != "success":
            self.log(
                f"Unable to stop acquisition: {response['message']}", level="error"
            )
            raise Exception(f"Failed to stop acquisition: {response['message']}")
        else:
            parameter.value = False
            parameter.last_known_value = False

    def _set_parameter(self, parameter: DeviceParameter, value):
        """sets the given parameter to the device"""
        response = self._send_command("set_params", {parameter.name: value})
        if response["status"] != "success":
            raise Exception(
                f"Failed to set parameter {parameter.name}: {response['message']}"
            )

    def _read(self, parameter: DeviceParameter):
        """reads the given parameter from the device"""
        if parameter.name == "data":
            return self._poll_data()
        elif parameter.name == "start_acq" or parameter.name == "stop_acq":
            return {"status": "booleans", "value": parameter.last_known_value}

    def _poll_data(self):
        delay: int = (
            self.parameter_by_name(name="integration_time").last_known_value * 3 / 1000
        )
        something_went_wrong_counter: int = 1000
        counter = 0
        while something_went_wrong_counter >= 0:
            time.sleep(delay)
            response = self._send_command("poll_data")
            match response["status"]:
                case "success":
                    if response["message"] == "waiting":
                        self.log(
                            "Spectrometer has not finished measuring yet, please wait"
                        )
                    elif (
                        response["message"] == "data acquired"
                        and response["data"] == "no data available"
                    ):
                        self.log(
                            "spectrometer may have not have started to measure yet, possibly doing dark"
                        )
                        counter += 1
                        if counter >= 5:
                            self.log(
                                "something went wrong in measurement",
                                level="error",
                                indent="reset",
                            )
                            raise Exception("something went wrong in measurement!")
                    elif response["message"] == "done":
                        self.log("Spectrometer has no more data for this measurement")
                        return "done"
                    elif response["message"] == "data acquired":
                        self.log(
                            "Spectrometer has returned data", level="ok", indent="exit"
                        )
                        return response
                    else:
                        self.log("How the fuck did you get here?", level="error")
                        raise Exception("Unknown spectrometer return!")
                case "fail":
                    self.log(
                        "Spectrometer not measuring, and no data available, sorry!"
                    )
                    return None
                case "error":
                    self.log(f"Error in data polling {response['message']}")
                    raise Exception(f"error in data polling: {response['message']}")
                case _:
                    self.log("Unknown spectrometer return!")
                    raise Exception(
                        f"Data not in the right format, I have no idea how you got here"
                    )

            # time.sleep(delay)
            something_went_wrong_counter -= 1

        self.log("Data Polling timed out", level="error", indent="reset")

    def _extract(self, parameter: DeviceParameter, response: Any):
        """
        extracts the given value from the device
        the responses can be:
        - a string that says  done
        - a None which means the data is not ready yet
        - an exception which should be raised/thrown
        - a dictionary with the data that should be turned into a df.
        """
        if response == None:
            return None
        if isinstance(response, Exception):
            self.log(f"Response was exception:{response}", level="error")
            raise response

        status = response.get("status", "error")
        message = response.get("message", None)
        data = response.get("data", None)

        if status == "error":
            self.log(
                f"Error in reading value from parameter parameter response: {message}",
                level="error",
            )
            raise Exception(
                f"Error in reading value from parameter parameter response: {message}"
            )
        elif status == "failure":
            self.log(
                f"Failure in reading value from parameter {parameter}, response: {message}",
                level="warning",
            )
            return None
        elif status == "success":
            self.log(
                f"successfully read parameter {parameter} value, extracting",
                indent="enter",
            )
            self.log(f"status: {status}")
            self.log(f"message: {message}")
            if isinstance(data, dict):
                data = pd.DataFrame(data)
                self.log(
                    f"data was dataframe, with columns: {data.columns}", indent="exit"
                )
                return data
            elif data == "done":
                self.log(f"No more spectras to wait for, data: done", indent="exit")
                return data
            elif isinstance(data, str):
                data = pd.DataFrame(json.loads(data))
                self.log("Data transformed from str to DF", indent="exit")
                return data
            else:
                self.log(f"data: {data}", indent="exit")
                return data

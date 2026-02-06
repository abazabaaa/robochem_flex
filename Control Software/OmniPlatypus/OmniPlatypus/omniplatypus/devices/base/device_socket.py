"""
File: device_socket.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: Extends device.py with additional behaviour to control socket-based devices.
"""

import socket

from omniplatypus.devices.base.device import BaseDevice


class SocketDevice(BaseDevice):
    """
    Implements general socket-based communication for devices.
    """

    _socket: socket.socket | None

    def __init__(self):
        BaseDevice.__init__(self)
        self.generic_name = "Socket device"
        self._socket = None
        self._socket_data = None

    def open(self, host: str, port: int):
        """
        Connects to the device. The connection must open before data can be sent or received.

        @param host: str
            Hostname for the server device (e.g.: '194.168.1.120').
        @param port: int
            Port number for the server device.
        """
        self._socket_data = (host, port)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect(self._socket_data)

    def reopen(self):
        """
        Connects to the device after the connection was made a first time.
        Does not require to re-input the connection data.
        """
        if self._socket_data is None:
            raise RuntimeError(
                f"{self.complete_name}: connection cannot re-open before being opened."
            )
        self.open(self._socket_data[0], self._socket_data[1])

    def is_open(self) -> bool:
        """
        Check whether the connection is open.

        @return: bool
            True if the connection is open.
        """
        return self._socket is not None

    def close(self) -> None:
        """
        Disconnects the device.
        If you need to run some operations before closing, override self.prepare_for_close(),
        rather than this.
        """
        self._socket.close()
        self._socket = None

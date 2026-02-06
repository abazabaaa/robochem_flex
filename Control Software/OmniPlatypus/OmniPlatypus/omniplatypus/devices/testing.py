"""
File: testing.py
Author: Simone Pilon - Noël Research Group - 2023
GitHub: https://github.com/simone16

Description: Utilities for testing of the device module.
"""

from time import sleep

from omniplatypus.utilities.logger import Logger


class DummySerial:
    """
    Emulates pyserial serial interface without writing anything to actual serial ports.
    Used for testing only.
    """

    def __init__(self):
        self.is_open = False
        self.replies = []
        self.mute = False
        self.read_delay = 0.05

    @property
    def in_waiting(self):
        try:
            value = len(self.replies[0])
            if value == 0:
                self.replies.pop(0)
        except IndexError:
            value = 0
        return value

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False

    def read(self, bytes):
        if len(self.replies) == 0:
            raise RuntimeError("Ran out of replies!")
        message = self.replies.pop(0)
        delay = self.read_delay
        if isinstance(message, tuple):
            message, delay = message
        sleep(delay)
        if not self.mute:
            self._log("READ <- " + message.rstrip("\r\n"))
        return message.encode("ascii")

    def readline(self):
        return self.read(10)

    def write(self, message):
        if not self.mute:
            self._log(f"WRITTEN -> {message}")

    def reply(self, *args):
        for arg in args:
            self.replies.append(arg)

    @classmethod
    def _log(cls, message):
        Logger.log_message(message, origin=cls.__name__)

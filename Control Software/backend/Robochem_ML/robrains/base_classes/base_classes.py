"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from typing import Literal, Optional, Any

from robrains.base_classes.logger import Logger

from os import getcwd
from os.path import join


class BaseLoggedClass:
    """default class to build all functions in the project, has logger always available, has a default error handling
    strategy."""

    logger = Logger.log_message
    _LOGGER = Logger
    _hide = False
    _locked = False

    @classmethod
    def turn_off_logger(cls):
        """
        Turn off the logger.
        """
        cls._LOGGER._logger_on = False

    @classmethod
    def log_setter(cls, log_directory: str, lock: Optional[bool] = False) -> None:
        """
        Set the log directory for the logger.
        :param log_directory: str, the directory where the logs should be stored.
        :param lock: the first time this is passed as true this won't allow to change the directory anymore, will not
        raise warnings or anything, just won't change the directory
        """
        if cls._locked:
            cls.log_mssg("Logger directory is locked, cannot change it anymore", level="warning")
            return

        if lock:
            cls._locked = True

        cls._LOGGER._log_directory = {"log_directory": log_directory}


    def log_mssg(
        cls,
        message: Any,
        level: Literal["error", "warning", "ok", "none"] = "none",
        indent: Optional[int] = 0,
        hide: Optional[bool] = None,
    ) -> None:
        """
        Log a message through the configured logger.

        :param message: The message to log.
        :param level: The level of the message. Allowed values: 'error', 'warning', 'ok', 'none'.
        :param indent: Number of indent levels for hierarchical logging.
        """
        # Ensure warnings and errors are always shown in console
        if level in ("warning", "error"):
            hide = False
        elif hide is None:
            hide = cls._hide

        message_string = f"{message}"
        cls.logger(
            message_string,
            origin=cls.__class__.__name__,
            level=level,
            indent=indent,
            hide=hide,
            log_directory=cls._LOGGER._log_directory["log_directory"],
        )

    @classmethod
    def assertion_method(
        cls, obj: any, condition: callable, message: str = f"Assertion failed"
    ):
        """
        Assert that the object follows the condition, if not logs and raises error, this is done to properly log everything
        :param obj: any, object to check
        :param condition: callable, condition to check in the form lambda x: c > 0
        :param message: str, message to log if the assertion fails, if none base message is used
        """

        if not condition(obj):
            cls.log_mssg(message, level="error")
            raise AssertionError(message)


class BaseParamClass(BaseLoggedClass):
    """base class for parameters, has the getattr, setattr get and del methods defaulted with logger and error raising"""

    def __getitem__(self, key):
        try:
            return getattr(self, key)
        except AttributeError as e:
            self.log_mssg(f"Key {key} not found in parameter", level="error")
            raise e

    def __setitem__(self, key, value):
        try:
            setattr(self, key, value)
        except Exception as e:
            self.log_mssg(f"Could not set {key} to {value}", level="error")
            raise e

    def __delitem__(self, key):
        try:
            delattr(self, key)
        except AttributeError as e:
            self.log_mssg(f"Could not delete {key}", level="error")
            raise e

    def get(self, key, default=None):
        try:
            return getattr(self, key)
        except Exception as e:
            self.log_mssg(f"Key {key} not found in parameter", level="warning")
            return default

    def __lt__(self, other):
        return self.name < other.name

    def __eq__(self, other):
        return self.name == other.name

    def __le__(self, other):
        return self.name <= other.name

    def __gt__(self, other):
        return self.name > other.name

    def __ge__(self, other):
        return self.name >= other.name

    def __ne__(self, other):
        return self.name != other.name

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

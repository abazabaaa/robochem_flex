"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: logger and base logged class for all the functions and tasks in the project

"""

import os
from time import strftime
from os import getcwd, mkdir, environ
from os.path import isdir, join
from colorama import Fore, Style
from threading import Lock


class Logger:
    """Container for logging-relate methods."""

    _lock = Lock()

    @staticmethod
    def _logger_warning(message: str) -> None:
        """Print a warning message to the command line.
        Used by the logger in case of internal issues.

        :param message: str
            The message to print.
        """
        print(Fore.YELLOW + f"[WARNING] [Logger] " + message + Style.RESET_ALL)

    _forbidden_in_filename = "\\(){}[]/^`'\"\n\r"

    @classmethod
    def _print(cls, message: str, **kwargs) -> None:
        """prints a message using the lock"""
        with cls._lock:
            return print(message)

    @classmethod
    def _sanitize_file_name(cls, name: str) -> str:
        """Remove forbidden characters from file names or folder names.

        :param name: str
            The input name of file or folder.
        :return: str
            The name without the forbidden characters."""
        for character in cls._forbidden_in_filename:
            name = name.replace(character, "")
        return name

    _priority_levels = {
        "error": ("  [ERROR] ", Fore.RED),
        "warning": ("[WARNING] ", Fore.YELLOW),
        "ok": ("     [OK] ", Fore.GREEN),
    }

    @classmethod
    def _decorate(cls, message: str, **kwargs) -> tuple[str, str]:
        """Take an undecorated message and return two decorated versions, one for terminal and one for logfiles.

        :return: tuple[str, str]
            A decorated message for terminal and one for logfiles (in this order)."""
        indent = ""
        if "indent" in kwargs.keys():
            symbol_term = "├"
            symbol_con = "│ "
            if "indent_symbol" in kwargs.keys():
                if kwargs["indent_symbol"] == "inside":
                    symbol_term = "├"
                elif kwargs["indent_symbol"] == "last":
                    symbol_term = "└"
                elif kwargs["indent_symbol"] == "reset":
                    symbol_term = "┴"
                    symbol_con = "┴─"
                else:
                    raise ValueError(
                        f"Invalid indent symbol option '{kwargs['indent_symbol']}'."
                    )
            if kwargs["indent"] >= 1:
                indent = symbol_con * (kwargs["indent"] - 1) + symbol_term
        timecode = strftime("%H:%M:%S") + " "
        start_code = ""
        priority = "          "
        level = ""
        if "level" in kwargs.keys() and not kwargs["level"] == "none":
            level = kwargs["level"]
            if level not in cls._priority_levels.keys():
                raise ValueError(f"Invalid logging level option '{level}'.")
        if level:
            start_code = cls._priority_levels[level][1]
            priority = cls._priority_levels[level][0]
        origin = ""
        if "origin" in kwargs.keys():
            origin = kwargs["origin"]
        colored_origin = origin
        if not origin == "":
            origin = "[" + origin + "] "
            colored_origin = (
                Style.RESET_ALL + Fore.BLUE + origin + Style.RESET_ALL + start_code
            )
        colored_message = message
        colored_message = colored_message.replace(
            "[", Style.RESET_ALL + Fore.CYAN + "["
        )
        colored_message = colored_message.replace(
            "]", "]" + Style.RESET_ALL + start_code
        )
        colored_message = (
            priority + timecode + colored_origin + indent + colored_message
        )
        colored_message = start_code + colored_message + Style.RESET_ALL
        message = priority + timecode + origin + indent + message
        return colored_message, message

    @classmethod
    def log_message(cls, message: str, **kwargs) -> None:
        """Print messages to the console and store logs.
        All messages will appear on console and on the common log file.
        If an origin is specified in the kwargs, the message is stored in a dedicated log as well.

        :param message: str
            The message to be given to the user.
        :param kwargs:
            See below.

        :Keyword Arguments:
            * origin (``str``) --
              The class or object that originated the message.
              Default: no origin
            * subfolder (``str``) --
              The sub-folder that the origin should appear in (e.g.: 'devices').
              Note this must be used in combination with 'origin'.
              Default: no subfolder
            * level (``str``) --
              Warning level of the message: influences how it is highlighted.
              Accepted values:
                - 'error'
                - 'warning'
                - 'ok'
                - 'none'
              Default: 'none'
            * decorate (``bool``) --
              If False the message will be printed without additional formatting/info.
              Default: True
            * indent (``int``) --
              The message should appear to pertain to a previous one by indenting this many levels.
              Default: 0
            * indent_symbol (``str``) --
              A few options for indent symbols can be specified:
                - 'inside'
                - 'last' (if indent level will decrease after this message).
                - 'reset' (if indent level will reset after this message).
              Default: 'inside'
            * hide (``bool``) --
              If true, the message is logged to files only and not shown on console.
              Default: False
            * slack (```bool``) --
              If true, the message is always sent to slack, if false, never.
              Default: Only error messages are sent to slack.
        """
        log_directory = join(getcwd(), "log")
        if not isdir(log_directory):
            mkdir(log_directory)
        log_files = [join(log_directory, strftime("%Y-%m-%d.txt"))]
        if "subfolder" in kwargs.keys():
            subfolder = cls._sanitize_file_name(kwargs["subfolder"])
            log_directory = join(log_directory, subfolder)
            if not isdir(log_directory):
                mkdir(log_directory)
        if "origin" in kwargs.keys():
            origin = kwargs["origin"]
            log_directory = join(log_directory, cls._sanitize_file_name(origin))
            if not isdir(log_directory):
                mkdir(log_directory)
            log_files.append(join(log_directory, strftime("%Y-%m-%d.txt")))
        if ("decorate" in kwargs.keys()) and not kwargs["decorate"]:
            dull_message = message
            colored_message = message
        else:
            colored_message, dull_message = cls._decorate(message, **kwargs)
        if not ("hide" in kwargs.keys() and kwargs["hide"]):
            cls._print(colored_message)
        try:
            for file in log_files:
                with open(file, "a", encoding="utf-8") as log:
                    log.write(dull_message + "\n")
        except NameError as e:
            cls._print(f"{Fore.YELLOW}{e}")
            cls._print(f"{Fore.YELLOW}[logger.py] Could not log to files.")
            cls._print(
                f"Make sure all devices are explicitly closed before the program terminates.{Style.RESET_ALL}"
            )


class BaseLoggedClass:
    """default class to build all functions in the project, has logger always available, has a default error handling
    strategy."""

    logger = Logger.log_message

    @classmethod
    def log_mssg(cls, message: str, level: str = "warning") -> None:
        """
        Log a message.
        :param message: The message to log.
        :param level: The level of the message. allowed: 'error', 'warning', 'ok', 'none'
        """
        cls.logger(message, origin=cls.__name__, level=level)

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

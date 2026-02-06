"""
File: logger.py
Author: Simone Pilon - Noël Research Group - 2023
GitHub: https://github.com/simone16

Description: Utilities for storing and logging error and status messages from all devices.
"""

import copy
import os
from time import strftime
from os import getcwd, makedirs, environ
from os.path import isdir, join
from threading import Thread, current_thread
from queue import Queue
from time import sleep

from colorama import Fore, Style
from slack_sdk import WebClient

from omniplatypus.utilities.gui import Gui, GuiError


class Logger:
    """
    Provides logging functions for devices, platforms and unit tasks.
    Logging is performed asynchronously on a dedicated thread, so minimal delay is caused by a log request.
    """

    _forbidden_in_filename = "\\(){}[]/^`'\"\n\r"
    _priority_levels = {
        "error": ("  [ERROR] ", Fore.RED),
        "warning": ("[WARNING] ", Fore.YELLOW),
        "ok": ("     [OK] ", Fore.GREEN),
        "none": ("          ", Fore.WHITE),
    }
    _indent_symbols = {
        "left_continue": "│ ",
        "left_terminate": "┴─",
        "right_continue": "├",
        "right_end": "└",
        "right_terminate": "┴",
    }
    _platform = "Platform"
    _log_requests = Queue()
    _origins = {}
    _slack_token = None
    _slack_client = None
    _main_thread = None
    _logging_thread = None
    _running = False
    _logs_directory = ""
    _master_log = ""
    _use_console = False

    @classmethod
    def start_logging_thread(
        cls,
        platform: str = "Platform",
        logs_path: str | None = None,
        use_slack: bool = False,
        use_console: bool = False,
        use_gui: bool = True,
        use_mood: bool = True,
    ) -> None:
        """
        Starts the logging machinery.
        Logging happens on a dedicated thread and calls are placed on a queue. This thread needs to be started once
        (here).

        @param platform: str
            Name of the platform this logger is tied to.
        @param logs_path: str | None = None
            Path to desired logs location.
            A 'log' subfolder will be created there.
        @param use_slack: bool
            If false, no messages will be sent to slack, no matter what.
        @param use_console: bool = False
            If false, no messages will be sent to console, no matter what.
            Note: this is currently causing issues and should not be set to True.
        @param use_gui: bool = True
            If false, the dedicated gui window will not be activated.
        @return: None
        """
        cls._platform = platform
        cls._use_console = use_console
        cls._use_gui = use_gui

        if use_gui:
            Gui.start_gui(platform=platform)

        if logs_path is None:
            logs_path = getcwd()
        cls._logs_directory = join(logs_path, "log", cls._platform)
        if not isdir(cls._logs_directory):
            makedirs(cls._logs_directory)
        cls._master_log = join(cls._logs_directory, strftime("%Y-%m-%d.txt"))

        if use_slack:
            if "SLACK_TOKEN" in os.environ.keys():
                cls._slack_token = environ["SLACK_TOKEN"]
                cls._slack_client = WebClient(token=cls._slack_token)
            else:
                cls.log_message(
                    "No slack client token found! Slack notifications will be disabled.",
                    origin="Logger",
                    level="warning",
                )

        if cls._logging_thread is None or not cls._logging_thread.is_alive():
            cls._running = True
            cls._main_thread = current_thread()
            cls._logging_thread = Thread(target=cls._log_queued, name="Logging thread")
            cls._logging_thread.start()
            cls.log_message("Logger started.", origin="Logger")
        else:
            cls.log_message(
                "Attempt to start Logger thread while it is already running.",
                origin="Logger",
                level="warning",
            )

    @classmethod
    def stop_logging_thread(cls) -> None:
        """
        Stops the logger thread.
        Ensures all remaining requests are handled.
        """
        cls._running = False
        assert isinstance(cls._logging_thread, Thread)
        cls._logging_thread.join()
        cls._log_message(message="Logger stopped.", origin="Logger")
        if cls._use_gui:
            Gui.stop_gui()

    @classmethod
    def _sanitize_file_name(cls, name: str) -> str:
        """
        Remove forbidden characters from file names or folder names.
        Used to get a proper file or folder name from devices and tasks.

        @param name: str
            The input name of file or folder.
        @return: str
            The name without the forbidden characters.
        """
        for character in cls._forbidden_in_filename:
            name = name.replace(character, "")
        return name

    @classmethod
    def _message_to_tokens(cls, **kwargs) -> list[str]:
        """
        Take an undecorated message and add decorations as separate tokens.

        @param kwargs:
            See _log_message for a list of arguments.
        @return: list[str]
            A list of strings which, when concatenated, give the decorated message.
        """
        origin = kwargs.get("origin", "")
        message = kwargs.get("message", "")
        timestamp = kwargs.get("timestamp", "")
        level = kwargs.get("level", "none").lower()

        indentation = ""
        indent = kwargs.get("indent", "continue").lower()
        if cls._origins[origin]["indent"] > 0:
            symbol_right = Logger._indent_symbols["right_continue"]
            symbol_left = Logger._indent_symbols["left_continue"]
            if indent == "exit":
                symbol_right = Logger._indent_symbols["right_end"]
            elif indent == "reset":
                symbol_right = Logger._indent_symbols["right_terminate"]
                symbol_left = Logger._indent_symbols["left_terminate"]
            indentation = (
                symbol_left * (cls._origins[origin]["indent"] - 1) + symbol_right
            )

        try:
            priority, start_code = Logger._priority_levels[level]
        except KeyError:
            raise ValueError(f"Invalid logging level option '{level}'.")

        tokens = [priority + timestamp + "[", origin, "] "]

        # indent each line if multi-line
        if "\n" in message:
            base_indentation = indentation
            if len(base_indentation) > 0:
                if base_indentation[-1] == Logger._indent_symbols["right_continue"]:
                    base_indentation = base_indentation[:-1] + "│"
                else:
                    base_indentation = base_indentation[:-1] + " "
            multi_line_indentation = (
                "\n" + sum(len(token) for token in tokens) * " " + indentation
            )
            message = message.replace("\n", multi_line_indentation)

        tokens[2] += indentation + message
        return tokens

    @classmethod
    def _gui_tokens(cls, tokens: list[str], **kwargs) -> list[tuple[str, str]]:
        """
        Add tags to tokens to be sent to gui log window.

        @param tokens: list[str]
            Tokenized message.
        @param kwargs:
            See _log_message for a list of arguments.
        @return: list[tuple[str, str]]
            List of token, tag tuples.
        """
        level = kwargs.get("level", "none").lower()
        folder = kwargs.get("subfolder", "")

        origin_tag = "origin"
        if folder == "devices":
            origin_tag = "device"
        elif folder == "tasks":
            origin_tag = "task"
        return [
            (tokens[0], level),
            (tokens[1], origin_tag),
            (tokens[2] + "\n", level),
        ]

    @classmethod
    def _console_message(cls, **kwargs) -> str:
        """
        Add colors to tokens and return a single string.

        @param kwargs: dict
            See '_log_message' for the supported keyword arguments.
        @return: str
            A decorated message for terminal.
        """
        message_tokens = kwargs.get("tokens", ["", "", ""])

        level = kwargs.get("level", "none").lower()
        try:
            priority, start_code = Logger._priority_levels[level]
        except KeyError:
            raise ValueError(f"Invalid logging level option '{level}'.")

        return (
            start_code
            + message_tokens[0]
            + Style.RESET_ALL
            + Fore.BLUE
            + message_tokens[1]
            + Style.RESET_ALL
            + start_code
            + message_tokens[2]
        )

    @classmethod
    def slack_notify(
        cls,
        message: str,
        channel: str = "voice_of_the_platypus",
    ) -> None:
        if cls._slack_client is None:
            return
        message = "```\n" + message + "```\n"
        cls._slack_client.chat_postMessage(
            channel=channel,
            text=message,
            username=cls._platform,
            mrkdwn=True,
        )

    @classmethod
    def _log_message(cls, **kwargs) -> None:
        """
        Print messages to the console and store logs.
        All messages will appear on console and on the common log file.
        If an origin is specified in the kwargs, the message is stored in a dedicated log as well.
        If multiple origins are given, the message will be stored in all logs, but the first origin will
        be used as 'primary' to craft the message.
        Note: this method writes to console and log files synchronously.

        @param kwargs:
            See below.
        @keyword message: str
              The message. Must be given.
        @keyword origin: str
          The origin. Must be given.
        @keyword timestamp: str
          Time stamp string. Must be given.
        @keyword subfolder: str
          The sub-folder that the origin should appear in (e.g.: 'devices').
          Default: no sub-folder
          Note: if the origin is known, this will be ignored and stored value will be used instead.
        @keyword level:str
          Warning level of the message: influences how it is highlighted and where it is displayed.
          Accepted values:
            - 'error'
            - 'warning'
            - 'ok'
            - 'none'
          Default: 'none'
        @keyword priority: int = 0
            Priority indicator used by GUI only to show the message based on user settings.
            Low numbers correspond to lower priority (less chance of the message being seen.
            Warning and errors are automatically given high priority.
        @keyword indent: str
          Messages can be indented to appear to be related to a previous message,
          for example the start of a routine.
          These options can be specified:
            - 'enter' (Increase indentation level by one).
            - 'continue' (Use current indentation level).
            - 'exit' (Decrease indentation level by one).
            - 'reset' (Exit all indentation levels back to 0).
          Default: None: specifying no indentation keeps the current level for the message.
        @keyword decorate: bool
          If False the message will be printed without additional formatting/info.
          Default: True
        @keyword hide: bool
          If true, the message is logged to files only and not shown on console.
          Default: False
        @keyword mute_logs: bool
          If true, the message is not logged to the files.
          Default: False
        @keyword slack: bool
          If true, the message is always sent to slack, if false, never.
          Default: Only error messages are sent to slack.
          Note: if Slack was not set up, this will be silently ignored.
        """
        origin = kwargs.get("origin", None)
        if origin is None:
            cls.log_message(
                message="Attempt to log a message with no body or no origin.",
                origin="Logger",
                level="warning",
            )
        origin: str

        if origin not in cls._origins.keys():
            log_file = [cls._logs_directory]
            subfolder = kwargs.get("subfolder", None)
            if subfolder is not None:
                subfolder: str
                subfolder = Logger._sanitize_file_name(subfolder)
                log_file.append(subfolder)
            log_file.append(Logger._sanitize_file_name(origin))
            log_file = join(*log_file)
            if not isdir(log_file):
                makedirs(log_file)
            log_file = join(log_file, strftime("%Y-%m-%d.txt"))
            cls._origins[origin] = {
                "log_files": [log_file],
                "indent": 0,
                "trace": [],
            }

        message_tokens = cls._message_to_tokens(**kwargs)
        if cls._use_gui:
            if kwargs.get("level", "none") in ("error", "warning"):
                kwargs["priority"] = 10
            try:
                Gui.log_tokens(
                    *cls._gui_tokens(message_tokens, **kwargs),
                    priority=kwargs.get("priority", 0),
                )
            except GuiError:
                pass

        kwargs["tokens"] = message_tokens

        log_files = [cls._master_log]
        log_files += cls._origins[origin]["log_files"]

        decorate = kwargs.get("decorate", True)
        if decorate:
            colored_message = cls._console_message(**kwargs)
            dull_message = "".join(message_tokens)
        else:
            colored_message = kwargs.get("message", "")
            dull_message = colored_message

        stack_trace = None
        indent_action = kwargs.get("indent", "continue")
        if indent_action == "enter":
            cls._origins[origin]["trace"].append((colored_message, dull_message))
            cls._origins[origin]["indent"] += 1
        elif indent_action == "continue":
            if not cls._origins[origin]["indent"] == 0:
                cls._origins[origin]["trace"].append((colored_message, dull_message))
        elif indent_action == "exit":
            cls._origins[origin]["trace"].append((colored_message, dull_message))
            indent = cls._origins[origin]["indent"]
            indent -= 1
            if indent < 0:
                indent = 0
            if indent == 0:
                stack_trace = cls._origins[origin]["trace"]
                cls._origins[origin]["trace"] = []
            cls._origins[origin]["indent"] = indent
        elif indent_action == "reset":
            cls._origins[origin]["indent"] = 0
            cls._origins[origin]["trace"].append((colored_message, dull_message))
            stack_trace = cls._origins[origin]["trace"]
            cls._origins[origin]["trace"] = []

        if cls._use_console:
            if not ("hide" in kwargs.keys() and kwargs["hide"]):
                print(colored_message)
            else:
                if kwargs.get("level", "none") == "error":
                    if stack_trace is not None:
                        for colored, dull in stack_trace:
                            print(colored)
                    else:
                        print(colored_message)

        if not kwargs.get("mute_logs", False):
            for file in log_files:
                try:
                    with open(file, "a", encoding="utf-8") as log:
                        log.write(dull_message + "\n")
                except NameError as e:
                    message = str(e)
                    message += "\nCould not log to files."
                    message += "\nMake sure all devices are explicitly closed before the program terminates."
                    for line in message.splitlines():
                        cls.log_message(
                            line,
                            origin="Logger",
                            level="warning",
                            mute_logs=True,
                        )

        if cls._slack_client is not None:
            slack_message = ""
            if kwargs.get("level", "none") == "error":
                if stack_trace is not None:
                    for colored, dull in stack_trace:
                        slack_message += dull + "\n"
                else:
                    slack_message = dull_message
            elif kwargs.get("slack", False):
                slack_message = dull_message
            if not slack_message == "":
                cls.slack_notify(slack_message)


    @classmethod
    def log_message(
        cls, message: str | Exception, origin: str = "unknown", **kwargs
    ) -> None:
        """
        Print messages to the console and store logs.
        All messages will appear on console and on the common log file.
        If an origin is specified in the kwargs, the message is stored in a dedicated log as well.
        If multiple origins are given, the message will be stored in all logs, but the first origin will
        be used as 'primary' to craft the message.

        @param message: str | Exception
            The message to be given to the user or an exception (will not be raised here).
        @param origin: str
            Name of originating device or task.
        @param kwargs:
            See below.
        @keyword subfolder: str
          The sub-folder that the origin should appear in (e.g.: 'devices' or 'tasks').
          Default: no sub-folder
          Note: if the origin is known, this will be ignored and stored value will be used instead.
        @keyword level:str
          Warning level of the message: influences how it is highlighted and where it is displayed.
          Accepted values:
            - 'error'
            - 'warning'
            - 'ok'
            - 'none'
          Default: 'none'
        @keyword priority: int = 0
            Priority indicator used by GUI only to show the message based on user settings.
            Low numbers correspond to lower priority (less chance of the message being seen.
            Warning and errors are automatically given high priority.
        @keyword indent: str
          Messages can be indented to appear to be related to a previous message,
          for example the start of a routine.
          These options can be specified:
            - 'enter' (Increase indentation level by one).
            - 'continue' (Use current indentation level).
            - 'exit' (Decrease indentation level by one).
            - 'reset' (Exit all indentation levels back to 0).
          Default: 'continue'
        @keyword decorate: bool
          If False the message will be printed without additional formatting/info.
          Default: True
        @keyword hide: bool
          If true, the message is logged to files only and not shown on console.
          Default: False
        @keyword mute_logs: bool
          If true, the message is not logged to the files.
          Default: False
        @keyword slack: bool
          If true, the message is always sent to slack, if false, never.
          Default: Only error messages are sent to slack.
          Note: if Slack was not set up, this will be silently ignored.
        """
        if cls._running and not cls._logging_thread.is_alive():
            raise RuntimeError("Logging thread stopped unexpectedly.")
        if isinstance(message, Exception):
            text_message = message.__class__.__name__
            if hasattr(message, "parameter"):
                text_message += " on " + str(message.parameter)
            if len(message.args) >= 1:
                text_message += ": " + str(message)
            if hasattr(message, "device") and message.device is not None:
                origin = message.device.specific_name
            message = text_message
            if "indent" not in kwargs.keys():
                kwargs["indent"] = "reset"
            if "level" not in kwargs.keys():
                kwargs["level"] = "error"

        timestamp = strftime("%H:%M:%S")
        arguments = copy.deepcopy(kwargs)
        arguments["message"] = message
        arguments["origin"] = origin
        arguments["timestamp"] = timestamp

        cls._log_requests.put(arguments)

    @classmethod
    def _log_queued(cls) -> None:
        """
        Handles the messages queued by the users.
        This is meant to run in a dedicated thread, while log requests are made in the main thread.
        """
        while (
            cls._running and cls._main_thread.is_alive()
        ) or not cls._log_requests.empty():
            try:
                while not cls._log_requests.empty():
                    request_kwargs = cls._log_requests.get(block=True)
                    cls._log_message(**request_kwargs)
                    cls._log_requests.task_done()
                sleep(0.1)
            except Exception as e:
                print("Logger error!")
                print(e)

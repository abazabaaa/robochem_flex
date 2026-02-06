#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Author: O. Bayley
Description: function to get a custom Logger object (RbcLogger) that can both log and print
"""
import os
import logging
import time
from src.utils.get_project_path import get_project_path

class CustomLogger(logging.Logger):
    """
    A custom Logger that adds a `print_msg` keyword argument to its logging methods
    so you can choose whether to print to stdout as well.
    """

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)

    def debug(self, msg, *args, print_msg=False, **kwargs):
        kwargs['stacklevel'] = kwargs.get('stacklevel', 1) + 1
        super().debug(msg, *args, **kwargs)
        if print_msg:
            print(msg)

    def info(self, msg, *args, print_msg=False, **kwargs):
        kwargs['stacklevel'] = kwargs.get('stacklevel', 1) + 1
        super().info(msg, *args, **kwargs)
        if print_msg:
            print(msg)

    def warning(self, msg, *args, print_msg=False, **kwargs):
        kwargs['stacklevel'] = kwargs.get('stacklevel', 1) + 1
        super().warning(msg, *args, **kwargs)
        if print_msg:
            print(msg)

    def error(self, msg, *args, print_msg=False, **kwargs):
        kwargs['stacklevel'] = kwargs.get('stacklevel', 1) + 1
        super().error(msg, *args, **kwargs)
        if print_msg:
            print(msg)

    def critical(self, msg, *args, print_msg=False, **kwargs):
        kwargs['stacklevel'] = kwargs.get('stacklevel', 1) + 1
        super().critical(msg, stacklevel=1,*args, **kwargs)
        if print_msg:
            print(msg)

class CustomFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        # Define the maximum length of log level names
        self.level_lengths = {
            "INFO": 4,
            "WARNING": 7,
            "ERROR": 5,
            "DEBUG": 5,
            "CRITICAL": 8,
        }

    def format(self, record):
        # Calculate the padding required for the log level
        levelname = record.levelname
        max_length = max(self.level_lengths.values())
        padding = max_length - self.level_lengths.get(levelname, 0)

        # Add the padding to the log level
        record.levelname = " " * padding + f"[{levelname}]"

        # Shorten the file name to just the base name (without extension)
        record.filename = os.path.splitext(os.path.basename(record.filename))[0]

        # Only include location for DEBUG and ERROR logs
        if record.levelno in (logging.DEBUG, logging.ERROR):
            record.location = f"[File:{record.filename}, Function:{record.funcName}, Line:{record.lineno}]"
        else:
            record.location = f"[{record.funcName}]"

        # Call the parent class's format method
        return super().format(record)


def get_logger(name: str, lowest_level: int = logging.DEBUG) -> CustomLogger:
    """Function to set up Logger objects sharing the same configuration.
    Logs are saved in: Platform/Activity_logs

    Args:
        name(str): Name of the Logger object
        lowest_level(int): lowest logging level to be registered in the log (Default: logging.DEBUG)
    Returns:
        logging.Logger: Logger object for the corresponding file
    """
    # Create the logger
    logger: CustomLogger = CustomLogger(name, level=lowest_level)

    # Remove existing handlers for the logger to avoid duplicating logs
    if logger.hasHandlers():
        logger.handlers.clear()

    # Create filename and corresponding file handler
    timestamp = time.strftime("%Y_%m_%d-%H_%M_%S")
    day = time.strftime("%Y_%m_%d")
    save_dir = os.path.join(get_project_path(), "log_files", name, day)
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{timestamp}_{name}.log"
    file_path = os.path.join(save_dir, filename)
    handler = logging.FileHandler(file_path)

    # Set the logging parameters and details for the log entries
    handler.setLevel(lowest_level)
    formatter = CustomFormatter(
        fmt="%(levelname)s %(asctime)s %(location)s %(message)s",
        datefmt="%H:%M:%S")
    handler.setFormatter(formatter)

    # Add the filehandler to the logger object
    logger.addHandler(handler)

    return logger



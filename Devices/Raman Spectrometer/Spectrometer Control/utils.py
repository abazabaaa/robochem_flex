"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

import logging
from datetime import datetime
import os


def setup_logger(name, log_file, level=logging.DEBUG):
    """Creates and configures a logger."""
    logger = logging.getLogger(name)

    log_directory = (
        f"/mnt/Raman_Data/raman_logs/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(log_directory, exist_ok=True)
    log_file = os.path.join(log_directory, f"{log_file}.log")

    if not logger.hasHandlers():
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s // %(name)s // [%(levelname)s]: %(message)s",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.setLevel(level)

    return logger

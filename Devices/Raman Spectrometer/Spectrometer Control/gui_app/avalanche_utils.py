"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: This bad boy will run all the functions on the spectrometer(as a way to call them from the gui)

"""

import os
import sys
import tkinter as tk
from tkinter import filedialog
import matplotlib as mpl
from cycler import cycler
import numpy as np


def load_path(path_entry):
    # Open the file dialog and store the chosen directory path
    selected_directory = filedialog.askdirectory()

    # Check if a directory was selected
    if selected_directory:
        # Set the directory path in the entry widget
        path_entry.delete(0, tk.END)
        path_entry.insert(0, selected_directory)


def thread_safe_updates(root, func, *args):
    """Thread safe way to update the gui"""

    def safe_call():
        try:
            func(*args)
        except Exception as e:
            print(e)
            raise e

    # safe_call()
    # print("updating gui")
    root.after(10, safe_call)


def aesthetic_setup(ax):
    """Sets up the plot"""
    mpl.rcParams["axes.prop_cycle"] = cycler(
        color=mpl.pyplot.cm.magma(np.linspace(0, 1, 20))
    )
    ax.set_xlabel("Raman Shift (cm$^{-1}$)")
    ax.set_ylabel("Intensity (counts)")
    ax.set_xlim(200, 3500)


def clear_plot(ax):
    """clears the plot"""
    ax.clear()
    aesthetic_setup(ax)


def update_display(canvas, ax, dat, meas_method="single"):
    """updates the plot"""
    # print("updating plot")
    if meas_method == "single" or meas_method == "monitor":
        ax.clear()
    if dat.shape[1] < 2:
        return
    ax.plot(dat.iloc[:, 0], dat.iloc[:, 1], linewidth=0.5)
    ax.set_ylim(auto=True)
    canvas.draw()


def on_close(root, backend):
    """Thread safe closing routine"""
    print("Killing backend")
    backend.stop()
    print("Killing GUI")
    root.destroy()
    print("Hasta la vista, baby")
    sys.exit()


def save_path_manager(save_path):
    """manages the path for saving the spectra:
    if the path is none, just saves in the default mount without a moniker
    if the path is just a name saves it in the default mount with the given moniker
    if the path is a path that doesn't end with a / saves it in the given path with the given moniker
    if the path is a path that ends with a / saves it in the given path without a moniker

    :param save_path: str, path to save the spectra
    :return: tuple of strings, path and moniker
    """

    # Handle the case where save_path is None or an empty string
    default_path, default_moniker = None, None
    if not save_path:
        return default_path, default_moniker
    # If save_path is a file
    if os.path.dirname(save_path) and not save_path.endswith("/"):
        path = os.path.dirname(save_path)
        moniker = os.path.basename(save_path)
        return path, moniker
    # If save_path is a directory
    if save_path.endswith("/"):
        path = save_path
        return path, default_moniker
    # If save_path is just a name
    return save_path, default_moniker


def submit_parameters(
    queue,
    int_time=1000,
    rep_number=1,
    n_avs=1,
    save_each=1000,
    correct_dark=True,
    dynamic_dark=False,
    save_path=None,
    meas_method="Single",
) -> None:
    """
    This function will submit the parameters to the queue, so the backend can read them and set them
    :param queue: queue object
    :param int_time: int, integration time in ms
    :param rep_number: int, number of total spectra repetitions
    :param n_avs: number of averages per spectra repetition
    :param save_each: number of spectra repetitions to await before saving (also way to set the delay)
    :param save_path: path to save spectra. Note, timestamp of spectra will be appended to basename (also if basename is '')
    :param correct_dark: bool, whether to correct the dark current
    :param dynamic_dark: bool, whether to use dynamic dark correction
    :param meas_method: kinetic, monitor or single. Single will save each spectrum to a different file (slower but lower RAM usage, recommended for long measurements)
                        monitor will not save anything, just plots the spectras (good for alignments)
                        kinetic will keep the spectra in RAM and save them all in one csv at the end (columns with timestamps, good for fast kinetics, high ram usage)
    :return:
    """
    translation_dict = {"Single": "single", "Monitor": "monitor", "Kinetic": "kinetic"}

    parameters = {
        "integration_time": float(int_time),
        "n_averages": int(n_avs),
        "n_scans": int(rep_number),
        "delay (ms)": int(save_each),
        "correct_dark": correct_dark,
        "dynamic_correct_dark": dynamic_dark,
        "save_as": translation_dict[meas_method],
        "safety_threshold": int(10000),
    }
    path, monicker = save_path_manager(save_path)
    if path is not None:
        parameters["save_path"] = path
    if monicker is not None:
        parameters["save_moniker"] = monicker

    command = {"command": "set_parameters", "parameters": parameters}
    queue.put(command)


def start_acq(backend, queue, ax, **kwargs):
    """
    This function will start the acquisition, it will also submit the parameters to the queue
    :param backend: backend object
    :param queue: queue object
    :param args: arguments for submit_parameters
    :return:
    """
    # clear the plot:
    clear_plot(ax)
    submit_parameters(queue, **kwargs)
    backend.start_acq()

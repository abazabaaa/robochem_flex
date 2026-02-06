"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:
So basically we will use this so we can run the spectrometer without avasoft offline. So this will have some nice apps and shiz. 
Lightweight and fast.what do we need:


1) one way to set the acq parameters, 
2) continuous reading method (to check evolution of spectra)
3) acquisition with params
4) dark corrections
5) write to df mode (so you can run the kinetics nicely)


6) save on cloud!! don't save anything on raspberry it will die in about 20 min, but we have server space


"""

import os
import sys

sys.path.append(os.getcwd())

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue

from gui_app.avalanche_utils import (
    aesthetic_setup,
    on_close,
    submit_parameters,
    update_display,
    thread_safe_updates,
    load_path,
    start_acq,
)
from gui_app.AVAPP_backend import AVAPP_Backend


class MainApplication:
    """
    MainApplication encapsulates the entire application including the GUI layout and backend interactions.

    Attributes:
        master (tk.Tk): The main window of the application.
        fig (Figure): The matplotlib Figure object for plotting.
        ax (Axes): The matplotlib Axes object for plotting.
        command_queue (queue.Queue): Queue for commands to the backend.
        backend (AVAPP_Backend): The backend for handling spectrometer operations.
        frame_plot (tk.Frame): Frame to hold the plot.
        frame_controls (tk.Frame): Frame to hold the controls like buttons and entries.
    """

    def __init__(self, master):
        """
        Initialize the application, setting up the GUI components and backend.

        :param: master (tk.Tk): The main window of the application.
        """
        self.master = master
        self.master.title("AVAlanCHE Spectrometer Control")

        self.frame_plot = tk.Frame(self.master)
        self.frame_controls = tk.Frame(self.master)

        self.setup_layout()
        self.setup_plot()

        self.command_queue = queue.Queue()
        self.backend = AVAPP_Backend(
            update_gui_callback=lambda data, method: thread_safe_updates(
                self.master, update_display, self.canvas, self.ax, data, method
            ),
            parameter_queue=self.command_queue,
            root=self.master,
        )
        self.setup_controls()
        # Ensure a proper shutdown when the window is closed
        self.master.protocol(
            "WM_DELETE_WINDOW", lambda: on_close(self.master, self.backend)
        )

    def setup_layout(self):
        """Set up the grid layout for the main frames."""
        self.frame_plot.grid(row=0, column=0, sticky="nsew")
        self.frame_controls.grid(row=0, column=1, sticky="ns")

        # Configure column and row weights to control the resizing behavior
        self.master.grid_columnconfigure(0, weight=1)
        self.master.grid_columnconfigure(1, weight=0)
        self.master.grid_rowconfigure(0, weight=1)

    def setup_plot(self):
        """Set up the plot area in the GUI."""
        self.fig, self.ax = plt.subplots()
        aesthetic_setup(self.ax)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_plot)
        self.widget = self.canvas.get_tk_widget()
        self.widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def setup_controls(self):
        """Set up the control widgets in the GUI."""
        self.controls_container = tk.Frame(self.frame_controls)
        self.controls_container.pack(pady=10, padx=10, fill=tk.BOTH, expand=True)
        self.controls_container.grid_columnconfigure(0, weight=1)
        self.current_row = 0

        self.setup_path_controls()
        self.setup_buttons()
        self.setup_mode_selection()
        self.setup_param_entries()

    def setup_path_controls(self):
        """Set up controls for path selection and saving."""
        self.path_entry = tk.Entry(self.controls_container)
        self.path_button = tk.Button(
            self.controls_container,
            text="Save Path",
            command=lambda: load_path(self.path_entry),
        )
        self.path_button.grid(row=self.current_row, column=0, sticky="ew")
        self.current_row += 1
        self.path_entry.grid(row=self.current_row, column=0, sticky="ew")
        self.current_row += 1

    def setup_buttons(self):
        """Set up action buttons (Collect Dark, Collect Spectrum, Stop Acquisition)."""
        self.buttons = ["Collect Spectrum", "Stop Acquisition"]
        self.button_commands = {
            "Collect Spectrum": lambda: start_acq(
                self.backend,
                self.command_queue,
                self.ax,
                int_time=self.param_vars["Integration Time (ms)"].get(),
                rep_number=self.param_vars["Repetition Number"].get(),
                n_avs=self.param_vars["Number of Averages"].get(),
                save_each=self.param_vars["Delay (ms)"].get(),
                correct_dark=self.dark_correction.get(),
                dynamic_dark=self.advenced_dark_correction.get(),
                save_path=self.path_entry.get(),
                meas_method=self.mode_var.get(),
            ),
            "Stop Acquisition": self.backend.stop_acq,
        }
        for button in self.buttons:
            tk.Button(
                self.controls_container,
                text=button,
                command=self.button_commands[button],
            ).grid(row=self.current_row, column=0, sticky="ew")
            self.current_row += 1

    def setup_mode_selection(self):
        """Set up selector for dark correction"""

        # add a check box for dark correction
        tk.Label(self.controls_container, text="Acquisition Settings").grid(
            row=self.current_row, column=0, sticky="w"
        )
        self.current_row += 1
        self.dark_correction = tk.BooleanVar()
        self.dark_correction.set(True)
        ttk.Checkbutton(
            self.controls_container,
            text="Dark Correction",
            variable=self.dark_correction,
        ).grid(row=self.current_row, column=0, sticky="w")
        self.current_row += 1

        self.advenced_dark_correction = tk.BooleanVar()
        self.advenced_dark_correction.set(False)
        ttk.Checkbutton(
            self.controls_container,
            text="Dynamic Dark Correction",
            variable=self.advenced_dark_correction,
        ).grid(row=self.current_row, column=0, sticky="w")
        self.current_row += 1

        """Set up mode selection radio buttons."""
        self.modes = ["Single", "Monitor", "Kinetic"]
        self.mode_var = tk.StringVar(value="Single")
        for mode in self.modes:
            ttk.Radiobutton(
                self.controls_container, text=mode, value=mode, variable=self.mode_var
            ).grid(row=self.current_row, column=0, sticky="w")
            self.current_row += 1

    def setup_param_entries(self):
        """Set up parameter entries for user input."""
        self.params = {
            "Integration Time (ms)": 1000,
            "Number of Averages": 1,
            "Repetition Number": 1,
            "Delay (ms)": 1000,
        }
        self.param_vars = {}
        for param, val in self.params.items():
            tk.Label(self.controls_container, text=param).grid(
                row=self.current_row, column=0, sticky="w"
            )
            self.current_row += 1
            var = tk.StringVar()
            self.param_vars[param] = var
            entry = tk.Entry(self.controls_container, textvariable=var)
            entry.insert(0, val)
            entry.grid(row=self.current_row, column=0, sticky="ew")
            self.params[param] = entry
            self.current_row += 1


if __name__ == "__main__":
    # from unittest.mock import MagicMock, patch
    #
    # with patch(
    #     "gui_app.AVAPP_backend.AVAlanCHE", return_value=MagicMock(), autospec=True
    # ):
    root = tk.Tk()
    app = MainApplication(root)
    root.mainloop()

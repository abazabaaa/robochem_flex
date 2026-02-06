""" Run Platform

This file is automatically activated by running 'streamlit run robochem.py'.

If all parameters are filled in correctly and all necessary files are located in the right place, the platform can be
run. When this page loads the experimental_setup file and empty results files are generated.
The platform can be run by pressing the run button on the Run Platform page.
This will call the run_optimization_from_gui.py in a separate thread.
The optimization script will create and update the platform instruction files and  results files.

In between each run, the new results will be displayed. This by reading the figures named figure_hypervolyme.png and
figure_objectives.png.

Author: Elia Savino, Simone Pilon
"""

import PIL
import streamlit as st
import numpy as np
import pandas as pd
import asyncio
import os
import json
import time
import subprocess
from os.path import join
from PIL import ImageFile, Image
from backend.frontend_functions import page_header
from backend.frontend_results import main_results, save_periodically

page_header()
backend = st.session_state["platform_backend"]
# backend has 4 items that we can play with:
# backend.ready_to_roll is a boolean that decides if the platform has all the data to run
# backend.rolling is a threading.Event that is set when the platform is running (check this with backend.rolling.is_set())
# backend.frontend_queue is a queue specifically made by the backend to communicate with the frontend.

# --------------------------------------------- Run Platform -------------------------------------------------#

st.subheader("Running The Platform")

# validate the data:
# Shows error messages if something is off.
validation_return = backend.validate_data()

st.markdown(
    "Before starting the experiment, save the current settings with the 'Save Session' button below.\n"
    "Data and logs are saved automatically.\n"
)
col_button, col_button_2 = st.columns([1, 1])
with col_button:
    st.empty()
    if st.button("Save Session", help="Save campaign settings"):
        ret = backend.session_container.save_session()
        if ret == "Success":
            st.success(
                f"Session saved in '{backend.session_container['experiment_path']}'."
            )
        else:
            st.error(
                f"Could not save session in '{backend.session_container['experiment_path']}'."
            )

st.markdown(
    "\n"
    "Start and stop the campaign with the buttons below.\n\n"
    "**Note: Make sure the platform is safe to operate:**\n"
    "- **No-one is working inside / around the platform.**\n"
    "- **Platform lid is closed.**\n"
    "- **Power switches are ON.**\n\n"
    "Additional checks:\n"
    "- Solvent reservoir is full.\n"
    "- Waste is empty.\n"
    "- Stock solutions have been filled (including reaction solvents, wash solutions and cleaning agents).\n"
    "- Vial septa have been replaced.\n"
)

if not backend.ready_to_roll:
    if validation_return is not None:
        st.error(validation_return)
    else:
        st.error("Platform is not ready, we are not sure why.")
else:
    st.checkbox(
        "Skip startup cleaning",
        value=False,
        help="Initializes the platform without performing an initial cleaning cycle."
        " If the platform recently shutdown correctly, the extra cleaning is not required.",
        key="platform_start_skip_cleaning",
    )

    def experiment_shutdown():
        backend.stop()
        ret = backend.session_container.save_session()
        if ret == "Success":
            st.success(
                f"Session saved in '{backend.session_container['experiment_path']}'."
            )
        else:
            st.error(
                f"Could not save session in '{backend.session_container['experiment_path']}'."
            )

    def experiment_pause():
        backend.pause()
        ret = backend.session_container.save_session()
        if ret == "Success":
            st.success(
                f"Session saved in '{backend.session_container['experiment_path']}'."
            )
        else:
            st.error(
                f"Could not save session in '{backend.session_container['experiment_path']}'."
            )

    col_start, col_pause, col_stop, col_emergency_stop = st.columns([1, 1, 1, 1.5])
    start = col_start.button(
        "Start",
        on_click=backend.start,
        disabled=backend.rolling.is_set(),
        help="Turn on the platform and start the campaign.",
    )
    # pause = col_pause.button(
    #     "Pause",
    #     on_click=backend.pause,
    #     disabled=not backend.rolling.is_set(),
    #     help="Pause the execution of the runs after the current run is over.",
    # )
    stop = col_stop.button(
        "Shutdown",
        on_click=experiment_shutdown,
        disabled=not backend.rolling.is_set(),
        help="Trigger the shut-down procedure once the current run is over, even if the campaign is not concluded.",
    )

# main_results shows the results, errors and basically everything.
if backend.rolling.is_set():
    st.write("The platform is running")
    main_results()
    save_periodically()

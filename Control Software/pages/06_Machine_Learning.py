"""Machine Learning Settings

This file is automatically activated by running 'streamlit run robochem.py'.

Here the variable space as needed by the machine learning model (dragonfly Bayesian Optimization) is defined.
The different chemcial parameters has been decided in the Reagent Settings already. For each type of parameter, the boundaries
are defined, i.e. the min and max values it can take.

Here it is also decided how many experiments that should be done. How many initial runs and how many extra refinement
runs. It is also possible to start the optimization from previous run files. The corresponding json file has to be saved
in the folder corresponding to current campaign.

Author: Elia Savino
"""

import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import os
from os.path import join
from pathlib import Path
from backend.frontend_functions import (
    page_header,
    initialise_ml_parameters,
    display_physical_ml_parameters,
    display_ml_chem_param,
    display_ml_parameters,
    display_ml_task_all_settings,
    display_ml_task_settings,
)

# import custom functions to run these are stored elsewhere to make them easier to unit test


# --------------------------------------------- Streamlit page setup ---------------------------------------------------
page_header()
backend = st.session_state["platform_backend"]
# --------------------------------------------- Machine Learning Settings ----------------------------------------------
st.subheader("Machine Learning Settings")

st.markdown("_____")
# --------------------------------------------- Create Variable space -------------------------------------------------#

# initialise the ml_params

initialise_ml_parameters()

# get all the Ml parameters
ml_params = backend.session_container.search_by_tag("ML_parameter")
chemical_ml_parameters = [
    backend.session_container[param]
    for param in ml_params
    if backend.session_container[param].phy_chem == "Chemical"
]
physical_ml_parameters = [
    backend.session_container[param]
    for param in ml_params
    if backend.session_container[param].phy_chem == "Physical"
]

st.markdown("## Set the chemical space for the ML algorithm")
if len(chemical_ml_parameters) > 0:
    st.markdown("### Chemicals")
    for chemical_ml_param in chemical_ml_parameters:
        display_ml_chem_param(chemical_ml_param)

    if backend.session_container["experiment_type"] in [
        "MultiTaskScope",
        "MultiTaskScope_HITL",
    ]:
        st.markdown("### Task Settings")
        display_ml_task_all_settings(chemical_ml_parameters)
    elif backend.session_container["experiment_type"] == "ScopeAcceleratorTask":
        st.markdown("### Task Settings")
        display_ml_task_settings(chemical_ml_parameters)

if len(physical_ml_parameters) > 0:
    st.markdown("### Physical Parameters")
    for physical_ml_param in physical_ml_parameters:
        display_physical_ml_parameters(physical_ml_param)


## --------------------------------------------- Decide Objectives -------------------------------------------------#
objectives = (
    "yield",
    "conversion",
    "cost",
    "throughput",
    "selectivity",
    "residence_time",
    "light_power_efficiency",
    "light_power",
    "cost_per_unit_product",
    "mass_balance",
    "Elia-metric",
    "integral_product",
    "integral_sideproduct",
    "enantiomeric excess",
    "diastereomeric ratio",
    "total integral chiral",
)
# objective_index = objectives.index(st.session_state["objective"])
obj = st.multiselect(
    "Select the objectives to optimize on",
    objectives,
    default=(
        backend.session_container["objectives"]
        if "objectives" in backend.session_container
        else objectives[0]
    ),
)
backend.session_container["objectives"] = obj

# --------------------------------------------- Number of runs -------------------------------------------------#
# st.markdown("---")
# load previous data file:
# st.markdown("Load previous data file")
# previous_run_file = st.file_uploader(
#     "Upload your previous data", type=["json"], key="session_file"
# )
# if st.session_state["session_file"] is not None:
#     ret_value = backend.load_previous_experiments(previous_run_file)
#     if ret_value == "Success":
#         st.success("Session Loaded Successfully!")
#     elif ret_value == "Warning":
#         st.warning = "Experimental space does not match, previous run not loaded"
#     elif ret_value == "Error":
#         st.error("Something went terribly wrong, we're all about to die, RUN!")


# display the input parameter for the machine learning model

st.markdown("---")
st.write("## ML Parameters")
display_ml_parameters()

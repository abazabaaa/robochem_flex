""" User Settings

This file is automatically activated by running 'streamlit run robochem.py'.

Here the user can fill in the name of the user which will create a new folder (if it doesn't already exist)
and the name of the experiments that will be run. All corresponding files will be saved at this location with
the name of the experiments.

Author: Elia Savino
"""

import streamlit as st
import os
from backend.frontend_functions import page_header


# --------------------------------------------- Streamlit page setup ---------------------------------------------------
backend = st.session_state["platform_backend"]
page_header()

# ----------------------------------------------------- User -----------------------------------------------------------
st.subheader("User name")
st.write("")

# Create list with available users, move guest folder to front (default option)
available_users = [
    name
    for name in os.listdir(backend.robochem_path)
    if os.path.isdir(os.path.join(backend.robochem_path, name))
]

available_users.insert(0, "Guest")
available_users = list(set(available_users))

col1, col2, col3 = st.columns([10, 2, 10])

# Choose the user
user_name = col1.selectbox(
    "Current user",
    available_users,
    key="user_name",
    index=available_users.index(backend.session_container.get("user_name", "Guest")),
)
backend.user_path = os.path.join(backend.robochem_path, st.session_state["user_name"])
backend.session_container.update_session("user_name", user_name)

# Add new user if not in list of available users
new_folder = col3.text_input("If your name is not in list, add your name here")
if new_folder != "":
    if not os.path.exists(os.path.join(backend.robochem_path, new_folder)):
        os.makedirs(os.path.join(backend.robochem_path, new_folder))
col3.button("Update user-list")
col2.write("")

st.markdown("_____")

# ------------------------------------------------ FILENAME ------------------------------------------------------------

st.subheader("Experiment")
col21, col22 = st.columns([28, 28])
experiment_name = col21.text_input(
    label=f"Insert the name for the files generated this run:",
    value=backend.session_container.get("experiment_name", ""),
    key="experiment_name",
)
backend.session_container.update_session("experiment_name", experiment_name)

if st.session_state["experiment_name"] != "":
    backend.experiment_path = os.path.join(
        backend.user_path,
        st.session_state["experiment_name"],
    )
    if not os.path.exists(backend.experiment_path):
        os.makedirs(backend.experiment_path)
    backend.session_container.update_session("experiment_path", backend.experiment_path)


st.markdown("_____")
# ------------------------------------------------- Platform Selection -------------------------------------------------

st.subheader("Platform and Experiment Selection")
st.write("")
st.write("")
col31, col32, col33 = st.columns([1, 1, 1])
# Choose the platform
platform_name = col31.selectbox(
    "Select the platform you are using",
    backend.available_platforms,
    key="platform_name",
    index=(
        backend.available_platforms.index(
            backend.session_container.get("platform_name", None)
        )
        if backend.session_container.get("platform_name", None) is not None
        else 0
    ),
)
backend.session_container.update_session("platform_name", platform_name)
# find the available experiment:
if platform_name != "":
    available_experiments = backend.platform_available_experiments(platform_name)
else:
    available_experiments = []

platform_experiment = col32.selectbox(
    "Select the experiment for the platform",
    available_experiments,
    key="platform_experiment",
    index=(
        available_experiments.index(
            backend.session_container.get("platform_experiment", None)
        )
        if backend.session_container.get("platform_experiment", None) is not None
        else 0
    ),
)
backend.session_container.update_session("platform_experiment", platform_experiment)

available_ml = list(backend.ML_classes.keys())

# Choose the experiment
experiment_name = col33.selectbox(
    "Select the experiment machine learning method",
    available_ml,
    key="experiment_type",
    index=(
        available_ml.index(backend.session_container.get("experiment_type", None))
        if backend.session_container.get("experiment_type", None) is not None
        else 0
    ),
)
backend.session_container.update_session("experiment_type", experiment_name)

# save the choice and initialise the experiment class.

if "experiment_class" in backend.session_container.keys():
    if isinstance(
        backend.session_container["experiment_class"],
        backend.ML_classes[experiment_name],
    ):
        st.success(
            f"There is a ML module loaded, probably you are continuing a previous session."
            f" The experiment class is {experiment_name}. This will not be overridden."
        )
        backend.ml_experiment_class = backend.session_container["experiment_class"]
    else:
        st.warning(
            f"There is a ML module loaded of type {type(backend.session_container['experiment_class'])}."
            f"this doesn't match the selection in the form: {experiment_name}. Do you want to override it and start a new one?"
        )
        if st.button("Yes"):
            backend.ml_experiment_class = backend.ML_classes[experiment_name]()
            backend.session_container.update_session(
                "experiment_class", backend.ml_experiment_class
            )
elif not hasattr(backend, "ml_experiment_class") or backend.ml_experiment_class is None:
    st.success("There is no ML module loaded, initialising a new one.")
    backend.ml_experiment_class = backend.ML_classes[experiment_name]()
    backend.session_container.update_session(
        "experiment_class", backend.ml_experiment_class
    )
else:
    st.error(
        "Something went wrong with the experiment class, please restart the platform."
    )

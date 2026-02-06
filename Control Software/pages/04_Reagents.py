""" Reagent Settings

This file is automatically activated by running 'streamlit run robochem.py'.

The different reagents and their corresponding type is filled out here. The types are used to set up
the variable space of the machine learning model (Bayesian Optimization) in the proceeding page and the
chemical information is stored in a spreadsheet.

Author: Elia Savino
"""

import pandas as pd
import streamlit as st
from backend.frontend_functions import (
    page_header,
    display_chemical_inputs,
    display_sample_and_stock_solution_ui,
)
from backend.platform_backend import PlatformBackend

# --------------------------------------------- Streamlit page setup ---------------------------------------------------
backend: PlatformBackend = st.session_state["platform_backend"]
page_header()

# --------------------------------------------- Reagents and Solvents --------------------------------------------------
st.subheader("Reagents and Solvents")

st.markdown(
    "Specify all chemical species required for the optimization such as reagents and solvents."
)
text_columns = st.columns([0.75, 1])
text_columns[0].markdown(
    "##### Identifiers\n"
    "Unique identifiers are added to match the name of each chemical to a known chemical species."
    "The identifier can point to your labjournal (Internal_ID) or to standardized identifiers, such as CAS numbers."
)
text_columns[1].markdown(
    "##### Purpose\n"
    "Each reagent is assigned a role in the reaction.\n\n"
    "- Reagents with identical role will be considered alternatives to each other and"
    " exchanged by the ML algorithm during the optimization rounds.\n"
    "- There must be at least one limiting reagent.\n"
)
st.write("\n\n\n")

columns = st.columns([2, 2, 1])
# Number of reagents
number_reagents = columns[0].number_input(
    label="Number of chemical species used in the experiment (reagents and solvents):",
    min_value=1,
    value=backend.session_container.get("number_of_reagents", 1),
    step=1,
    help="Make sure to add all chemicals here",
    key="number_of_reagents",
)
backend.session_container.update_session("number_of_reagents", number_reagents)

# Prices
price = columns[2].selectbox(
    "Add Prices to Reagents?",
    ["Yes", "No"],
    key="price",
    index=["Yes", "No"].index(backend.session_container.get("price", "No")),
)
st.session_state["platform_backend"].session_container.update_session("price", price)

existing_chemical_keys = backend.session_container.search_by_tag("chemical_parameter")

# Display inputs for existing chemicals first and check for modifications
for key in existing_chemical_keys:
    chemical = backend.session_container.get(key)
    display_chemical_inputs(
        chemical, existing_chemical_keys.index(key)
    )  # Get modified chemical
    if chemical.name == key and chemical:
        backend.session_container.update_session(key, chemical)
    elif chemical:
        backend.session_container.update_session(chemical.name, chemical)
        del st.session_state["platform_backend"].session_container[key]
    elif not chemical.name:
        del st.session_state["platform_backend"].session_container[key]


# Calculate the number of additional chemicals needed
num_existing_chemicals = len(existing_chemical_keys)
additional_chemicals_needed = number_reagents - num_existing_chemicals

for i in range(additional_chemicals_needed):
    chemical = st.session_state["platform_backend"].chemical_parameter()
    display_chemical_inputs(chemical, i + num_existing_chemicals)
    if chemical is not None:
        st.session_state["platform_backend"].session_container.update_session(
            chemical.name, chemical
        )
st.markdown("----")

# ------------------------------------------------- Reagent Concentration and Volumes From Platform Template------------
display_sample_and_stock_solution_ui()

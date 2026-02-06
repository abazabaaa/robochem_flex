""" Experiment Settings

This file is automatically activated by running 'streamlit run robochem.py
All extra parameters that are needed to run the platform are filled in here.

Author: Elia Savino
"""

import streamlit as st
import copy

from backend.platform_backend import PlatformBackend
from backend.frontend_functions import (
    page_header,
    render_analytical_parameter,
    create_input_widget,
    traverse_methods,
)

# --------------------------------------------- Streamlit page setup ---------------------------------------------------
backend: PlatformBackend = st.session_state["platform_backend"]
page_header()

# ------------------------------------------------- Analytics ------------------------------------------------
st.subheader("Analysis")
st.write("")
st.markdown(
    "Please specify below the analytical parameters to use during the optimization."
)

analytics1, analytics2 = st.columns([1, 1])
if "HITL" not in backend.session_container.get("experiment_type", ""):
    available_analytics_names = backend.get_analytics()
    if available_analytics_names != []:
        analysis_techique = analytics1.selectbox(
            "Select the analytical technique",
            available_analytics_names,
            key="analysis_type",
            index=(
                list(available_analytics_names).index(
                    backend.session_container.get("analysis_type", None)
                )
                if backend.session_container.get("analysis_type", None)
                else 0
            ),
        )

        backend.session_container.update_session("analysis_type", analysis_techique)

    analytic_parameters = backend.get_analytic_parameters()
    if analytic_parameters == None:
        st.error(
            "this technique is not allowed for the current experiment. something went wrong"
        )
    elif len(analytic_parameters) == 0:
        st.write(
            "You are using a human in the loop technique, therefore no further selection is needed from you!"
        )
    else:
        required_parameters, optional_parameters, maths_available = analytic_parameters

        maths = analytics2.selectbox(
            "Select the mathematical technique for the analysis",
            maths_available,
            key="analysis_maths",
            index=(
                list(maths_available).index(
                    backend.session_container.get("analysis_maths", None)
                )
                if backend.session_container.get("analysis_maths", None)
                and backend.session_container.get("analysis_maths", None)
                in maths_available
                else 0
            ),
        )
        backend.session_container.update_session("analysis_maths", maths)
        # first handle the required parameters

        # first remove all parameters that are in the session container
        # but not in the required parameters or optional parameters
        analytical_parameters_in_sesh = backend.session_container.search_by_tag(
            "analytical_parameter"
        )

        # Get the optional parameters that match the analytical_maths tag
        optional_params = [
            param for param in optional_parameters if param.tag in (maths, "all")
        ]
        required_parameters = [
            param for param in required_parameters if param.tag in (maths, "all")
        ]

        # Merge required and optional parameters
        merged_params = copy.deepcopy(required_parameters + optional_params)
        merged_param_names = {param.name for param in merged_params}

        # Remove analytical parameters not in the merged list
        for analytical_param in analytical_parameters_in_sesh:
            if analytical_param not in merged_param_names:
                del backend.session_container[analytical_param]

        # Process merged parameters
        for parameter in merged_params:
            if parameter.name not in backend.session_container:
                parameter_data = backend.analytical_parameter(
                    name=parameter.name,
                    min_value=parameter.min_value,
                    max_value=parameter.max_value,
                    unit=parameter.units,
                    omni_tag=parameter.tag,
                    value=parameter.value,
                    allowed_values=parameter.discrete_values,
                )
                parameter_data.value = parameter.value
            else:
                parameter_data = backend.session_container[parameter.name]

            render_analytical_parameter(parameter.name, parameter_data)
            backend.session_container.update_session(parameter.name, parameter_data)


else:
    backend.session_container["analysis_type"] = "Human"
    st.success(
        "You are using a human in the loop technique, therefore no further selection is needed from you!"
    )

# analytics1, analytics2 = st.columns([1, 1])
# analysis_technique = analytics1.selectbox(
#     "Select the analytic technique",
#     (
#         backend.get_analytics(
#             backend.session_container.get("platform_name", None)
#         ).keys()
#         if backend.session_container.get("platform_name", None)
#         else []
#     ),
#     key="analysis_technique",
#     index=(
#         list(
#             backend.get_analytics(
#                 backend.session_container.get("platform_name", None)
#             ).keys()
#         ).index(backend.session_container.get("analysis_technique", None))
#         if backend.session_container.get("analysis_technique", None)
#         else 0
#     ),
# )
#
# backend.session_container.update_session("analysis_technique", analysis_technique)
# available_classes = (
#     backend.get_analytics(backend.session_container.get("experiment_type", None))
#     .get(analysis_technique, [])
#     .get("classes", [])
# )
# analysis_method = analytics2.selectbox(
#     "Select the analytic method",
#     available_classes if available_classes else [],
#     key="analysis_method",
#     index=(
#         available_classes.index(backend.session_container.get("analysis_method", None))
#         if backend.session_container.get("analysis_method", None)
#         else 0
#     ),
# )
# backend.session_container.update_session("analysis_method", analysis_method)
# # ------------------------------------------------- Analytical Parameters ------------------------------------------------
# st.subheader("Set up the analytical parameters")
# st.write("")
# available_parameters = (
#     backend.get_analytics(backend.session_container.get("platform_name", None))
#     .get(analysis_technique, [])
#     .get("parameters", [])
# )
#
# if backend.session_container.get("analysis_method", None) == "IntegraLazy":
#     available_parameters["integration_bounds"] = {
#         "min_value": 200.0,
#         "max_value": 3500.0,
#         "unit": "cm$^{-1}$",
#         "default_low": 200.0,
#         "default_high": 3500.0,
#     }
# elif "integration_bounds" in available_parameters.keys():
#     del available_parameters["integration_bounds"]
#
# for parameter_key, parameter_value in available_parameters.items():
#     if parameter_key not in backend.session_container.keys():
#         parameter_data = backend.analytical_parameter(
#             name=parameter_key,
#             min_value=parameter_value["min_value"],
#             max_value=parameter_value["max_value"],
#             unit=parameter_value["unit"],
#         )
#     else:
#         parameter_data = backend.session_container[parameter_key]
#
#     render_analytical_parameter(parameter_key, parameter_value, parameter_data)
#
#     backend.session_container.update_session(parameter_key, parameter_data)

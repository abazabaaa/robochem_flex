"""
File: batch_nmr_analysis.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: Get calibration curve data for in-line NMR analysis.
"""

import os.path
import pandas as pd

from omniplatypus.procedures.analytics.analytics_parameters import AnalyticalParameter
from omniplatypus.procedures.analytics.nmr_analysis import NMRAnalysis
from omniplatypus.procedures.unit_tasks.sampling.liquid_handler_sampling import (
    RecipeComponent,
)

if __name__ == "__main__":
    analysis = NMRAnalysis(None)

    folder_path = input("Files location (parent folder):")

    files = next(os.walk(folder_path))[1]

    print("Found:")
    for file in files:
        print(file)
    if not input("proceed (y/N)?").lower() in ("y", "yes"):
        exit(0)

    target_peak = float(input("What is the chemical shift of the target? "))

    data_folder = os.path.join(
        "\\\\fnwi-s0.science.uva.nl",
        "hims-nrg-robochem",
        "nmr_data",
        "Perry",
        "calibration",
    )
    integrals = None

    for sample_path in files:
        parameters = [
            AnalyticalParameter(name="sample_name", value=sample_path),
            AnalyticalParameter(
                name="data_folder",
                value=folder_path,
            ),
            AnalyticalParameter(name="target_peak", value=target_peak),
            AnalyticalParameter(name="target_peak_deviation", value=3),
            AnalyticalParameter(name="peak_resolution", value=1),
            AnalyticalParameter(name="target_peak_calibration_coeff_1", value=1917.58),
            AnalyticalParameter(
                name="yield_calculation_chemical",
                value="SM",
            ),
        ]
        result = analysis.analyse(
            {p.name: p for p in parameters},
            recipe=[RecipeComponent("SM", 100.0)],
            process_only=True,
        )
        result["sample"] = sample_path
        if integrals is None:
            integrals = pd.DataFrame(columns=list(result.keys()))
        integrals.loc[len(integrals.index)] = result

    print(integrals.to_string())
    outfile = os.path.join(folder_path, "batch.csv")
    if input(f"Save to '{outfile}'?").lower() in ("y", "yes", "yeah"):
        integrals.to_csv(outfile)

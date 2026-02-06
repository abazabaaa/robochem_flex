"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: Functions for the buchwald ligand featurization and defeaturization.

"""

import pandas as pd
import numpy as np
import torch
from typing import List
from torch import Tensor
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "featurizations", "buchwald_ligand_umap.csv")


def buchwald_translation_discrete(x: str, ligands: List[str]) -> np.ndarray:
    """
    Encodes the discrete ligand `x` as a scaled embedding (each dimension between 0 and 1).

    The embedding information is loaded from the CSV file
    'featurizations/buchwald_ligands_umap.csv' which must contain at least
    the columns 'name', 'Dim1', and 'Dim2'. Only the rows whose 'name' is in
    the provided `ligands` list will be used.

    Parameters:
        x (str): The ligand name to be converted.
        ligands (List[str]): List of ligand names (must be a subset of the CSV 'name' column).

    Returns:
        np.ndarray: A 1D torch tensor containing the scaled embedding for `x`, rounded to 5 decimals.

    Raises:
        ValueError: If any ligand in `ligands` is not found in the CSV or if `x` is not in the list.
    """
    # Load CSV file
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        raise RuntimeError(f"Error loading CSV file: {e}")

    # Verify that all provided ligands exist in the CSV 'name' column
    csv_names = set(df["name"])
    if not set(ligands).issubset(csv_names):
        missing = set(ligands) - csv_names
        raise ValueError(f"The following ligands are not found in the CSV: {missing}")

    # Also ensure x is one of the provided ligands
    if x not in ligands:
        raise ValueError(f"Ligand '{x}' is not in the provided ligand list.")

    # Subset the dataframe to include only the specified ligands
    subset_df = df[df["name"].isin(ligands)]

    # Extract the embedding matrix (columns "Dim1" and "Dim2")
    dims = subset_df[["Dim1", "Dim2"]].to_numpy(dtype=np.float64)

    # Compute column-wise min and max for the subset
    min_vals = dims.min(axis=0)
    max_vals = dims.max(axis=0)

    # Get the embedding for ligand x from the subset
    ligand_row = subset_df[subset_df["name"] == x]
    if ligand_row.empty:
        raise ValueError(f"Ligand '{x}' was not found in the filtered data.")

    embedding = ligand_row[["Dim1", "Dim2"]].values.flatten()

    # Min-max scale the embedding for x so that within this subset, it lies in [0, 1]
    scaled_embedding = (embedding - min_vals) / (max_vals - min_vals)
    # Avoid division by zero
    scaled_embedding = np.clip(scaled_embedding, 0, 1)
    scaled_embedding = np.round(scaled_embedding, 5)
    scaled_embedding = torch.tensor(scaled_embedding, dtype=torch.float64)

    return scaled_embedding


def buchwald_backtranslation_discrete(x: Tensor, ligands: List[str]) -> str:
    """
    Backtranslates a scaled embedding (with each dimension in [0, 1]) back to the original ligand name.

    The function loads embeddings from the CSV 'featurizations/buchwald_ligands_umap.csv',
    subsets them by `ligands`, and uses the same min–max scaling parameters of the subset to invert
    the scaling and then find the ligand name whose original embedding is closest (in Euclidean distance)
    to the computed value.

    Parameters:
        x (Tensor): A tensor with 2 elements (scaled [0, 1] embedding).
        ligands (List[str]): List of ligand names (must be a subset of the CSV 'name' column).

    Returns:
        str: The ligand name corresponding to the closest original embedding.

    Raises:
        ValueError: If any ligand in `ligands` is not found in the CSV or if `x` does not have exactly 2 elements.
    """
    # Convert x to a tensor if needed and validate shape
    if not isinstance(x, torch.Tensor):
        raise ValueError("Input x must be a torch.Tensor.")
    if x.numel() != 2:
        raise ValueError(
            "Input x must be a tensor with exactly 2 elements corresponding to the two dimensions."
        )

    # Load CSV file
    try:
        df = pd.read_csv(CSV_PATH)
    except Exception as e:
        raise RuntimeError(f"Error loading CSV file: {e}")

    # Verify that all provided ligands exist in the CSV 'name' column
    csv_names = set(df["name"])
    if not set(ligands).issubset(csv_names):
        missing = set(ligands) - csv_names
        raise ValueError(f"The following ligands are not found in the CSV: {missing}")

    # Subset the dataframe to include only the specified ligands
    subset_df = df[df["name"].isin(ligands)]

    # Extract the embedding matrix (columns "Dim1" and "Dim2")
    dims = subset_df[["Dim1", "Dim2"]].to_numpy(dtype=np.float64)

    # Compute column-wise min and max for the subset
    min_vals = dims.min(axis=0)
    max_vals = dims.max(axis=0)

    # Inverse min–max scaling: rescale the input value back to the original embedding space
    scaled_input = x.cpu().detach().numpy().astype(np.float64)
    original_embedding = scaled_input * (max_vals - min_vals + 1e-8) + min_vals

    # Calculate Euclidean distances from the computed original_embedding to each ligand's embedding in the subset
    distances = np.linalg.norm(dims - original_embedding, axis=1)

    # Identify the index of the ligand with the minimum distance
    closest_index = distances.argmin()

    # Return the corresponding ligand name
    closest_ligand = subset_df.iloc[closest_index]["name"]
    return closest_ligand

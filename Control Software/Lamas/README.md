# Lamas
<img src="Docs/images/NMRLama_logo.png" alt="NMRLama Logo" width="150">

> ***"We have seen the lama, and he is us"***  
> — *Adapted from Walt Kelly (Pogo: Earth Day)*



## Overview

**Lamas** (Library for Automated Multivariate Analysis of Spectra) is a Python package for the **handling, preprocessing, fitting, and quantitative analysis of spectroscopic data**.  
It was originally developed as part of the **Robochem automated optimization platform** for autonomous reaction monitoring and data-driven chemistry.

Lamas provides a **modular, extensible pipeline** for spectral data workflows — from raw preprocessing to machine-learning-ready feature generation.

---

## Core Modules

### DISCLAIMER
some parts of this codebase are still under development, Alpaca and Vicuna work, the rest is prone to bugs. So be careful and if you want fix and contribute!


### Alpaca — *Preprocessing*
Handles spectral cleaning and preparation, including:
- Baseline correction
- Denoising and smoothing (Savitzky–Golay, FFT, etc.)
- Range trimming and normalization
- Format conversion and spectral alignment

### Vicuna — *Peak Fitting*
Performs iterative **Voigt/Lorentzian/Gaussian** profile fitting with adaptive convergence control.  
Outputs include:
- Fitted peaks with center, width, amplitude
- Integrated peak areas
- Quantitative feature extraction for calibration

### Guanaco — *Model Building*
Maps **spectral features to chemical concentrations** using multivariate regression or calibration models.  
Supports:
- Classical least squares calibration (CLS)
- PLS/PCA-based dimensionality reduction
- Cross-validation and error analysis

### Lama — *Decomposition & Prediction*
Performs full-spectrum **multivariate decomposition**, **prediction**, and **model validation**.  
Integrates seamlessly with the machine-learning models used in Robochem for:
- Concentration prediction
- Feature space embedding
- Online reaction monitoring

---

## Installation

Clone the repository and install **Lamas** in editable mode for development or use within the Robochem ecosystem:

```bash
git clone https://github.com/NoelResearchGroup/Lamas.git
cd Lamas
python setup.py sdist bdist_wheel
pip install -e .
```

> **Tip:** Always install Lamas inside the same Conda environment as Robochem-Flex for full interoperability.

---

## Documentation

Detailed documentation, API references, and usage tutorials are available in the `Docs/` directory and the Robochem documentation site.

- `Docs/usage_examples/` – step-by-step Jupyter notebooks
- `Docs/api_reference/` – full API docs for each module
- `Docs/images/` – figures and diagrams for publication or teaching

---

## Integration with Robochem

Lamas serves as the **analytical backend** of the Robochem platform, allowing:
- Real-time **spectral acquisition and interpretation**
- Automated **peak quantification** during optimization campaigns
- Easy coupling to ML-driven experiment control loops

It is compatible with data from NMR, Raman, UV-Vis, and IR spectrometers, assuming appropriate data adapters.

---

## Authors & Contributors

Developed by the **Noël Research Group (NRG)**  
*University of Amsterdam – Flow Chemistry & Machine Learning Automation*

**Contributors:** Elia Savino.

---

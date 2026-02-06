# RoboChem-Flex Session Folder Documentation

Each **session folder** represents a single saved experiment state. It contains all information required to reproduce an optimisation run — including the machine-learning setup, chemical inventory, and vial mapping — in a **human-readable** and **GUI-recoverable** format.

```
Session_Folder/
│
├── session.json
├── StockSolutionDF.csv
├── VialDF.csv
└── (optional) results/, logs/, figures/ ...
```

---

## `session.json`

This file is the **central configuration artifact** for a RoboChem experimental session. It contains every element required to reproduce a campaign, including the hardware configuration, reagents, analytical parameters, and machine-learning settings. It is **automatically generated** by the GUI and mirrors the structure of the internal runtime session state dictionary.

While the JSON is **human-readable** for verification, it is **not intended for manual editing**. All changes should be performed through the GUI interface to maintain structural consistency.

---

## 1. Project Metadata

Defines the high-level context in which the experiment operates.

| Key | Example | Description |
|-----|----------|-------------|
| `platform_name` | `"Perry"` | Identifier for the RoboChem hardware setup. |
| `experiment_type` | `"BO_optimisation_HITL"` | Specifies the experiment logic, e.g. Bayesian Optimization (BO) or Human-in-the-Loop (HITL). |
| `experiment_name` | `"CS2"` | Session-specific name for the campaign. |
| `platform_experiment` | `"PhotochemicalReaction"` | Backend module that determines which process model is used. |
| `experiment_path` | `"./runs/CS2/"` | Local directory where data and logs are stored. |

---

## 2. Chemical Parameters

Each reagent, solvent, or additive is represented as a `Chemical` object, defining **identity and role** in the reaction system.

Example:
```json
"Ru(bpy)3(PF6)2": {
  "class": "Chemical",
  "name": "Ru(bpy)3(PF6)2",
  "identifier": "60804-74-2",
  "identifier_type": "CAS",
  "_purpose": "Catalyst",
  "value": null
}
```

Every `Chemical` object includes:
- `name`: Human-readable name or abbreviation.
- `identifier`: CAS number or internal code.
- `_purpose`: Functional role (`Limiting Reagent`, `Catalyst`, `Solvent`, etc.).
- `value`: Optional numerical concentration when fixed.

---

## 3. Physical Parameters

Defined as `PhysicalParameter` objects describing the **physical and operational conditions** of the experiment. Examples include:
- `residence_time`
- `light_intensity`
- `flowrate_sampling`
- `slug_volume`
- `bubble_volume`

Each entry contains:

| Field | Description |
|--------|-------------|
| `min_value`, `max_value` | Numerical range for the variable. |
| `unit` | Measurement unit (e.g. `"S"`, `"mL/min"`, `"%"`, `"uL"`). |
| `style` | `"constant"` (fixed) or `"variable"` (optimizable). |
| `value` | Actual numerical value if fixed. |

Physical parameters constrain how the platform hardware operates. Available parameters for each experiment can be found in the omniplatipus documentation.

---

## 4. Analytical Parameters

Parameters describing **data acquisition and processing** are stored as `AnalyParameter` objects. These specify instrument settings, integration methods, and calibration constants. To know more about these look at Omniplatipus.analysis. These can't be optimized at the moment.

Examples:

| Key | Description |
|---|---|
| `analysis_type` | Specifies the analysis instrument (`NMR`, `UPLC`, `Raman`, etc.). |
| `protocol` | Measurement program or pulse sequence. |
| `target_rt` / `target_peak` | Expected retention time or chemical shift for the target compound. |
| `baseline_model` | Baseline subtraction algorithm (`flatfit`, `asls`, `arpls`). |
| `peak_model` | Fitting model (`Bemg`, `BiGaussian`, `FraserSuzuki`, etc.). |
| `target_peak_calibration_coeff_*` | Conversion coefficients from signal intensity to concentration. |


Each `AnalyParameter` object includes:
- `unit` (e.g. `"ppm"`, `"min"`, `"mM/AU"`)
- `style` (constant or variable)
- optional `allowed_values` (enumerated instrument settings)

These are implemented and validated via **`robrains.parameter_backends.AnalyticalParameter`**.

---

## 5. Machine-Learning Parameters (`*_ml`)

Entries ending with `_ml` represent the **optimizable variables** controlled by the machine-learning engine (e.g., Bayesian Optimization). These are generated dynamically by **`robrains.parameter_backends.MLParameter`**. These get generated from chemical and physical parameters and server the purpose of holding featurization.

Example:
```json
"Catalyst_ml": {
  "class": "ML_parameter",
  "name": "Catalyst",
  "unit": "eq",
  "phy_chem": "Chemical",
  "discrete": ["Ir(ppy)3", "Ru(bpy)3(PF6)2"],
  "min_value": 0.0025,
  "max_value": 0.02,
  "discrete_embed_dim": 2
}
```

Each ML parameter may be:
- **Continuous**, like residence time or light intensity.
- **Discrete**, like catalyst or solvent identity.

Core fields:

| Field | Description |
|--------|-------------|
| `phy_chem` | Indicates whether the parameter is chemical or physical. |
| `discrete` | List of discrete choices (if categorical). |
| `min_value`, `max_value` | Numerical range for optimization. |
| `discrete_embed_dim` | Size of embedding vector used for categorical encoding. |
| `fidelity_feature`, `task_feature` | Flags for multi-fidelity or multi-task experiments. |

Categorical variables are internally embedded using PyTorch `Embedding` layers, enabling hybrid discrete-continuous optimization.

---

## 6. Experiment Class

Defines the **optimization logic** governing the run.

Example:
```json
"experiment_class": {
  "class": "SingleBayesianOptiHITL",
  "parameters": {
    "Model": "SingleTaskGP",
    "Acquisition Function": "UCB",
    "Initialisation Method": "LHS",
    "Number of total points": 40,
    "Number of Experiments per batch": 5
  },
  "ML_parameters": [
    "Limiting Reagent_ml",
    "Catalyst_ml",
    "light_intensity_ml",
    "residence_time_ml"
  ],
  "targets": ["yield"]
  "results_df": "@DataFrame]Results.csv"
}
```

Defines:
- the optimization strategy (`SingleBayesianOpti`, `SingleBayesianOptiHITL`, etc.),
- model type (e.g. `SingleTaskGP`),
- acquisition function (`UCB`, `EI`, etc.),
- number of planned and batch experiments,
- optimization targets (e.g., yield, selectivity).\
- results df pointer if present

---

## 7. Linked DataFrames

The JSON links two external CSVs, defining all **chemical and vial information**.

```json
"StockDF": "@DataFrame]StockSolutionDF.csv",
"VialDF": "@DataFrame]VialDF.csv"
```

- **`StockSolutionDF.csv`** — chemical stock inventory (name, solvent, concentration).  
- **`VialDF.csv`** — mapping of vials, holders, and handlers in the robotic system.

These are loaded automatically when the session is restored.

---

## 8. Provenance and Interrelation

All parameter classes — `Chemical`, `PhysicalParameter`, `AnalyParameter`, and `ML_parameter` — are instantiated by the **`robrains.parameter_backends`** package, ensuring:
- unified parameter validation,
- consistent serialization across experiments,
- full compatibility with the Bayesian optimization backend.

For extended class-level details, refer to the RoBrains README or module documentation.

---

## `StockSolutionDF.csv`

This file defines all stock solutions available in the current session.
Each row represents a physically prepared container containing one or more solutes at defined concentrations in a solvent.
These stock solutions act as sources for reagent delivery, both in automated and manual workflows.

Typical columns include:

| Stock Name |  solvent | chemical A mM | chemical B mM | chemical C mM | etc |
|------------|--------------|---------------|----------|---------------|-----|

Each stock is referenced in the JSON under `"StockDF" → "chemicals_list"` and can be selected in the GUI when assigning reagents to vials.

| Column | Description |
| --- | --- |
|Stock Name|Unique identifier automatically generated as Stock_0, Stock_1, etc. Used internally by the robotic handler and referenced in VialDF.csv and the session JSON.|
|Solvent|The solvent in which all solutes are dissolved (e.g. MeCN, DMSO, THF). Must match a Chemical object in the session JSON.|
|Chemical A/B/C (mM)|Columns corresponding to each solute in the stock. The header name matches the chemical name from the session JSON; the value is the molar concentration in millimolar units.|

Each stock solution entry is automatically referenced in the JSON under:

```json
"StockDF": {
  "class": "StockSolutionDF",
  "df": "@DataFrame]StockSolutionDF.csv",
  "chemicals_list": ["SM", "THF", "TBADT"],
  "solvent_list": ["MeCN"]
}
```

---

## `VialDF.csv`

This file specifies the physical layout, identity, and mapping of all vials accessible to the RoboChem platform.
It connects stock solutions to their physical positions (holder, rack, coordinates) within the robotic sampling and collection systems.

Typical columns include:

|VialID|VialName|StockID|Volume|Sampler|Holder|Position|Type|Counter|Viable|
|----------|-----------|---------|---------------|---------------|---------|---|---|---|---| 

Additional nested data (e.g. `handler_info`, `available_handlers`, `available_holders`) describe how robotic samplers and collectors map to physical racks.

| Column      | Description                                                                                                                                          |
|-------------|------------------------------------------------------------------------------------------------------------------------------------------------------|
| VialID      | Numerical or string identifier (e.g. 1, A1, Collector_D3). Used internally to track vial order.                                                      |
| VialName    | Human-readable label (e.g. “Sample 1,” “Standard 3”). Generated automatically or set via GUI.                                                        |
| StockID     | Reference to a stock name from StockSolutionDF.csv (e.g. Stock_2). Indicates which stock solution is dispensed into this vial.                       |
| Volume (µL) | Total liquid volume contained or allocated for filling. Used by the liquid handler to determine pipetting parameters.                                |
| Sampler     | Identifier for the robotic module that interacts with this vial (Sampler_cnc, Collector_cnc, etc.).                                                  |
| Holder      | Refers to the physical tray or rack within a given sampler (e.g. holder_A, holder_D, holder_F). Matches entries in the JSON field available_holders. 
| Position    | Well or coordinate label within the holder (e.g. A1, B4, E6). Defined by available_positions.                                                        |
| Type        | Vial category (e.g. Sample, Stock, Solvent, Cleaning, Gas, Mixing). Used by the control layer to define robot behavior.                              |
| Counter     | Integer counter for tracking how many times a vial has been accessed (e.g. for repeated sampling).                                                   |
| Viable      | Boolean or flag (True/False) indicating whether the vial is still usable (e.g. volume > threshold).                                                  |

Nested metadata (handler_info and available_*)

The VialDF entry in session.json includes nested dictionaries defining the hierarchy of handlers and their spatial mappings:
```json
"VialDF": {
  "handler_info": {
    "Sampler_cnc": {
      "holder_A": ["A1", "A2", "A3", ...],
      "holder_B": ["A1", "A2", "A3", ...]
    },
    "Collector_cnc": {
      "holder_D": ["A1", "A2", "A3", ...]
    }
  },
  "available_handlers": ["Collector_cnc", "Sampler_cnc"],
  "available_holders": ["holder_A", "holder_B", ...],
  "available_positions": ["A1", "A2", ..., "F6"]
}
```
---

## Relationships between files

- The **JSON** holds the global experiment state (ML model, parameters, reagent identities).  
- The **StockSolutionDF** describes what is physically present in stock containers.  
- The **VialDF** maps those stocks to vial positions and holders for robotic access.

At runtime, these objects are collocated in memory within the session container, and this exact state is serialized to `session.json`.  
This ensures experiment reproducibility while keeping the format transparent and inspectable.

## Excluded Files and Data Availability

During each experimental campaign, several auxiliary files are automatically generated alongside the session data — including detailed log files and result archives.
These files are not included in this repository for the following reasons:
1. Log files
	- Hundreds of detailed runtime and debugging logs are generated during automated operation, covering hardware communication, sensor feedback, ML model updates, and GUI events.
	- These logs are large (hundreds of megabytes), highly verbose, and primarily intended for internal troubleshooting rather than external inspection.
    - In addition, they may contain system paths, IP addresses, or other sensitive configuration details that should not be shared publicly.
	- For these reasons, log files have been omitted from version control.
2.	Results files
      - Processed experimental results are already stored and curated elsewhere, namely:
        - in the Supplementary Information (SI) accompanying the publication, and
        - in the Data Analysis/Visualization/ directory of this repository.
        - Including duplicate result files here would add redundancy and repository bloat without improving reproducibility, since all relevant data are already accessible and documented in those locations.

This approach keeps the repository lightweight, secure, and focused on the essential session structures (session.json, VialDF.csv, and StockSolutionDF.csv) that define the experimental context and ensure reproducibility.

---

## Best Practices

- Use the GUI to create or load sessions.  
- Use the JSON only to verify data integrity or troubleshoot.  
- Never edit IDs, lists, or field names manually — these are auto-managed.  
- Keep the three files together; they are **not interchangeable** between unrelated sessions.

---

**Notes:** The `session.json` format is automatically generated from the GUI runtime `session_state`. While human-readable for verification, it is **not intended for manual editing**. The design choice ensures full reproducibility and transparency while allowing flexibility in session management.

# 06 — Sample Prep & Liquid-Handling Automation

This report covers how a declarative slug recipe ("0.15 mM Ru(bpy)Cl photocatalyst + 500 mM SM in MeCN") is turned into the ordered sequence of pump fills, sampler moves, and injections that physically produce the reaction slug. The two central artefacts are the `StockSolutionDF` / `VialDF` tables (what liquids exist and where) and the `GenerateComposition` unit task (a linear program that picks stock volumes to hit target concentrations).

Code paths are quoted relative to the repo root. Unit-task source lives under `Control Software/OmniPlatypus/OmniPlatypus/omniplatypus/procedures/unit_tasks/` — this doc uses the shorthand `unit_tasks/…` for brevity.

## 1. Data model: `StockSolutionDF` and `VialDF`

The system keeps two user-editable tables that together describe the liquid inventory of a campaign.

**`StockSolutionDF`** (`Control Software/backend/Robochem_ML/robrains/parameter_backends/stocksolutiondf.py:16`) is a wide-format DataFrame with one row per stock bottle and one column per chemical. Its columns for CS1 (`Examples/CS1/StockSolutionDF.csv`) are:

```
StockID, Solvent, Conc_SM, Conc_Ru(bpy)Cl, Conc_Ru(bpy)PF6, Conc_Ir(ppy), …
Stock_SM,     MeCN, 1000, 0, 0, 0, …     # 1 M SM in MeCN
Stock_RuCl,   MeCN,    0, 3, 0, 0, …     # 3 mM Ru(bpy)Cl in MeCN
Stock_IrCF3,  MeCN,    0, 0, 0, 0, 3, …  # 3 mM Ir(CF3)ppy in MeCN
Stock_TFAA,   MeCN,    0, 0, 0, 0, 0, 0, 3500, …  # 3.5 M TFAA
```

Concentrations are in **mM** (see `RecipeComponent.concentration_units`, `unit_tasks/sampling/liquid_handler_sampling.py:939`). The column convention `Conc_<chemical_name>` is critical: `GenerateComposition.chemical_to_column` (`…:1147`) maps recipe component names directly to DataFrame columns.

**`VialDF`** (`…/robrains/parameter_backends/vialdf.py`) is the *physical* inventory. CS1 (`Examples/CS1/VialDF.csv`) opens with:

```
VialID, VialName,       StockID,      Volume, Sampler,      Holder,   Position, Type,     Counter, Viable
0x001,  SM,             Stock_SM,     4000,   Sampler_cnc,  holder_D, A1,       Stock,    0,       TRUE
0x003,  sol_MeCN_1,     MeCN,         10000,  Sampler_cnc,  holder_F, B1,       Solvent,  0,       TRUE
0x005,  wash_MeCN_1,    Not a Stock,  10000,  Sampler_cnc,  holder_F, A1,       Cleaning, 0,       TRUE
0x007,  N2,             Not a Stock,  0,      Sampler_cnc,  holder_A, D1,       Gas,      0,       TRUE
0x008,  Mixing,         Not a Stock,  0,      Sampler_cnc,  holder_A, D2,       Mixing,   0,       TRUE
```

Each physical vial therefore carries: a unique `VialID`, the `StockID` it was filled from (inherited concentration profile), remaining `Volume` in µL, its `Sampler`/`Holder`/`Position` (physical coordinates), a `Type` that drives task filtering (`Stock`, `Solvent`, `Cleaning`, `Gas`, `Mixing`, `Waste`, `Reaction`…), and a `Viable` flag that the orchestrator toggles off when a vial has been exhausted.

At platform startup, `GenerateSampleDataframe` (`unit_tasks/sampling/liquid_handler_sampling.py:292`) joins the `VialDF` with the `StockSolutionDF` to produce a per-vial concentration profile plus physical `X`,`Y` coordinates (resolved from the sampler's `setup.locations` + the holder geometry). The resulting table is stored on the platform as `platform.samples` and is the sole DataFrame that downstream tasks read — there is no direct access to `VialDF`/`StockSolutionDF` from the run loop.

## 2. Recipe composition: `GenerateComposition` (the LP solver)

`GenerateComposition` (`unit_tasks/sampling/liquid_handler_sampling.py:1069`) is the heart of sample prep. Its signature (`:1088`) is:

```python
GenerateComposition.run(
    platform,
    experiment_recipe: list[RecipeComponent],   # target chemicals + concentrations in mM
    slug_volume: float,                          # µL
    sampler_name: str,
    max_relative_error: float = 0.05,
    min_volume: float = 0.5,                     # µL — drop-ingredient threshold
    min_accurate_volume: float = 20.0,           # µL — retry-with-alternative-stock threshold
) -> tuple[list[VialRecipeComponent], list[RecipeComponent]]
```

The comment block at `:1348–1395` lays out the LP directly:

> `Ax + s = C`, where **A** is the concentrations-in-vials matrix (rows = unique stock profiles, cols = compounds), **C** is the desired moles vector (`target_concentration × slug_volume`), **x** is the unknown per-stock-vial volume. The objective minimizes `Σ x_i` (total solvent/stock volume used).

The actual call is SciPy's HiGHS LP:

```python
opt = linprog(
    c=objective,          # np.ones(n_vials) — minimize total volume
    A_ub=-A.T,            # -A_ij ≤ -C_j  ⇔  Σ_i A_ij x_i ≥ C_j  (concentration floor)
    b_ub=-moles_vector_c,
    bounds=bnd,           # per-vial volume bounds from _calculate_bounds
    method="highs",
)
```

Quality gates after the solve (`:1396–1423`):

1. Each volume is **rounded** to the precision implied by `min_volume` (`round_to_digits = -⌊log₁₀(min_volume)⌋`). For CS1's default `min_volume=0.5 µL` this means rounding to the nearest 0.5 µL — enforcing that the recipe respects the syringe's actual dispensable resolution.
2. Back-computed concentrations are compared against targets; any compound whose relative error exceeds `max_relative_error` (5 % default) triggers a `RecipeError` with a per-compound note. The error propagates up to the experiment loop, where `BaseExperiment._run` catches it as a skippable iteration rather than a halt (see report 10).
3. If an ingredient needs `< min_accurate_volume` (20 µL) the code searches for an alternative lower-concentration stock that would push the required volume into the accurate range (recursive retry via `stashed_recipies`, excluding the first-choice vial).

The output is a `list[VialRecipeComponent]` — one entry per physical vial that will be sampled, each with `vial_id`, `volume_uL`, and `sampling_priority` — plus the adjusted `RecipeComponent` list whose concentrations reflect the rounding.

## 3. Scheduling: `FindVial` and `OrderRecipe`

`FindVial` (`unit_tasks/sampling/liquid_handler_sampling.py:725`) is the point-lookup: "closest viable vial of type X that has room for Y µL". It consults `platform.samples`, filters by `Type`, `Viable`, and (optionally) volume-after-operation, then orders candidates by Euclidean distance from either the sampler's current position or a reference sample's position. If no candidate is found and `allow_user_requests=True`, it enqueues a `UserActionRequest` on the experiment's `_action_request_queue` (see report 04); the user must load a fresh vial before the task resolves. Otherwise it raises `NoSuitableVialError` (`unit_tasks/errors.py`), which the experiment loop can either propagate or skip depending on context.

`OrderRecipe` (`:2180`) takes the `VialRecipeComponent` list from `GenerateComposition` and reorders it for the actual dispense: high-priority components (tagged by `RecipeComponent.sampling_priority`, default `1000`) are drawn first so they don't sit in the needle or slug while slower dilutions are prepared. Distance-aware ordering also reduces needle travel.

## 4. Pump actions — the driving side

`unit_tasks/driving/driving_pumps.py` defines four tasks that operate on a `SyringePump` device:

| Class | Lines | Purpose |
|---|---|---|
| `PrimePump` | `:231` | Flush pump + lines with a wash solvent before use. Typically runs once per campaign, not per slug. |
| `FillPump` | `:367` | Draw `volume` µL from a specified vial into the barrel. Called once per unique stock-vial component of the recipe. |
| `PumpVolume` | `:490` | Push a specified volume out of the barrel (downstream to reactor or waste). This is the canonical dispense — see report 03 for the wire trace: host emits `S9=<uL>\n`, firmware acks `'k'`. |
| `MixSlug` | `:721` | Oscillate the plunger to mix the contents of the barrel before dispensing — used when multiple stocks were drawn in sequence to homogenise the slug. |

Every task inherits from `PumpTask` (`:24`), which in turn extends `BaseUnitTaskTemplate` from `unit_tasks/base_unit_task.py`. The `validate → execute → cleanup` contract there ensures a failing pump operation is wrapped as `TaskTimeoutError` or `DeviceCommunicationError` (report 10) rather than leaking `pyserial` exceptions.

## 5. Sampler actions — the moving/injecting side

Two sibling files handle the XY(Z) Cartesian sampler:

- **`unit_tasks/sampling/liquid_handler_moving.py`** — `SamplerTask` (`:16`) is the sampler-aware base. `HomeSampler` (`:188`) homes the gantry; generic movement lives in other `SamplerTask` subclasses invoked internally when `Inject` / `ConnectInjectionPort` need to reach a vial.
- **`unit_tasks/sampling/liquid_handler_injecting.py`** — `ConnectInjectionPort` (`:166`) couples the needle to a vial's septum; `Inject` (`:228`) performs the actual piercing/aspiration/dispense at a vial. Both are where the abstract `VialRecipeComponent` meets the physical vial — they consume a `VialID`, resolve its `(Holder, Position)` → `(X, Y)` via `platform.samples`, drive the sampler, then delegate to the syringe pump for the actual fluid motion.

Under the hood the CNC sampler used by CS1 is `Sampler_cnc` — a Cartesian Arduino-driven device whose firmware is documented in report 09. Its coordinate origin and per-holder offsets come from the platform config's `sampler.setup.locations` block (report 02).

## 6. Worked example: one CS1 slug

Suppose the BO backend proposes one experimental point that asks for **0.15 mM Ir(CF3)ppy**, **500 mM SM**, **50 mM TFAA**, **500 µL slug** in MeCN. The flow (conceptual — the real numbers depend on the solve):

1. **Recipe construction.** `ToandFromMachine._to_machine` (report 05) converts the ML tensor into three `RecipeComponent`s. The experiment calls `GenerateComposition.run(platform, recipe, slug_volume=500, sampler_name="Sampler_cnc")`.
2. **LP solve.** The unique stock profiles considered are `Stock_SM (SM=1000)`, `Stock_IrCF3 (Ir(CF3)ppy=3)`, `Stock_TFAA (TFAA=3500)`, plus solvent. Target moles in 500 µL are `SM: 0.5·500·1e-3=0.25 µmol`, `Ir(CF3)ppy: 0.075 µmol`, `TFAA: 25 µmol`. HiGHS returns approximately:
   - Stock_SM:    **250 µL** (gives 1000 mM × 250 µL = 250 nmol → 500 mM in 500 µL ✓)
   - Stock_IrCF3: **25 µL**  (3 mM × 25 µL = 75 nmol → 0.15 mM in 500 µL ✓)
   - Stock_TFAA:  **7.5 µL** — **below `min_accurate_volume=20 µL`**, triggering the alternative-stock retry. With no alternative TFAA stock defined in CS1's inventory, the code falls through and keeps the 7.5 µL value (rounded to the nearest 0.5 µL) with a logged warning.
   - Pure MeCN (from `Stock_MeCN` or a `Solvent`-type vial): **217.5 µL** as makeup.
3. **Vial scheduling.** `FindVial` is called per stock: it returns `0x001 (Stock_SM, holder_D A1)`, the nearest viable Ir(CF3)ppy stock, etc. `OrderRecipe` emits them in descending sampling priority so SM (bulk) goes last and catalyst (trace) goes first, minimising cross-contamination risk.
4. **Physical execution.** For each entry the experiment executes `SamplerTask` → move to vial → `ConnectInjectionPort` → `FillPump(volume=X)` to draw that stock into the barrel; repeat; then `MixSlug` to homogenise the 500 µL slug; then a `SamplerTask` move to the injection port of the reactor, followed by `PumpVolume(500)` to push the whole slug into the photoreactor at the proposed flow rate. The gas-segmented phase sensor (report 09) gates the downstream analytical step.
5. **Bookkeeping.** Each `FillPump` decrements the corresponding vial's `Volume` column in `platform.samples` by the drawn amount; when a vial drops below the next task's required volume, `FindVial` transparently switches to the next parallel vial (e.g. `0x002 SM2` for the SM reservoir). The experiment loop only sees `RecipeError` or `NoSuitableVialError` if the whole inventory is exhausted.

## 7. Summary

The sampling stack is **declarative at the top and procedural at the bottom**: the user writes a chemistry recipe in `mM`/`µL`, an LP picks physical stock volumes that satisfy it within equipment-accuracy constraints, and a Cartesian sampler + syringe pump execute the draws in a priority-aware order. The two DataFrames (`StockSolutionDF`, `VialDF`) are the single source of truth for the liquid world, and their joined `platform.samples` view is the only structure the runtime queries — which makes the whole system largely decoupled from the specific robot: any Cartesian sampler + syringe pump that implements the respective device APIs can be dropped in without touching the LP or scheduler.

# 07 — Analytics & Measurement Feedback

This report traces the life of a single measurement through robochem_flex: from the moment a Raman/NMR/HPLC/UV-Vis device emits raw numbers, to the scalar `yield`/`conversion`/`concentration` figure that the experiment scheduler (and by extension the ML optimizer) consumes. The pipeline is split between two packages: `omniplatypus.procedures.analytics` contains the hardware-aware wrapper classes that implement the experiment-facing API, and the `lamas` package provides the science: baseline correction (Alpaca), peak fitting (Vicuna), calibration (Guanaco), and concentration prediction (Lama).

## 1. `AnalyticsTemplate` — the contract

Every analytical modality subclasses `AnalyticsTemplate` (`Control Software/OmniPlatypus/OmniPlatypus/omniplatypus/procedures/analytics/analytics_template.py:41`). The contract is intentionally thin: a subclass must implement a single abstract method, `analyse(conditions, recipe) -> dict`, and declare its parameter surface via three class attributes:

- `_required_parameters: list[AnalyticalParameter]` — must be set for every run (template.py:49).
- `_optional_parameters: list[AnalyticalParameter]` — defaulted if missing (template.py:52).
- `_processing_methods: list[str]` — algorithm selectors (template.py:56).
- `_lama_parameters: Dict[str, Any] = LamaSheperd.structure.copy()` — introspects the four LAMAS structures (template.py:57).

The template also provides the convenience utilities each subclass re-uses: `validate_parameters` fills in optional defaults and raises on missing required ones (template.py:171); `get_reference_concentration` walks the `recipe` list to find the limiting reagent's concentration in mM (template.py:219); `get_matching_peak` finds the peak closest to a target RT/ppm, filtering by spectral correlation if a reference is supplied (template.py:252); `get_concentration` applies a polynomial calibration of arbitrary order (template.py:311):

```python
def get_concentration(self, integral: float, coefficients: list[float]) -> float:
    concentration = 0.0
    for exponent in range(len(coefficients)):
        concentration += coefficients[exponent] * (integral**exponent)
    return concentration
```

The result of `analyse()` is a plain `dict`. It is stored as `RunResult.result` in `experiment_parameters.py:222`, and the experiment driver converts the dict's `"pass"` key into the run's boolean `success` at `chemistry.py:1103-1106`:

```python
if results.result is not None:
    results.success = results.result.get("pass", True)
else:
    results.success = True
```

This is the single contract seen by the ML side — `RunResult.result` is a dict and `RunResult.success` is a bool. The dict keys each analyser guarantees are enumerated in its `result_metrics` class attribute.

## 2. Data flow: acquisition → preprocessing → fitting → matching → prediction

The LAMAS animal kingdom mirrors the four-step pipeline. `LamaSheperd.structure` (`Control Software/Lamas/lamas/lama_sheperd.py:18`) bundles them:

```python
class LamaSheperd:
    structure = {
        "Alpaca": Alpaca.structure,
        "Guanaco": Guanaco.structure,
        "Vicuna": Vicuna.structure,
        "Lama":  Lama.structure,
    }
```

**Alpaca** (`alpaca.py:20`) ingests raw x/y data (`load_data(..., from_filetype="ramaberry"|"nmr_lama"|...)`) and exposes `crop_data`, `normalise_to`, `denoise`, and `baseline`. Baseline offers six algorithms dispatched in `alpaca.py:464-478`: `cholesky`, `linear_fit`, `als`, `rubberband`, `poly_fit`, `wavelet`. Denoise offers `fft`, `moving_average`, `wavelet`, `median`, `savgol` (alpaca.py:513-527). Normalisation can be by integrated area or by peak height (alpaca.py:359-427).

**Vicuna** (`vicuna.py:33`) fits a spectrum as a sum of Voigt/Lorentzian/Gaussian peaks. `fit_all` has three termination modes — `peak_limit`, `residual_stabilization`, `no_more_successful_fits` (vicuna.py:79-91) — so the iterative "add peak, fit, check residual" loop halts on whichever first fires. Output is a `List[VoigtPeakIdentification]`, each peak carrying `mu`, `A`, `fwhm_g`, `fwhm_l`, etc.

**Guanaco** (`guanaco.py:33`) clusters peaks across multiple training spectra of known concentration, producing a `VoigtLinearFitResult` per compound/peak. Peak-to-peak matching uses the same similarity modes as Lama (relative_threshold, weighted_similarity, mu_only, correlation, fwhm, overlap_area, dtw, cross_correlation — guanaco.py:65-78). The result is `filtered_peak_fits: Dict[str, List[VoigtLinearFitResult]]`.

**Lama** (`lama.py:28`) ingests Guanaco's fits via `absorb_fits_from_class` (lama.py:183) and then `predict_concentrations(new_spectrum, known_compounds, mode=...)` assigns each new peak to a compound using the slope/intercept of the calibration line (lama.py:351-355):

```python
similarity_score = fit_result.check_similarity_with_peaks(
    new_peak, self.peak_same, **kwargs
)
if similarity_score == True:
    estimated_concentration = (
        fit_result.slope_amplitude * new_peak.A
        + fit_result.intercept_amplitude
    )
    compound_concentrations[compound].append(estimated_concentration)
```

If multiple peaks match one compound, `_calculate_compound_concentrations` (lama.py:372) returns the mean and stdev across them, plus `num_peaks_used`. Unmatched peaks are carried in the return value's `unmatched_peaks` list — these are the anomaly signal.

## 3. Per-modality behaviour

### Raman (`raman_analysis.py`)
`AnalyticsRaman` is the most deeply integrated with LAMAS. Its `result_metrics` are `["yield", "integral", "integral_starting_material", "pass"]` (raman_analysis.py:53). The class owns a `RamaBerry` SocketDevice (raman_analysis.py:21; device at `omniplatypus/devices/nrg/rama_berry.py:40`), which is a thin socket client to a Pi-hosted Avantes AvaSpec server.

`analyse()` (raman_analysis.py:222) does: (1) validate params, (2) optional fluorescence check (a 1-s / 1-avg acquisition whose max intensity must stay below `fluorescence_threshold=800` — raman_analysis.py:294-334), (3) set parameters on the device, (4) start acquisition and block-poll in `_spectrometer_poll` (raman_analysis.py:380), (5) dispatch to a named processing function selected by the `processing_function` optional parameter (raman_analysis.py:173-182). The default is `processing_function_integrated_isotope_exchange`, discovered via introspection of `lama_analytics` at class definition time.

The processing function in `lama_analytics.py:27` wires Alpaca steps directly:

```python
spectrum = Alpaca(logger=logger)
spectrum.load_data(data, from_filetype="ramaberry")
spectrum.crop_data(bounds=crop_range)
spectrum.baseline(method="cholesky", lam=10**6, p=0.03, n_iter=20)
spectrum.normalise_to(to=1000, range_vals=normalise_range)
spectrum.crop_data(bounds=(1520, 2000))
spectrum.denoise(method="median", window=5)
spectrum.crop_data(bounds=crop_zoom)
spectrum.baseline(method="als", lam=10**6, p=0.03, n_iter=20)
integral_product = spectrum.integrate(bounds=bounds_product, mode="area")
integral_starting_material = spectrum.integrate(bounds=bounds_starting_material, mode="area")
```

The integration bounds are derived from user-given peak-of-product, peak-of-starting-material, and `isosbestic_point` (a spectral crossing point used as the boundary between the two peaks). `_make_results` (raman_analysis.py:490) then turns the two integrals into yield/conversion using a calibration linear model loaded from disk (see §4). The pass criteria dict (raman_analysis.py:650) gates `results["pass"]` on seven conditions including `product_variance_not_excessive: sqrt(pi_var) / reference_concentration < 0.50` — the single most important guard against noisy fits being fed back to the optimizer.

### NMR (`nmr_analysis.py`)
`NMRAnalysis` integrates against a Magritek Spinsolve via the `SpinsolveClient` (imported `nmr_analysis.py:20`; started via `self._device["start"] = protocol` at nmr_analysis.py:192). The unique workflow step is `open_spinsolve` (nmr_analysis.py:221) which reads the Spinsolve output directory with `nmrglue.spinsolve.read`, then applies phase correction with `nmrglue.process.proc_autophase.autops` using the `peak_minima` method (nmr_analysis.py:302), strips imaginaries, Gaussian-smooths (sigma=5), and baselines via the `median` method from `custom_baseline_methods.baseline` (nmr_analysis.py:322-324). Peak detection is pure `scipy.signal.find_peaks` with a noise floor estimated by median absolute deviation over the central 75% of intensities (nmr_analysis.py:328-357). The scalar output path is: `best_peak = self.get_matching_peak(...)` → `self.get_concentration(best_peak["integral"], coefficients=[coeff_0, coeff_1])` → `yield = concentration / target_concentration * 100` (nmr_analysis.py:542-563). NMR therefore uses the simple linear calibration, not the LAMAS chain.

### HPLC (`hplc_analysis.py`)
`HPLCAnalysis` targets a Knauer stack through `chromtroller`. The device already returns peaks integrated: `raw_result["data"]` is a dict that becomes a `pd.DataFrame` of peak RT/integral/spectral-correlation columns (hplc_analysis.py:255). The analyser's job is matching rather than processing: `get_matching_peak(peaks, target_column="peak_rt", target=target_rt, max_deviation=0.1, reference_spectrum="target", min_spectral_correlation=60.0)` — it picks the closest peak in retention time whose UV spectrum matches the stored reference by at least 60% (hplc_analysis.py:258). The same loop runs for each of up to four sideproducts A–D (hplc_analysis.py:289). Each matched peak is converted with `get_concentration` using product-specific `*_calibration_coeff_0` and `*_calibration_coeff_1` params (hplc_analysis.py:92-150). HPLCProcessingSettings fields are auto-expanded into AnalyticalParameters (hplc_analysis.py:151-164).

### UV-Vis (`uv_analytics.py`)
`AnalyticsUV` is a near-clone of the Raman class, reusing the RamaBerry device interface and `_process_analytics`/`_make_results` identically (uv_analytics.py:44, 488). Its `_processing_methods` list distinguishes `single_point_absorbance` from `integrated_absorbance` (uv_analytics.py:218), and `result_metrics` is `["yield", "integral", "absorbance_at_wlen", "pass"]`. The class includes a `molar_extinction_coefficient` parameter for Beer-Lambert conversion (uv_analytics.py:162-169).

## 4. Calibration models

Calibrations come in two flavours:

1. **Simple polynomial (NMR/HPLC).** Two scalars per compound (`*_calibration_coeff_0`, `*_calibration_coeff_1`) live inside the experiment YAML and default to sane values in the `_optional_parameters` lists (e.g., hplc_analysis.py:92-102; nmr_analysis.py:140-151). They are consumed by `AnalyticsTemplate.get_concentration` and can be regenerated offline from CSVs of known-concentration runs.

2. **Linear model bundle (Raman/UV).** `Examples/CS4/integral_models.linear_model` and `Data Analysis/Isotope Effect Calibration/integral_models.linear_model` are joblib-pickled dicts. `load_linear_model` (lama_analytics.py:196) enforces the `.linear_model` extension and `joblib.load`s a dict containing `bins_PI`, `bins_SM`, `stds_PI`, `stds_SM` (at minimum) plus — when full confidence intervals are enabled — `model_PI`, `X_PI`, `y_PI` (currently commented out, lama_analytics.py:147-184). `interpolate_std_for_new_integral` (lama_analytics.py:259) performs piecewise-linear interp/extrapolation of the per-bin stdev to obtain a variance estimate around the new integral. The notebook `Data Analysis/Isotope Effect Calibration/exploratory_plots.ipynb` is the end-to-end training recipe: it imports Alpaca and Vicuna, loops across known-concentration spectra, and ends with the saved `.linear_model` file used at runtime.

The path to the calibration file is just another `AnalyticalParameter("path_to_calibration_file", tag="all")` (raman_analysis.py:172), so recipes can swap calibrations without code changes.

## 5. `result_metrics` / `RunResult` contract

The experiment consumes `results.result` as an opaque `dict` (chemistry.py:1022). Guarantees:

- Every analyser declares `result_metrics` and initialises the dict either to `0.0`/`np.nan` or to `None` before selectively overwriting fields.
- All analysers emit `pass: bool`. Its meaning varies: HPLC defaults to `True` and fails only on no-peaks (hplc_analysis.py:254); NMR fails if `FWHM >= max_peak_width` (nmr_analysis.py:548); Raman fails via a seven-criterion AND at raman_analysis.py:650-664.
- Additional scalar keys (yield, conversion, concentration, integral, widths, retention times, variances) supply the ML optimizer's objective and — for modalities that compute it — the aleatoric uncertainty via `*_variance` keys. For Raman these are derived from the calibration bin stdevs (lama_analytics.py:186-193).
- `chemistry.py:1104` collapses the dict to `results.success = result.get("pass", True)`; downstream ML code reads both the scalar metric(s) and this bool.

## 6. Failure modes

**Bad fits.** If Vicuna never terminates successfully or the processing function returns `None`, `processing_function_integrated_isotope_exchange` substitutes zeros (lama_analytics.py:109-116), which propagates as `yield=0` but not necessarily `pass=False`. The Raman variance check (`sqrt(pi_var)/reference_concentration < 0.50`, raman_analysis.py:660) is the catch-net: a noisy fit whose stdev exceeds 50% of the reference concentration fails the run.

**Out-of-calibration samples.** `interpolate_std_for_new_integral` extrapolates linearly beyond the outermost bin (lama_analytics.py:275-285), so the variance grows unbounded — extreme concentrations are flagged through the same variance guard rather than a hard range check. If `path_to_calibration_file` is missing or not a `.linear_model`, `load_linear_model` returns `None` and the processing function silently degrades to returning just the raw areas (lama_analytics.py:122-133), leaving the optimizer without a yield but with an integral it can still rank.

**Anomaly detection in `Lama.predict_concentrations`.** Peaks that match no known compound are returned as `unmatched_peaks` (lama.py:266-269). Compounds with zero matched peaks get `{"predicted_concentration": 0.0, "error": 0.0, "num_peaks_used": 0}` (lama.py:401-405) and a warning is logged. This lets the caller distinguish "compound absent" from "spectrum broken" (many unmatched peaks → suspicious sample).

**Device-level failures.** Raman/UV enforce a fluorescence check (raman_analysis.py:294) that aborts before acquisition if the blank-read exceeds threshold, returning `{metric: None, "pass": False}` for every key. `_spectrometer_run` raises `AnalysisError("No data was returned")` (raman_analysis.py:374-376) if the socket returns empty. NMR `wait_for_file` raises `RuntimeError` after a 10-s timeout if Spinsolve never writes `spectrum_processed.1d` (nmr_analysis.py:214). All of these bubble up as exceptions stored in `RunResult.exception`, which the chemistry driver catches and flags as `success=False`.

The net effect: from raw spectrum to optimizer-ready scalar, every modality converges on the same `dict + pass bool` contract, with the LAMAS chain supplying the science for Raman/UV and scipy/nmrglue for NMR/HPLC.

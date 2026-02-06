<a id="lamas"></a>

# lamas

<a id="lamas.glama"></a>

# lamas.glama

Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: repo of base classes for each of the backends

<a id="lamas.glama.Glama"></a>

## Glama Objects

```python
class Glama()
```

Base class for ALPACA, GUANACO, lamas, and VICUNA. Handles I/O operations, logging, and shared utility functions
such as Gaussian, Lorentzian, Voigt, and Savitzky-Golay filtering.

<a id="lamas.glama.Glama.data"></a>

#### data

data after processing

<a id="lamas.glama.Glama.x_col"></a>

#### x\_col

the column name for the x-axis values

<a id="lamas.glama.Glama.y_cols"></a>

#### y\_cols

the column names for the y-axis values

<a id="lamas.glama.Glama.raman_required_kwargs"></a>

#### raman\_required\_kwargs

required keyword arguments for Raman data

<a id="lamas.glama.Glama.nmr_required_kwargs"></a>

#### nmr\_required\_kwargs

required keyword arguments for NMR data

<a id="lamas.glama.Glama.load_data"></a>

#### load\_data

```python
def load_data(input_data: Union[str, pd.DataFrame],
              x_col: str = None,
              y_cols: List[str] = None,
              from_filetype: Union[str, None] = None) -> None
```

Load data from a CSV file or pandas DataFrame.

**Arguments**:

- `input_data`: str (path to CSV) or pandas DataFrame
- `x_col`: str, column name for x-axis values
- `y_cols`: list of str, column names for y-axis values
- `from_filetype`: str, file type of the data allowed values are:
Ramaberry (for our raman spectrometer),
lama_nmr(from nmr data previously processed by lamas),
lama_raman( from raman data previously processed by lamas),
Bruker (NMR data from bruker),
Spinsolve (NMR data from magritek spinsolve)
jdx (JCAMP-DX file)

<a id="lamas.glama.Glama.set_spectroscopy"></a>

#### set\_spectroscopy

```python
def set_spectroscopy(spectroscopy: str = "Raman",
                     required_args: List[initkwargs] = None) -> None
```

Initialize the Alpaca class with the given spectroscopy type and required arguments.

**Arguments**:

- `spectroscopy`: str: type of spectroscopy data to preprocess, default is "Raman"
- `required_args`: list of InitKwargs: required keyword arguments specific to the spectroscopy type

<a id="lamas.glama.Glama.absorb_class"></a>

#### absorb\_class

```python
def absorb_class(class_data: Glama) -> None
```

Absorb the data from another glama-type class. (i.e. going from Alpaca to Lama)

<a id="lamas.glama.Glama.save_data"></a>

#### save\_data

```python
def save_data(output_path: str | None = None) -> None
```

Save the processed data to a CSV file.

**Arguments**:

- `output_path`: str, path to the output CSV file. If output_path is None,
a filename will be generated based on timestamp and spec_type,
and it will save to self.data_parent_dict if available.

**Raises**:

- `ValueError`: If neither output_path nor self.data_parent_dict is set,
preventing random saving in the code directory.

<a id="lamas.glama.Glama.gaussian"></a>

#### gaussian

```python
@staticmethod
def gaussian(x: np.ndarray, mu: float, sigma: float) -> np.ndarray
```

Function for a Gaussian peak.

**Arguments**:

- `x`: array-like, x-axis values
- `mu`: float, mean of the Gaussian
- `sigma`: float, standard deviation of the Gaussian

**Returns**:

array-like, Gaussian function values

<a id="lamas.glama.Glama.lorentzian"></a>

#### lorentzian

```python
@staticmethod
def lorentzian(x: np.ndarray, mu: float, gamma: float) -> np.ndarray
```

Function for a Lorentzian peak.

**Arguments**:

- `x`: array-like, x-axis values
- `mu`: float, location parameter (mean) of the Lorentzian
- `gamma`: float, scale parameter (half-width at half-maximum) of the Lorentzian

**Returns**:

array-like, Lorentzian function values

<a id="lamas.glama.Glama.voigt"></a>

#### voigt

```python
@staticmethod
def voigt(x: np.ndarray, A: float, mu: float, sigma: float,
          gamma: float) -> np.ndarray
```

Pseudo-Voigt function, which is a combination of Gaussian and Lorentzian peaks.

**Arguments**:

- `x`: array-like, x-axis values
- `A`: float, amplitude of the peak
- `mu`: float, mean of the peak
- `sigma`: float, standard deviation of the Gaussian component
- `gamma`: float, scale parameter of the Lorentzian component

**Returns**:

array-like, Pseudo-Voigt function values

<a id="lamas.glama.Glama.find_fwhm"></a>

#### find\_fwhm

```python
@extract_columns
def find_fwhm(x: np.ndarray, y: np.ndarray, y_max: float) -> float
```

Find the full width at half maximum (FWHM) of the peak defined by x and y.

**Arguments**:

- `x`: array-like, x-axis values
- `y`: array-like, y-axis values
- `y_max`: float, maximum y value of the peak

**Returns**:

float, FWHM value

<a id="lamas.glama.Glama.crop_data"></a>

#### crop\_data

```python
def crop_data(bounds: Tuple[float, float]) -> None
```

Crop the data within the specified x-axis range.

**Arguments**:

- `bounds`: tuple of floats, lower and upper bounds for the x-axis

<a id="lamas.glama.Glama.peak_same"></a>

#### peak\_same

```python
@staticmethod
def peak_same(peak1: VoigtPeakIdentification,
              peak2: VoigtPeakIdentification,
              mode: Literal[
                  "relative_threshold",
                  "weighted_similarity",
                  "mu_only",
                  "correlation",
                  "fwhm",
                  "overlap_area",
                  "dtw",
                  "cross_correlation",
              ] = "relative_threshold",
              threshold: float = 0.05,
              weights: dict = None,
              **kwargs) -> float
```

Checks for similarity between two VoigtPeakIdentification instances based on the selected mode.

**Arguments**:

- `peak1`: VoigtPeakIdentification, the first peak to compare.
- `peak2`: VoigtPeakIdentification, the second peak to compare.
- `mode`: str, the comparison mode to use:
- "relative_threshold": Checks if parameters are within a relative difference threshold.
- "weighted_similarity": Calculates a weighted similarity score.
- "mu_only": Only checks the center position (mu).
- "correlation": Compares the peak profiles based on amplitude-normalized overlap.
- "fwhm": Compares the FWHM (Full Width at Half Maximum) of both peaks.
- "overlap_area": Calculates the area of overlap between the two normalized peak profiles.
- "dtw": Uses Dynamic Time Warping to calculate similarity.
- "cross_correlation": Uses cross-correlation for similarity of peak profiles.
- `threshold`: float, the relative difference threshold for applicable modes.
- `weights`: dict, parameter weights for "weighted_similarity" mode.

**Returns**:

float, similarity score between 0 and 1 (1 indicates high similarity).

<a id="lamas.glama.Glama.load_model_from_class"></a>

#### load\_model\_from\_class

```python
def load_model_from_class(vicuna_class: "Vicuna")
```

loads model from a vicuna class, if the vicuna class has multiple models it will load all of them,
alternatively it will load the first model

<a id="lamas.vicuna"></a>

# lamas.vicuna

Author: Elia Savino
GitHub: github.com/EliaSavino

Happy Hacking!

Descr:
Vicunas are a subspecies of Llamas known to be the best peak fitters in the animal kingdom, due to their ability to climb
the highest peaks in the Andes mountains.

This module does something similar, it takes a spectrum and proceeds to automatically fit a number of Voigt peaks to describe
the spectrum for our deconvolutions later.

<a id="lamas.vicuna.Vicuna"></a>

## Vicuna Objects

```python
class Vicuna(Glama)
```

This class is responsible for fitting the peaks and generating model spectra
Usage: instantiate the class Vicuna(**kwargs), kwargs see Glama class for more info

load the data with the load_data method (for doc see Glama class)

OPTIONAL: provide concentration of the components in the spectrum with the provide_concentrations method
Vicuna.provide_concentrations(["column_name1", "column_name2"], [{"Conc_A": 1.0, "Conc_B": 0.5}, {"Conc_A": 0.5, "Conc_B": 1.0}]) (this will be made into a DF available for Guanaco Later)

fit the peaks with the fit_all method:
Vicuna.fit_all(**kwargs)
kwargs:
fitting kwargs
    termination_method: list of str, termination criteria to apply.
                                    Options: 'peak_limit', 'residual_stabilization', 'no_more_successful_fits'
    residual_threshold: float, threshold for residual stabilization.
    stabilization_window: int, number of iterations to consider for stabilization.
    min_improvement: float, minimum residual improvement to consider a fit successful.
baseline kwargs:
   baseline_method: str, the method to use for baseline generation. (default: "als", see Glama for available options) each method requires different kwargs, see Glama for more info

FWHM calculation kwargs:
    distance: int, minimum distance between peaks (default: 1)
    prominence: float, minimum prominence of peaks (default: 0.1)
    height: float, minimum height of peaks (default: None)
    peak_multiplier: int, multiplier for the number of peaks (default: 3)
    see Glama._find_peaks for more info

<a id="lamas.vicuna.Vicuna.__init__"></a>

#### \_\_init\_\_

```python
def __init__(**kwargs) -> None
```

Initialize the Vicuna class by calling the superclass initializer.

<a id="lamas.vicuna.Vicuna.provide_concentrations"></a>

#### provide\_concentrations

```python
def provide_concentrations(y_col: List[str],
                           concentrations: List[Dict[str, float]]) -> None
```

Provide the concentration of each component known to be in the spectrum. This is saved in a separate DataFrame.

**Arguments**:

- `y_col`: List of filenames or columns the concentrations correspond to.
- `concentrations`: List of dictionaries containing concentration values for each component.
Each dictionary may have different combinations of compounds
(e.g., {"Conc_A": 1.0, "Conc_B": 0.5}).

<a id="lamas.vicuna.Vicuna.extend_to_baseline"></a>

#### extend\_to\_baseline

```python
def extend_to_baseline(x_fit,
                       y_fit,
                       y_baseline,
                       side="right",
                       num_fit_points=3) -> np.ndarray
```

Extend x_fit and y_fit on the specified side using linear extrapolation until y_fit crosses the baseline.

**Arguments**:

  - x_fit: Original x data (1D array).
  - y_fit: Original y data (1D array).
  - y_baseline: Baseline y data (1D array).
  - side: 'left' or 'right' indicating which side to extend.
  - num_fit_points: Number of boundary points to use for linear fitting.
  

**Returns**:

  - x_ext: Extended x data for the specified side (1D array).
  - y_ext: Extended y data for the specified side (1D array).

<a id="lamas.vicuna.Vicuna.fit_all"></a>

#### fit\_all

```python
def fit_all(termination_method: list = [
    "peak_limit",
    "residual_stabilization",
    "no_more_successful_fits",
],
            residual_threshold: float = 5,
            stabilization_window: int = 5,
            min_improvement: float = 1e-4,
            **kwargs)
```

Fit Voigt peaks for all y_cols until stopping conditions are met based on termination_method.

**Arguments**:

- `termination_method`: list of str, termination criteria to apply.
Options: 'peak_limit', 'residual_stabilization', 'no_more_successful_fits'
- `residual_threshold`: float, threshold for residual stabilization.
- `stabilization_window`: int, number of iterations to consider for stabilization.
- `min_improvement`: float, minimum residual improvement to consider a fit successful.

<a id="lamas.vicuna.Vicuna.voigt_to_df"></a>

#### voigt\_to\_df

```python
def voigt_to_df() -> pd.DataFrame
```

Converts the dictionary of Voigt parameters into a single DataFrame.

Each key in the dictionary is added as a new column ('y_col') in the resulting DataFrame,
which contains all rows from the original DataFrames.

**Returns**:

- `pd.DataFrame` - A concatenated DataFrame with an additional 'y_col' column representing
  the dictionary keys.

<a id="lamas.vicuna.Vicuna.save_voigt_parameters"></a>

#### save\_voigt\_parameters

```python
def save_voigt_parameters(path: Optional[str] = None) -> None
```

Saves the Voigt parameters to a CSV file.

If a path is not provided, and the instance has a 'data_parent_dir' attribute, the file
is saved to "voigt_parameters.csv" in that directory. Otherwise, an exception is raised.

**Arguments**:

- `path` _Optional[str]_ - The parent directory path.'data_parent_dir'.
  

**Raises**:

- `Exception` - If no path is provided and 'data_parent_dir' is not set.

<a id="lamas.vicuna.Vicuna.plot_stuff"></a>

#### plot\_stuff

```python
def plot_stuff()
```

debugging function to plot the data, model and residual

<a id="lamas.logger"></a>

# lamas.logger

File: logger.py
Author: Simone Pilon - Noël Research Group - 2023
Simplified by Elia Savino - 2024 for lamas purposes
GitHub: https://github.com/simone16

Description: Utilities for storing and logging error and status messages from all devices.

<a id="lamas.logger.Logger"></a>

## Logger Objects

```python
class Logger()
```

Provides logging functions for devices, platforms and unit tasks.
Logging is performed asynchronously on a dedicated thread, so minimal delay is caused by a log request.

<a id="lamas.logger.Logger.start_logging_thread"></a>

#### start\_logging\_thread

```python
@classmethod
def start_logging_thread(cls,
                         platform: str = "Platform",
                         use_console: bool = False) -> None
```

Starts the logging machinery.
Logging happens on a dedicated thread and calls are placed on a queue. This thread needs to be started once
(here).

@param platform: str
Name of the platform this logger is tied to.
@param use_console: bool = False
If false, no messages will be sent to console, no matter what.
Note: this is currently causing issues and should not be set.
@return: None

<a id="lamas.logger.Logger.stop_logging_thread"></a>

#### stop\_logging\_thread

```python
@classmethod
def stop_logging_thread(cls) -> None
```

Stops the logger thread.
Ensures all remaining requests are handled.

<a id="lamas.logger.Logger.log_message"></a>

#### log\_message

```python
@classmethod
def log_message(cls,
                message: str | Exception,
                origin: str = "unknown",
                **kwargs) -> None
```

Print messages to the console and store logs.
All messages will appear on console and on the common log file.
If an origin is specified in the kwargs, the message is stored in a dedicated log as well.
If multiple origins are given, the message will be stored in all logs, but the first origin will
be used as 'primary' to craft the message.

@param message: str | Exception
The message to be given to the user or an exception (will not be raised here).
@param origin: str
Name of originating device or task.
@param kwargs:
See below.
@keyword subfolder: str
The sub-folder that the origin should appear in (e.g.: 'devices' or 'tasks').
Default: no sub-folder
Note: if the origin is known, this will be ignored and stored value will be used instead.
@keyword level:str
Warning level of the message: influences how it is highlighted and where it is displayed.
Accepted values:
- 'error'
- 'warning'
- 'ok'
- 'none'
Default: 'none'
@keyword indent: str
Messages can be indented to appear to be related to a previous message,
for example the start of a routine.
These options can be specified:
- 'enter' (Increase indentation level by one).
- 'continue' (Use current indentation level).
- 'exit' (Decrease indentation level by one).
- 'reset' (Exit all indentation levels back to 0).
Default: 'continue'
@keyword decorate: bool
If False the message will be printed without additional formatting/info.
Default: True
@keyword hide: bool
If true, the message is logged to files only and not shown on console.
Default: False
@keyword mute_logs: bool
If true, the message is not logged to the files.
Default: False

<a id="lamas.templates"></a>

# lamas.templates

Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: logger and base logged class for all the functions and tasks in the project

<a id="lamas.templates.Logger"></a>

## Logger Objects

```python
class Logger()
```

Container for logging-relate methods.

<a id="lamas.templates.Logger.log_message"></a>

#### log\_message

```python
@classmethod
def log_message(cls, message: str, **kwargs) -> None
```

Print messages to the console and store logs.

All messages will appear on console and on the common log file.
If an origin is specified in the kwargs, the message is stored in a dedicated log as well.

**Arguments**:

- `message`: str
The message to be given to the user.
- `kwargs`: See below.

<a id="lamas.templates.BaseLoggedClass"></a>

## BaseLoggedClass Objects

```python
class BaseLoggedClass()
```

default class to build all functions in the project, has logger always available, has a default error handling
strategy.

<a id="lamas.templates.BaseLoggedClass.log_mssg"></a>

#### log\_mssg

```python
@classmethod
def log_mssg(cls, message: str, level: str = "warning") -> None
```

Log a message.

**Arguments**:

- `message`: The message to log.
- `level`: The level of the message. allowed: 'error', 'warning', 'ok', 'none'

<a id="lamas.templates.BaseLoggedClass.assertion_method"></a>

#### assertion\_method

```python
@classmethod
def assertion_method(cls,
                     obj: any,
                     condition: callable,
                     message: str = f"Assertion failed")
```

Assert that the object follows the condition, if not logs and raises error, this is done to properly log everything

**Arguments**:

- `obj`: any, object to check
- `condition`: callable, condition to check in the form lambda x: c > 0
- `message`: str, message to log if the assertion fails, if none base message is used

<a id="lamas.guanaco"></a>

# lamas.guanaco

Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: Guanacos are the buddies of the Vicunas, they live at lower altitudes,
similarly here the Guanaco class is used to generate linearity models for chemicals.
The idea is that given a set of spectra where the concentration of chemicals is known, (after being vicuna modelled)
this class should be able to pick out which peaks are from which chemical and then generate a caliration curve for each chemical

<a id="lamas.guanaco.Guanaco"></a>

## Guanaco Objects

```python
class Guanaco(Glama)
```

The `Guanaco` class isolates sets of peaks related to a single compound
and generates calibration curves for each compound.

**Attributes**:

- `conc_df` _pd.DataFrame_ - DataFrame containing the concentration for each compound
  in the corresponding file (or y column in the spectrum).
- `sub_df_dict` _Dict[str, pd.DataFrame]_ - Dictionary containing sub-dataframes for each compound,
  where each key is the compound name.
- `models` _Dict[str, List[VoigtPeakIdentification]]_ - Dictionary containing the Voigt models for each spectrum.
  
  Usage:
  first instantiate the class and then load the models and the concentration data
  naco = Guanaco(**kwargs) -> kwargs are passed to the superclass
  
  naco.load_

<a id="lamas.guanaco.Guanaco.__init__"></a>

#### \_\_init\_\_

```python
def __init__(**kwargs: Any) -> None
```

Initializes the Guanaco instance, passing any keyword arguments to the superclass initializer.

<a id="lamas.guanaco.Guanaco.load_models_from_dir"></a>

#### load\_models\_from\_dir

```python
def load_models_from_dir(path_to_models: str) -> None
```

Loads Voigt models from a specified directory.

This method searches for all files in the given directory that have the extension '.models',
reads each file as a CSV into a DataFrame, and creates a list of `VoigtPeakIdentification` objects
for each model. The models are stored in the `self.models` dictionary with the filename (without extension)
as the key.

**Arguments**:

- `path_to_models` _str_ - The path to the directory containing the model files.
  

**Raises**:

- `FileNotFoundError` - If the specified directory does not exist.

<a id="lamas.guanaco.Guanaco.load_models_from_class"></a>

#### load\_models\_from\_class

```python
def load_models_from_class(vicuna_class: Vicuna) -> None
```

loads the models from a vicuna class, converting them to VoigtPeakIdentification objects
and copying them to the self.models attribute

<a id="lamas.guanaco.Guanaco.load_concentrations_from_file"></a>

#### load\_concentrations\_from\_file

```python
def load_concentrations_from_file(conc_file: str) -> None
```

Loads the concentration DataFrame from a CSV file and stores it in the `conc_df` attribute.

**Arguments**:

- `conc_file` _str_ - The path to the concentration CSV file.
  

**Raises**:

- `FileNotFoundError` - If the specified concentration file does not exist.

<a id="lamas.guanaco.Guanaco.load_concentrations_from_class"></a>

#### load\_concentrations\_from\_class

```python
def load_concentrations_from_class(vicuna_class: Vicuna) -> None
```

Loads concentration data from a Vicuna class and merges it into `self.conc_df`.

If `self.conc_df` does not exist or is None, it initializes it with `vicuna_class.concentrations`.
If `self.conc_df` exists, it performs an outer merge with `vicuna_class.concentrations`
on the 'filename' column, adding any missing rows and columns with zero-fill for missing values.

Raises an error if there are conflicts in overlapping cells.

**Arguments**:

- `vicuna_class` _Vicuna_ - An instance of the Vicuna class with a `concentrations` attribute.
  

**Raises**:

- `TypeError` - If `vicuna_class.concentrations` is not a DataFrame.
- `ValueError` - If there are conflicts in overlapping cells during the merge.

<a id="lamas.guanaco.Guanaco.make_subdataframes_concentration"></a>

#### make\_subdataframes\_concentration

```python
def make_subdataframes_concentration() -> None
```

Creates a dictionary of sub-dataframes for each compound based on concentration data.

For each concentration column in `self.conc_df` (columns starting with "Conc"),
this method extracts the compound name and creates a sub-dataframe containing only
the rows where the concentration of that compound is non-zero. The sub-dataframes
are stored in `self.sub_df_dict` with the compound name as the key.

**Returns**:

  None

<a id="lamas.guanaco.Guanaco.peak_matching"></a>

#### peak\_matching

```python
def peak_matching(delta_mu: float = 0.1, **kwargs: Any) -> None
```

Identifies common peaks across spectra for each compound in `self.sub_df_dict`, organizing them as lists of matched peaks.

**Arguments**:

- `delta_mu` _float_ - The maximum difference in `mu` values for peaks to be considered matching.
- `**kwargs` - Additional arguments passed to `peaks_same` for peak similarity checks.
  

**Raises**:

- `ValueError` - If models or sub-dataframes are not properly loaded.
  

**Returns**:

- `None` - Results are stored directly in `self.common_peaks`, organized by compound.

<a id="lamas.guanaco.Guanaco.fit_and_filter_peaks"></a>

#### fit\_and\_filter\_peaks

```python
def fit_and_filter_peaks(r2_threshold: float = 0.8,
                         p_value_threshold: float = 0.05) -> None
```

Performs linear fitting of amplitude and area vs. concentration for each matched peak set.
Only keeps sets of peaks with good fit quality based on R^2 or p-value criteria.

**Arguments**:

- `r2_threshold` _float_ - Minimum R^2 value to consider the fit acceptable.
- `p_value_threshold` _float_ - Maximum p-value to consider the fit significant.
  

**Returns**:

- `None` - Results are stored directly in `self.filtered_peak_fits`, organized by compound as lists of `VoigtLinearFitResult` objects.

<a id="lamas.guanaco.Guanaco.save_data"></a>

#### save\_data

```python
def save_data(base_directory: str = "filtered_peak_fits")
```

Saves the filtered peak fits data in a structured format:
- Main DataFrame with each compound's fitting summary and paths to sub-DataFrames.
- Sub-DataFrames for each compound with peak fitting details.
- Separate files for each peak set in each compound's directory.

**Arguments**:

- `base_directory` _str_ - Directory to save the data files. Default is "filtered_peak_fits".
  

**Returns**:

  None

<a id="lamas.guanaco.Guanaco.load_fits"></a>

#### load\_fits

```python
def load_fits(base_directory: str = "filtered_peak_fits")
```

Loads the filtered peak fits data saved in a structured format:
- Loads the main summary DataFrame to access each compound's fitting summary and paths to sub-DataFrames.
- Loads each compound's sub-DataFrame with fit results.
- Reconstructs each peak set from separate files in each compound's directory.

**Arguments**:

- `base_directory` _str_ - Directory where the data files are saved. Default is "filtered_peak_fits".
  

**Returns**:

  None

<a id="lamas.utils"></a>

# lamas.utils

Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

<a id="lamas.utils.PeakIdentification"></a>

## PeakIdentification Objects

```python
class PeakIdentification()
```

Peak identification is a quick way to save and snapshot
a specific peak in a spectrum. This is done by knowing which
column and which x values the peak is represented by.

<a id="lamas.utils.PeakIdentification.__init__"></a>

#### \_\_init\_\_

```python
def __init__(min_x: float, max_x: float, y_col: str)
```

contains important information about the peak
min_x: float: the minimum x value of the peak
max_x: float: the maximum x value of the peak
y_col: str: the column name that contains the y values

<a id="lamas.utils.VoigtPeakIdentification"></a>

## VoigtPeakIdentification Objects

```python
class VoigtPeakIdentification(PeakIdentification)
```

This class is to keep track and identify a voigt peak, it stores the parameters
of the peak, can be created from a df row and can return a df row.

<a id="lamas.utils.VoigtPeakIdentification.__init__"></a>

#### \_\_init\_\_

```python
def __init__(df_row: pd.Series | None = None,
             y_col: str | None = None,
             min_x: float = 0.0,
             max_x: float = 0.0,
             mu: float = 0.0,
             sigma: float = 0.0,
             gamma: float = 0.0,
             amplitude: float = 0.0,
             area: float = 0.0)
```

**Arguments**:

- `df_row`: pd.Series: the row from a dataframe that contains the peak information
- `y_col`: str: the column name that contains the y values
- `min_x`: float: the minimum x value of the peak
- `max_x`: float: the maximum x value of the peak
- `mu`: float: the center of the peak
- `sigma`: float: the standard deviation of the peak
- `gamma`: float: the gamma value of the peak
- `amplitude`: float: the amplitude of the peak
Remember that a voigt peak has the formula:
    Amplitude * (exp(-((x - mu) ** 2) / (2 * sigma**2))*gamma / (np.pi * ((x - mu) ** 2 + gamma**2)))

<a id="lamas.utils.VoigtPeakIdentification.peak_to_df_row"></a>

#### peak\_to\_df\_row

```python
@property
def peak_to_df_row()
```

returns the peak as a df row

<a id="lamas.utils.VoigtLinearFitResult"></a>

## VoigtLinearFitResult Objects

```python
class VoigtLinearFitResult()
```

This class stores the linear fitting results of a Voigt peak across multiple spectra,
along with references to the original peak objects used for the fitting.

It includes the linear fit parameters for both amplitude and area with respect to concentration,
the relevant statistics, and the list of `VoigtPeakIdentification` objects that matched across spectra.

<a id="lamas.utils.VoigtLinearFitResult.__init__"></a>

#### \_\_init\_\_

```python
def __init__(peak_set: list[VoigtPeakIdentification], slope_amplitude: float,
             intercept_amplitude: float, r_value_amplitude: float,
             p_value_amplitude: float, slope_area: float,
             intercept_area: float, r_value_area: float, p_value_area: float)
```

**Arguments**:

- `peak_set`: list[VoigtPeakIdentification]: The matched set of Voigt peaks across spectra.
- `slope_amplitude`: float: Slope of the linear fit for amplitude vs. concentration.
- `intercept_amplitude`: float: Intercept of the linear fit for amplitude vs. concentration.
- `r_value_amplitude`: float: Correlation coefficient for amplitude fit.
- `p_value_amplitude`: float: p-value for amplitude fit.
- `slope_area`: float: Slope of the linear fit for area vs. concentration.
- `intercept_area`: float: Intercept of the linear fit for area vs. concentration.
- `r_value_area`: float: Correlation coefficient for area fit.
- `p_value_area`: float: p-value for area fit.

<a id="lamas.utils.VoigtLinearFitResult.fit_summary"></a>

#### fit\_summary

```python
@property
def fit_summary()
```

Returns a summary of the fit results as a dictionary.

**Returns**:

dict: Summary of fitting results.

<a id="lamas.utils.VoigtLinearFitResult.fit_to_df_row"></a>

#### fit\_to\_df\_row

```python
@property
def fit_to_df_row()
```

Converts the fit results to a DataFrame row with references to the original peaks.

**Returns**:

pd.Series: A row containing the fit results and peak references.

<a id="lamas.utils.VoigtLinearFitResult.check_similarity_with_peaks"></a>

#### check\_similarity\_with\_peaks

```python
def check_similarity_with_peaks(
        other_peak: VoigtPeakIdentification, similarity_function: Callable[
            [VoigtPeakIdentification, VoigtPeakIdentification], bool],
        **kwargs) -> bool
```

Checks similarity with another peak by comparing each peak in `self.peak_set` using a specified similarity function.

**Arguments**:

- `other_peak`: VoigtPeakIdentification: The peak to compare with each peak in `self.peak_set`.
- `similarity_function`: Callable: A function that checks if two peaks are similar (e.g., `peaks_same` from `glama`).
- `kwargs`: Additional keyword arguments to pass to `similarity_function`.

**Returns**:

bool: True if all peaks in `self.peak_set` are similar to `other_peak` based on the similarity function, False otherwise.

<a id="lamas.lama"></a>

# lamas.lama

Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: The `Lama` class is the last piece of the puzzle. It predicts the concentrations of known compounds in a new spectrum
using models built from previous data. It also identifies newly emerged peaks that cannot be assigned to known compounds.

Given a new spectrum and the models (linear fit results) from a `Guanaco` instance, this class can:
- Predict concentrations of known compounds present in the new spectrum.
- Identify unmatched peaks that may indicate unknown compounds or anomalies.

<a id="lamas.lama.Lama"></a>

## Lama Objects

```python
class Lama(Glama)
```

The `Lama` class predicts concentrations of known compounds in a new spectrum
and identifies any unassigned peaks.

**Attributes**:

- `linear_fit_results` _Dict[str, List[VoigtLinearFitResult]]_ - Linear fit results for known compounds.

<a id="lamas.lama.Lama.__init__"></a>

#### \_\_init\_\_

```python
def __init__(**kwargs: Any) -> None
```

Initializes the `Lama` instance, passing any keyword arguments to the superclass initializer.

<a id="lamas.lama.Lama.absorb_fits_from_class"></a>

#### absorb\_fits\_from\_class

```python
def absorb_fits_from_class(guanaco_class: Guanaco) -> None
```

Absorbs the linear fit results from a `Guanaco` class instance and stores them in the `Lama` instance.

**Arguments**:

- `guanaco_class` _Guanaco_ - The `Guanaco` class instance containing the linear fit results.
  

**Raises**:

- `AttributeError` - If the `Guanaco` class does not have the linear fit results.

<a id="lamas.lama.Lama.predict_concentrations"></a>

#### predict\_concentrations

```python
def predict_concentrations(new_spectrum: List[VoigtPeakIdentification],
                           known_compounds: Set[str],
                           **kwargs: Any) -> Dict[str, Any]
```

Predicts the concentrations of known compounds in a new spectrum and identifies any unassigned peaks.

**Arguments**:

- `new_spectrum` _List[VoigtPeakIdentification]_ - A list of Voigt peaks from the new spectrum.
- `known_compounds` _Set[str]_ - Set of compounds expected in the new spectrum.
- `**kwargs` - Additional keyword arguments passed to the `peaks_same` method for peak similarity checks.
  

**Raises**:

- `ValueError` - If linear fit results are not loaded.
  

**Returns**:

  Dict[str, Any]: Dictionary containing concentration predictions and unmatched peaks:
  - 'concentration_results': Dict with compound as key and dict with 'predicted_concentration',
  'error', and 'num_peaks_used' as values.
  - 'unmatched_peaks': List of `VoigtPeakIdentification` objects that could not be assigned.

<a id="lamas.alpaca"></a>

# lamas.alpaca

Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:
Alpacas are a subspecies of Llamas, and they are known for their grace and tidy appearence. This is a class that
aims at tidying up the raw Raman data by performing the preprocessing necessary to make the data ready for analysis.

<a id="lamas.alpaca.Alpaca"></a>

## Alpaca Objects

```python
class Alpaca(Glama)
```

class handles data preprocessing for both raman, and NMR spectra
(support for other types of spectra is planned)

<a id="lamas.alpaca.Alpaca.normalise_to"></a>

#### normalise\_to

```python
def normalise_to(to: float = 1000,
                 peak: float = None,
                 range_vals: Tuple[float, float] = None,
                 method: str = "area") -> None
```

Normalises the data in self.data to a specific value.

method is area: integrates by calculating the area under the curve.
method is peak: integrates by calculating the peak.

**Arguments**:

- `to`: float: value to normalise to (default 1000)
- `peak`: float: peak value to normalise against
- `range_vals`: Tuple[float,float]: range of values to normalise against
- `method`: str: method to use for normalisation (default 'area')
returns None

if peak and range_vals are not provided it will use the max value in the data and find
the range of values to normalise against
peak and range_vals are mutually exclusive and if both are provided range_vals will be used

<a id="lamas.alpaca.Alpaca.baseline"></a>

#### baseline

```python
def baseline(method: str = "cholesky", **kwargs) -> None
```

Baselines the data in self.data.

**Arguments**:

- `method`: str: method to use for baseline correction (default "cholesky",
also available:linear_fit, poly_fit, rubberband, als, wavelet)
kwargs wiki
cholesky: uses cholesky decomposition to baseline correct the data
        :param p: float (default 0.01): weight of peak term
        :param lam: float (default 10**2): weight of smoothness term
        :param niter: int (default 100): number of iterations
linear_fit: uses a linear fit to baseline correct the data
        :flat_range: Tuple[float,float] (default (None,None)): range of values to use for the linear fit
als: uses the asymmetric least squares method to baseline correct the data
        :param lam: float(default 10**6): weight of the smoothness term
        :param p: float(default 0.01): weight of the asymmetry term
        :param niter: int(default 100): number of iterations
rubberband: uses the rubberband method to baseline correct the data
        :param window: int(default 50): window size, in points
polyfit: uses polynomial fit to baseline correct the data
        :param order: int(default 3): order of the polynomial to fit
wavelet: uses wavelet transform to baseline correct the data
        :param wavelet: str(default 'db4'): wavelet to use
        :param level: int(default 1): level of wavelet transform
- `kwargs`: dict: additional arguments for the method, see above
returns None

<a id="lamas.alpaca.Alpaca.denoise"></a>

#### denoise

```python
def denoise(method: str = "fft", **kwargs) -> None
```

Applies noise reduction to the data in self.data.

**Arguments**:

- `method`: str, method to use for noise reduction (default "fft", options: moving_average, wavelet, median)
kwargs options:
    fft: Fast Fourier Transform denoising
        :param threshold: float (default 0.1), fraction of frequencies to retain (lower value means more
        aggressive denoising)

    moving_average: Applies a simple moving average filter for noise reduction
        :param window_size: int (default 5), size of the moving window for averaging

    wavelet: Wavelet transform denoising
        :param wavelet: str (default 'db1'), type of wavelet to use
        :param level: int (default 1), decomposition level for wavelet transform
        :param threshold: float (default 0.2), threshold for coefficient filtering

    median: Median filter denoising
        :param window_size: int (default 3), size of the filter window for median calculation

    savgol: Savitzky-Golay filter denoising
        :param window_size: float (default 0.05), window size as fraction of the data length
        :param order: int (default 3), order of the polynomial to fit
- `kwargs`: dict, additional arguments for the chosen method

**Returns**:

None

<a id="Tests"></a>

# Tests

<a id="Tests.unit.test_sub_scripts"></a>

# Tests.unit.test\_sub\_scripts

Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

<a id="Tests.unit.test_sub_scripts.plot_dtw_alignment"></a>

#### plot\_dtw\_alignment

```python
def plot_dtw_alignment(profile1, profile2, path, spacing=0.2, x_total=None)
```

Plots two profiles with some spacing between them and highlights the DTW alignment path.

**Arguments**:

- `profile1`: 1D numpy array of the first profile.
- `profile2`: 1D numpy array of the second profile.
- `path`: List of tuples representing the DTW alignment path.
- `spacing`: Float, the vertical space between the profiles for clarity.

<a id="Tests.unit.test_sub_scripts.plotting_function"></a>

#### plotting\_function

```python
def plotting_function(x, y, label=None, fig=None)
```

Plots data onto an existing figure or creates a new figure if none is provided.

**Arguments**:

- `fig`: Optional; matplotlib Figure object. If None, creates a new figure.
- `data`: The data to plot; expects a dictionary with 'x' and 'y' keys.
- `label`: Optional; legend label for the data being plotted.

**Returns**:

matplotlib Figure object with the plotted data.

<a id="Tests.unit.test_sub_scripts.TestBaselining"></a>

## TestBaselining Objects

```python
class TestBaselining(TestCase)
```

testing the baselining methods with the Glama class on a bit of raman data

<a id="Tests.unit.test_sub_scripts.TestBaselining.test_cholesky_baseline"></a>

#### test\_cholesky\_baseline

```python
def test_cholesky_baseline()
```

Testing the Cholesky baseline method

<a id="Tests.unit.test_sub_scripts.TestBaselining.test_linear_fit_baseline"></a>

#### test\_linear\_fit\_baseline

```python
def test_linear_fit_baseline()
```

Testing the linear fit baseline method

<a id="Tests.unit.test_sub_scripts.TestBaselining.test_als_baseline"></a>

#### test\_als\_baseline

```python
def test_als_baseline()
```

Testing the ALS baseline method

<a id="Tests.unit.test_sub_scripts.TestBaselining.test_rolling_ball_baseline"></a>

#### test\_rolling\_ball\_baseline

```python
def test_rolling_ball_baseline()
```

Testing the rolling ball baseline method

<a id="Tests.unit.test_sub_scripts.TestBaselining.test_polyfit_baseline"></a>

#### test\_polyfit\_baseline

```python
def test_polyfit_baseline()
```

Testing the polynomial fit baseline method

<a id="Tests.unit.test_sub_scripts.TestBaselining.test_wavelet_baseline"></a>

#### test\_wavelet\_baseline

```python
def test_wavelet_baseline()
```

Testing the wavelet baseline method

<a id="Tests.unit.test_sub_scripts.TestDenoising"></a>

## TestDenoising Objects

```python
class TestDenoising(TestCase)
```

testing the denoising methods with the Glama class on a bit of raman data

<a id="Tests.unit.test_sub_scripts.TestDenoising.test_savgol_filter_denoise"></a>

#### test\_savgol\_filter\_denoise

```python
def test_savgol_filter_denoise()
```

Testing the Savitzky-Golay filter denoise method with different window fractions

<a id="Tests.unit.test_sub_scripts.TestDenoising.test_median_filter_denoise"></a>

#### test\_median\_filter\_denoise

```python
def test_median_filter_denoise()
```

Testing the median filter denoise method with different window sizes

<a id="Tests.unit.test_sub_scripts.TestDenoising.test_wavelet_denoise"></a>

#### test\_wavelet\_denoise

```python
def test_wavelet_denoise()
```

Testing the wavelet denoise method with varying wavelet parameters

<a id="Tests.unit.test_sub_scripts.TestDenoising.test_moving_average_denoise"></a>

#### test\_moving\_average\_denoise

```python
def test_moving_average_denoise()
```

Testing the moving average denoise method with different window sizes

<a id="Tests.unit.test_sub_scripts.TestDenoising.test_fft_denoise"></a>

#### test\_fft\_denoise

```python
def test_fft_denoise()
```

Testing the FFT denoise method with different frequency thresholds

<a id="Tests.unit.test_sub_scripts.TestIntegrals"></a>

## TestIntegrals Objects

```python
class TestIntegrals(TestCase)
```

Testing the integral calculation methods:

<a id="Tests.unit.test_sub_scripts.TestIntegrals.test_find_peaks"></a>

#### test\_find\_peaks

```python
def test_find_peaks()
```

Testing the peak finding method

<a id="Tests.unit.test_sub_scripts.TestSimilarity"></a>

## TestSimilarity Objects

```python
class TestSimilarity(TestCase)
```

Testing the methodology of peak similarities:

<a id="Tests.unit"></a>

# Tests.unit

Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

<a id="Tests.unit.test_alpaca"></a>

# Tests.unit.test\_alpaca

Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

<a id="Tests.unit.test_alpaca.TestAlpaca"></a>

## TestAlpaca Objects

```python
class TestAlpaca(TestCase)
```

Test case for alpaca class, since a lot of stuff has been tested in te glama
class we will just test the normalise method and the actual performance on data

<a id="Tests.unit.test_alpaca.TestAlpaca.setUpClass"></a>

#### setUpClass

```python
@classmethod
def setUpClass(cls)
```

Set up the class and load the data

<a id="Tests.unit.test_alpaca.TestAlpaca.test_preprocess"></a>

#### test\_preprocess

```python
def test_preprocess()
```

Test the denoise method

<a id="Tests.unit.test_alpaca.TestAlpaca.test_normalise"></a>

#### test\_normalise

```python
def test_normalise()
```

Test the normalise method

<a id="Tests.unit.test_guanaco"></a>

# Tests.unit.test\_guanaco

Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

<a id="Tests.unit.__init__ 2"></a>

# Tests.unit.\_\_init\_\_ 2

<a id="Tests.unit.test_glama"></a>

# Tests.unit.test\_glama

Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

<a id="Tests.unit.test_vicuna"></a>

# Tests.unit.test\_vicuna

Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

<a id="Tests.unit.test_vicuna.TestVicuna"></a>

## TestVicuna Objects

```python
class TestVicuna(TestCase)
```

Testing case for vicuna, the goal here is to test how well
the class is able to model a cleaned spectrum.

<a id="Tests.unit.test_vicuna.TestVicuna.setUpClass"></a>

#### setUpClass

```python
@classmethod
def setUpClass(cls)
```

load cleaned data and start logger

<a id="Tests.unit.test_lama"></a>

# Tests.unit.test\_lama

Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:


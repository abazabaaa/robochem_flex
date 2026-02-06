"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from typing import List, Any, Optional, Dict

import pandas as pd
import pandas.api.types as ptypes
import torch
from torch import nn as nn

from robrains.base_classes import BaseParamClass
from robrains.parameter_backends.physicalparameter import PhysicalParameter
from robrains.parameter_backends.featurization_functions.buchwald_ligand import (
    buchwald_translation_discrete,
    buchwald_backtranslation_discrete,
)


class ML_parameter(BaseParamClass):
    """This is kind of the inverse of the chemical class, it represents all the parameters available to the ML
    model, it mostly acts like a dictionary with some added functionalities. The ML parameters will then be the ones
    passed to ML as effectively the ML does need not know the chemicals and what they are, at least for now.

    Members:
        -tag str: tag value containing "ml_parameter" to search all ml params in the session
        -min_value float:minimum value set by the user for ML, must be >= default_min if default_min exists
        -max_value float:maximum value set by the user for ML, must be <= default_max if default_max exists
        -phy_chem str:Boolean type string, either Physical or Chemical, refers to what sort of parameter it is
        -discrete_type str: defines what kind of data goes in discrete, str for chemicals, float or int for values
        -discrete list: list of values that compose the discrete values of a paramter, can contain int, float or str
        -default_min float: minimum value that the paramteter can have (from Chemical class or Physical param class)
        -default_max float: maximum value that the paramteter can have (from Chemical class or Physical param class)

    Methods:
        -__init__(name, unit, phy_chem, discrete_type, default_min, default_max):
            initialises the class, requires the args above
        -from_json(json_dict):
            classmethod, initialises the class from json_file dictionary, used in loading and saving the class
        -update_values(phy_chem,discrete,discrete_type,min_value,max_value):
            updates the values, if the user so chooses, not all arguments are necessary, may also be omitted. Performs
            checks on the values before updating
        -validate_and_update(discrete)
            updating the list of discretes is a bit more tricky, so we use this function, it checks for list item types
            and makes sure they are within min_max bounds (this is necessary as streamlit doesn't allow to do this at
            GUI level)
        -discrete_str:
            property, returns the discrete list as a comma serparated string, we do this for display purposes.
        - __bool__(): returns true if the class has a name and the discrete list and discrete type match. it's to make
        sure the class is not overwritten if not necessary.
        - ml_preprocessing: function prepares the data for use in ML models, for continuous data shifts the bounds to
            -1,1 with mean 0, for discrete data it one hot encodes the data.
        -ml_backtranslate: function backtranslates the ML requests to the original values (so the machine can read them)
            for floats it backtranslates the values to the original range, for discrete values it backtranslates the one
            hot encode to the original value.


    """

    tag = "ML_parameter"
    min_value = None
    max_value = None
    fidelity_feature = False
    task_feature = False
    high_categorical = False
    const_value = None

    def __init__(
        self,
        name: str,
        parent_parameter: Optional[PhysicalParameter] = None,
        discrete_type: str = "str",
        constant_valued: bool = False,
    ) -> None:
        """
        Initialize an ML parameter for optimization.

        :param name: Unique name of the ML parameter.
        :param parent_parameter: Optional PhysicalParameter providing default bounds and unit.
        :param discrete_type: Expected type for discrete embeddings (e.g. 'str', 'int').
        :param constant_valued: If True, this parameter remains constant (no variation).
        """
        # Log start of initialization
        self.log_mssg(f"Initializing ML parameter '{name}'")

        # Core attributes
        self.name: str = name
        self.constant_valued: bool = constant_valued

        # Derive unit from parent or name
        if parent_parameter is not None and hasattr(parent_parameter, "unit"):
            self.unit: str = parent_parameter.unit
        elif "limiting reagent" in name.lower():
            self.unit = "mM"
        else:
            self.unit = "eq"

        # Physical vs. chemical
        self.phy_chem: str = (
            "Physical"
            if isinstance(parent_parameter, PhysicalParameter)
            else "Chemical"
        )

        # Bounds defaults
        self.default_min: Optional[float] = getattr(parent_parameter, "min_value", None)
        self.default_max: Optional[float] = getattr(parent_parameter, "max_value", None)

        # Discrete embedding setup
        self.discrete: List[Any] = []
        self.discrete_type: str = discrete_type

        # Flags
        self.just_created: bool = True

        # Custom translators/back-translators (optional overrides)
        self.continuous_translator_custom = None
        self.continuous_backtranslator_custom = None
        self.discrete_translator_custom = None
        self.discrete_backtranslator_custom = None
        self.fidelity_translator_custom = None
        self.fidelity_backtranslator_custom = None
        self.task_translator_custom = None
        self.task_backtranslator_custom = None

        # Final log
        self.log_mssg(
            f"ML parameter '{self.name}' initialized: phy_chem={self.phy_chem}, "
            f"unit={self.unit}, constant_valued={self.constant_valued}",
            level="ok",
        )

    @classmethod
    def from_json(cls, json_data: dict):
        """initialises the class from a json dictionary
        :json_data: dict, dictionary of data read from json,
        :returns instance: instance of the initialised class"""
        try:
            instance = cls(
                name=json_data["name"],
                discrete_type=json_data["discrete_type"],
            )
            instance.default_min = json_data.get("default_min", None)
            instance.default_max = json_data.get("default_max", None)
            instance.constant_valued = json_data.get("constant_valued", False)
            instance.update_values(
                phy_chem=json_data.get("phy_chem", None),
                role=json_data.get("role", None),
                discrete=json_data.get("discrete", None),
                min_value=json_data.get("min_value", None),
                max_value=json_data.get("max_value", None),
                unit=json_data.get("unit", None),
                prices=json_data.get("prices", None),
                const_value=json_data.get("const_value", None),
            )
            instance.fidelity_feature = json_data.get("fidelity_feature", False)
            instance.fidelity_value = json_data.get("fidelity_value", None)
            instance.task_feature = json_data.get("task_feature", False)
            instance.task_value = json_data.get("task_value", None)
            instance.high_categorical = json_data.get("high_categorical", False)
            instance.discrete_continuous = json_data.get(
                "discrete_continuous", "Continuous"
            )
            instance.linked_physical_features = json_data.get(
                "linked_physical_features", None
            )
            return instance
        except Exception as e:
            cls.log_mssg(
                cls, f"Error initialising ML parameter from json: {e}", level="error"
            )

    def update_values(
        self,
        phy_chem: Optional[str] = None,
        role: Optional[str] = None,
        discrete: Optional[List[Any]] = None,
        discrete_type: Optional[str] = None,
        min_value: Optional[float] = None,
        max_value: Optional[float] = None,
        unit: Optional[str] = None,
        fidelity_value: Optional[Any] = None,
        task_value: Optional[Any] = None,
        prices: Optional[Dict[Any, float]] = None,
        high_categorical: Optional[bool] = None,
        const_value: Optional[float] = None,
    ) -> None:
        """
        Update multiple attributes of the MLParameter in one call.

        :param phy_chem: 'Physical' or 'Chemical'.
        :param role: Role or category of the parameter.
        :param discrete: List of allowable discrete values.
        :param discrete_type: Data type for discrete values ('str', 'int', etc.).
        :param min_value: Minimum allowable value (must be >= default_min if set).
        :param max_value: Maximum allowable value (must be <= default_max if set).
        :param unit: Measurement unit string.
        :param fidelity_value: A discrete value to designate fidelity.
        :param task_value: A discrete value to designate task.
        :param prices: Mapping from discrete values to unit prices.
        :param high_categorical: If True, treat discrete as high-cardinality.
        :param const_value: A fixed value for constant parameters.
        """
        self.log_mssg(f"Updating {self.name} attributes")

        if phy_chem is not None:
            self.phy_chem = phy_chem
        if role is not None:
            self.role = role
        if discrete is not None:
            self.discrete = discrete
        if discrete_type is not None:
            self.discrete_type = discrete_type
        if min_value is not None and (
            self.default_min is None or min_value >= self.default_min
        ):
            self.min_value = min_value
        if max_value is not None and (
            self.default_max is None or max_value <= self.default_max
        ):
            self.max_value = max_value
        if unit is not None:
            self.unit = unit
        if fidelity_value not in (None, "") and fidelity_value in getattr(
            self, "discrete", []
        ):
            self.fidelity_value = fidelity_value
        if task_value not in (None, "") and task_value in getattr(self, "discrete", []):
            self.task_value = task_value
        if prices is not None:
            self.prices = prices
        if high_categorical is not None:
            self.high_categorical = high_categorical
        if const_value is not None:
            self.const_value = const_value

        self.log_mssg("Attribute update complete", level="ok")
        self.log_mssg(f"New values: {self}", indent=1)

    @property
    def discrete_str(self):
        """returns the discrete values as a string"""
        return ",".join([str(i) for i in self.discrete])

    @classmethod
    def from_df_column(cls, df: pd.DataFrame, columns: List[str]) -> "ML_Parameter":
        """
        Construct an MLParameter from one or two DataFrame columns.

        Single-column (Physical):
          - Float column: continuous unless <80% unique → discrete list.
          - Int/string column: always discrete; ints record min/max, strings default to 0/1.
        Two-column (Chemical):
          - First column is discrete categories; second '<name>_conc' is continuous.
          - If concentration has a single unique value, mark as constant.

        :param df: DataFrame containing the raw data.
        :param columns: List of 1 or 2 column names.
        :return: Configured MLParameter instance.
        :raises ValueError: For invalid column counts or naming.
        """
        cls.log_mssg(f"from_df_column called with columns={columns}")

        if len(columns) == 1:
            col = columns[0]
            cls.log_mssg(f"Creating PhysicalParameter from column '{col}'")

            series = df[col]
            uniques = series.dropna().unique()
            total = len(series)

            constant_valued = len(uniques) == 1
            const_value: Optional[Any] = uniques[0] if constant_valued else None
            discrete_vals: List[Any] = []
            min_val: Optional[float] = None
            max_val: Optional[float] = None

            if ptypes.is_float_dtype(series):
                if len(uniques) < 0.8 * total:
                    discrete_vals = uniques.tolist()
                min_val = float(series.min())
                max_val = float(series.max())

            elif ptypes.is_integer_dtype(series) or ptypes.is_string_dtype(series):
                discrete_vals = uniques.tolist()
                if ptypes.is_integer_dtype(series):
                    min_val = float(series.min())
                    max_val = float(series.max())
                else:
                    min_val = 0.0
                    max_val = 1.0

            else:
                msg = f"Unsupported dtype for column '{col}'"
                cls.log_mssg(msg, level="error")
                raise ValueError(msg)

            phys = PhysicalParameter(
                name=col, min_value=min_val, max_value=max_val, unit=""
            )
            mlp = cls(
                name=f"{col}_ML", parent_parameter=phys, constant_valued=constant_valued
            )

        elif len(columns) == 2:
            name_col, conc_col = columns
            cls.log_mssg(f"Creating ChemicalParameter from columns {columns}")
            if conc_col != f"{name_col}_conc":
                msg = (
                    f"Expected concentration column '{name_col}_conc', got '{conc_col}'"
                )
                cls.log_mssg(msg, level="error")
                raise ValueError(msg)

            series_dis = df[name_col].dropna()
            series_conc = df[conc_col]
            uniques_dis = series_dis.unique().tolist()
            uniques_conc = series_conc.dropna().unique()

            constant_valued = len(uniques_conc) == 1
            const_value: Optional[Any] = uniques_conc[0] if constant_valued else None
            min_val = None if constant_valued else float(series_conc.min())
            max_val = None if constant_valued else float(series_conc.max())

            mlp = cls(name=f"{name_col}_ML", constant_valued=constant_valued)
            discrete_vals = uniques_dis

        else:
            msg = f"from_df_column requires 1 or 2 columns, got {len(columns)}"
            cls.log_mssg(msg, level="error")
            raise ValueError(msg)

        # Finalize configuration
        mlp.update_values(
            discrete=discrete_vals,
            min_value=min_val,
            max_value=max_val,
            const_value=const_value,
        )
        cls.log_mssg(f"MLParameter '{mlp.name}' created successfully", level="ok")
        cls.log_mssg(mlp, level="none", indent=1)
        return mlp

    def validate_and_update(self, discrete: list = None):
        """validates tha the discrete values are between the min and max values and updates the values
        :param discrete: list: the list of discrete values
        """
        if discrete is None:
            return
        try:
            # parse discrete values to list of discrete_type
            if self.discrete_type == "float":
                self.discrete = [float(i) for i in discrete.split(",")]
                if (
                    min(self.discrete) < self.min_value
                    or max(self.discrete) > self.max_value
                ):
                    self.log_mssg(
                        f"Discrete values not in range {self.min_value} - {self.max_value}",
                        level="warning",
                    )
                    return
                self.discrete = [
                    str(i) for i in discrete.split(",")
                ]  # typecasting to string just to be sure
            elif self.discrete_type == "str":
                self.discrete = [str(i) for i in discrete.split(",")]
            elif self.discrete_type == "int":
                self.discrete = [int(i) for i in discrete.split(",")]
                if (
                    min(self.discrete) < self.default_min
                    or max(self.discrete) > self.default_max
                ):
                    self.log_mssg(
                        f"Discrete values not in range {self.default_min} - {self.default_max}",
                        level="warning",
                    )
                    return
            if self.phy_chem == "Physical":
                self.min_value = None
                self.max_value = None
        except Exception as e:
            self.log_mssg(f"Could not parse discrete values: {e}", level="warning")

    def _continuous_preprocessing(self, value, a, b):
        """Transforms a continuous value to a value between 0 and 1 given the min and max
        # after transformation the value is truncated to 4 decimal places
        :param value: float: the value to be transformed
        :param a: float: the minimum value of the range
        :param b: float: the maximum value of the range
        """
        # type checking and range checking:
        if not isinstance(value, (int, float)):
            self.log_mssg(f"{self}: Value {value} is not a number", level="error")
            raise ValueError("Value is not a number")
        if value < a or value > b:
            self.log_mssg(f"{self}: Value {value} is not in range", level="error")
            raise ValueError("Value is not in range")
        value = (value - a) / (b - a)
        return round(value, 5)

    def _continuous_backtranslation(self, value, a, b):
        """Backtranslates a continuous value from 0 to 1 to the original range
        the value is truncated to 4 decimal places
        :param value: float: the value to be backtranslated
        :param a: float: the minimum value of the range
        :param b: float: the maximum value of the range
        """
        # type checking and range checking:
        if not isinstance(value, (int, float)):
            self.log_mssg(f"{self}: Value {value} is not a number", level="error")
            raise ValueError("Value is not a number")
        if value < -1 or value > 1:
            self.log_mssg(f"{self}: Value {value} is not in range", level="error")
            raise ValueError("Value is not in range")
        value = round(value, 5)
        return value * (b - a) + a

    def _continuous_variance_preprocessing(self, value, a, b):
        """Transforms the variance of the continuous value to the same range as the value

        :param value: float: the value to be transformed
        :param a: float: the minimum value of the range
        :param b: float: the maximum value of the range


        """
        # type checking and range checking:
        if not isinstance(value, (int, float)):
            self.log_mssg(f"{self}: Value {value} is not a number", level="error")
            raise ValueError("Value is not a number")
        if value < a or value > b:
            self.log_mssg(f"{self}: Value {value} is not in range", level="error")
            raise ValueError("Value is not in range")
        value = value * (1 / b - a) ^ 2
        return round(value, 5)

    def _continuous_variance_backtranslation(self, value, a, b):
        """
        Backtranslates the variance of the continuous value to the original range

        :param value: float: the value to be backtranslated
        :param a: float: the minimum value of the range
        :param b: float: the maximum value of the range

        """
        # type checking and range checking:
        if not isinstance(value, (int, float)):
            self.log_mssg(f"{self}: Value {value} is not a number", level="error")
            raise ValueError("Value is not a number")
        if value < -1 or value > 1:
            self.log_mssg(f"{self}: Value {value} is not in range", level="error")
            raise ValueError("Value is not in range")
        value = round(value, 5)
        return value * (1 / (b - a)) ** -2

    def _discrete_preprocessing(self, value, discrete_values):
        """Encodes the discrete value using an embedding and scales it between 0 and 1
        :param value: the human-readable value to be converted to an embedding
        :param discrete_values: the list of discrete values
        """
        # Ensure value is valid
        if value not in discrete_values:
            self.log_mssg(
                f"{self}: Value {value} is not in discrete values", level="error"
            )
            raise ValueError("Value is not in discrete values")

        # Find the index of the value in the discrete_values list
        index = discrete_values.index(value)

        # Get the embedding for this index (constant embeddings)
        embedding = self.embedding_layer(torch.tensor(index))

        # Apply min-max scaling to the embedding to scale it between 0 and 1
        min_val, _ = torch.min(self.embedding_layer.weight.data, dim=0)
        max_val, _ = torch.max(self.embedding_layer.weight.data, dim=0)
        scaled_embedding = (embedding - min_val) / (
            max_val - min_val + 1e-8
        )  # Min-max scaling
        scaled_embedding = scaled_embedding.detach().to(torch.float64)
        # Return the scaled embedding (rounded to avoid float precision issues)
        return torch.round(scaled_embedding, decimals=5)

    def _discrete_backtranslation(self, value, discrete_values):
        """Backtranslates the scaled embedding value to the original human-readable discrete value
        :param value: the scaled embedding value
        :param discrete_values: the list of discrete values
        """

        # Apply inverse min-max scaling to find the original embedding value
        min_val, _ = torch.min(self.embedding_layer.weight.data, dim=0)
        max_val, _ = torch.max(self.embedding_layer.weight.data, dim=0)

        # Rescale the value back to the original embedding space
        original_embedding = value * (max_val - min_val + 1e-8) + min_val

        # Find the closest embedding to the scaled value
        embeddings = self.embedding_layer.weight.data
        distances = torch.norm(
            embeddings - original_embedding, dim=1
        )  # Euclidean distance
        closest_index = torch.argmin(distances)

        # Return the human-readable discrete value
        return discrete_values[closest_index]

    def _translation_fidelity(self, value, fidelity_feature):
        """translates the value to the fidelity feature"""
        if value == fidelity_feature:
            return 1
        else:
            return 0

    def _back_translation_fidelity(self, value, fidelity_feature):
        """backtranslates the value to the fidelity feature"""
        if value == 1:
            return fidelity_feature
        else:
            return 0

    def _translation_task(self, value, discrete_values):
        """assigns a integer value to the task feature based on the discrete values"""
        return discrete_values.index(value)

    def _back_translation_task(self, value, discrete_values):
        """backtranslates the integer value to the task feature"""
        return discrete_values[value]

    def _high_categorical_translation(self, value, discrete_values):
        """assigns an integer value to the categorical feature based on the list of discrete values"""
        return discrete_values.index(value)

    def _high_categorical_backtranslation(self, value, discrete_values):
        """backtranslates the integer value to discrete value"""
        return discrete_values[value]

    def generate_translators(self) -> None:
        """
        Create translation and back-translation callables for this parameter:

          - Continuous: if min/max defined and not constant, map [min,max]↔[0,1].
          - Discrete: handle single-value, fidelity, task, or multi-value embeddings.
          - Fidelity: translate discrete fidelity into binary or scaled value.
          - Task: translate discrete task into ordinal index.
          - High-cardinality vs. low-cardinality embedding sizes.

        After generation, associated attributes:
          - translation_continuous / back_translation_continuous
          - translation_discrete / back_translation_discrete
          - translation_fidelity / back_translation_fidelity
          - translation_task / back_translation_task
          - discrete_embed_dim
        """
        self.log_mssg(f"Generating translators for parameter: {self.name}")

        # Continuous translator
        if (
            self.min_value is not None
            and self.max_value is not None
            and not self.constant_valued
        ):
            min_v, max_v = self.min_value, self.max_value
            self.translation_continuous = (
                self.continuous_translator_custom
                if self.continuous_translator_custom
                else lambda x: self._continuous_preprocessing(x, min_v, max_v)
            )
            self.back_translation_continuous = (
                self.continuous_backtranslator_custom
                if self.continuous_backtranslator_custom
                else lambda x: self._continuous_backtranslation(x, min_v, max_v)
            )
            self.log_mssg("Continuous translators generated", level="ok", indent=1)
        else:
            self.log_mssg(
                "Skipping continuous translator (min/max missing or constant)",
                level="warning",
            )

        # Discrete vs. fidelity/task
        self.discrete_single_value = False
        n_discrete = len(self.discrete or [])
        has_fid = bool(
            getattr(self, "fidelity_feature", False)
            and hasattr(self, "fidelity_value")
            and self.fidelity_value is not None
        )
        has_task = bool(getattr(self, "task_feature", False))

        if n_discrete <= 1:
            self.discrete_single_value = True if n_discrete == 1 else False
            msg = "Single discrete value; no embedding generated"
            if has_fid or has_task:
                self.log_mssg(msg + " (invalid for fidelity/task)", level="error")
                raise ValueError("Cannot embed single-value fidelity/task parameter")
            self.log_mssg(msg, level="warning")

        elif has_fid and has_task:
            self.log_mssg(
                "Cannot handle both fidelity and task features", level="error"
            )
            raise ValueError("Ambiguous fidelity/task feature combination")

        elif has_fid:
            fid = (
                self.fidelity_value[0]
                if isinstance(self.fidelity_value, (list, tuple))
                else self.fidelity_value
            )
            self.translation_fidelity = (
                self.fidelity_translator_custom
                if self.fidelity_translator_custom
                else lambda x: self._translation_fidelity(x, fid)
            )
            self.back_translation_fidelity = (
                self.fidelity_backtranslator_custom
                if self.fidelity_backtranslator_custom
                else lambda x: self._back_translation_fidelity(x, fid)
            )
            self.log_mssg("Fidelity translators generated", level="ok", indent=1)
            self.discrete_single_value = True

        elif has_task:
            vals = self.discrete
            self.translation_task = (
                self.task_translator_custom
                if self.task_translator_custom
                else lambda x: self._translation_task(x, vals)
            )
            self.back_translation_task = (
                self.task_backtranslator_custom
                if self.task_backtranslator_custom
                else lambda x: self._back_translation_task(int(x), vals)
            )
            self.log_mssg("Task translators generated", level="ok", indent=1)
            self.discrete_single_value = True

        else:
            # Multi-value discrete embeddings
            if n_discrete < 3:
                dim = 1
            elif n_discrete < 5:
                dim = 2
            elif n_discrete < 13:
                dim = 3
            else:
                dim = 4
            self.discrete_embed_dim = (
                getattr(self, "discrete_embed_dim", dim)
                if not getattr(self, "high_categorical", False)
                else 1
            )

            # Embedding layer for low-cardinality
            if not getattr(self, "high_categorical", False):
                self.embedding_layer = nn.Embedding(
                    len(self.discrete), self.discrete_embed_dim
                )
            # Translator/back-translator functions
            self.translation_discrete = (
                self.discrete_translator_custom
                if self.discrete_translator_custom
                else lambda x: self._discrete_preprocessing(x, self.discrete)
            )
            self.back_translation_discrete = (
                self.discrete_backtranslator_custom
                if self.discrete_backtranslator_custom
                else lambda x: self._discrete_backtranslation(x, self.discrete)
            )
            self.log_mssg(
                f"Discrete translators generated (embed_dim={self.discrete_embed_dim})",
                level="ok",
                indent=1,
            )

    def __bool__(self):
        """return true if
        1)the name is set
        2)the discrete_type is "str" and the discrete type is not empty and the min_max values are set,
         or the discrete_type is "float" and the list is not empty"""

        if not self.name:
            self.log_mssg(f"{self}:Name is not set", level="warning")
            return False

        if self.phy_chem == "Chemical":
            if self.discrete == []:
                self.log_mssg(f"{self}:Discrete list is empty", level="warning")
                return False
            if self.min_value is None or self.max_value is None:
                self.log_mssg(f"{self}:Min or max value is not set", level="warning")
                return False
            if self.min_value < 0 or self.max_value < 0:
                self.log_mssg(f"{self}:Min or max value is negative", level="warning")
                return False
        elif self.phy_chem == "Physical":
            discrete_empty = len(self.discrete) == 0 or self.discrete is None
            missing_values = self.min_value is None or self.max_value is None
            defaults_missing = self.default_min is None or self.default_max is None
            if defaults_missing:
                self.log_mssg(
                    f"{self}:Default min or max values not set", level="warning"
                )
                return False
            if self.just_created == True:
                self.just_created = False
                return True
            if discrete_empty and (missing_values and not defaults_missing):
                self.log_mssg(
                    f"{self}:Physical parameter did not pass the vibe check: mv {missing_values} de {discrete_empty}",
                    level="warning",
                )
                return False
            if not discrete_empty and not missing_values:
                self.log_mssg(
                    f"{self}:Physical parameter did not pass the vibe check: mv {missing_values} de {discrete_empty}",
                    level="warning",
                )
                return False

        else:
            self.log_mssg(f"{self}:Physical or Chemical not set", level="warning")
            return False

        return True

    def __repr__(self):
        """returns a string representation of the class"""
        return (
            f"MLParameter(name={self.name}, phy_chem={self.phy_chem}, "
            f"discrete={self.discrete}, min_value={self.min_value}, "
            f"max_value={self.max_value}, unit={self.unit})"
        )

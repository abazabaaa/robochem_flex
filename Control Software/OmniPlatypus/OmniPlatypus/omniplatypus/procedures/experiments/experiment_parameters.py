"""
File: experiment_parameters.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: Parameters for experimental procedures.
"""

import copy
from typing import Any, Optional

from omniplatypus.utilities.general import format_float, conversion_factor, dict_to_str


class ExperimentalParameter:
    """
    Defines and holds information over one parameter of the experiment which is set by the GUI.
    The way this data is handled by the experiment depends entirely on its implementation.
    """

    name: str
    value: Any
    _allowed_values: list | None
    _foreign_key: str | None

    def __init__(
        self,
        name: str,
        value: Any,
        allowed_values: list | None = None,
    ):
        """
        Constructor.

        @param name: str
            The name of the parameter. Required parameters are specified by name by an experiment.
        @param value: Any
            The value to use for the parameter.
        @param allowed_values: list | None
            List-like object with all possible values which can be set for this.
            Or None to allow any value.
        """
        self.name = name
        self._allowed_values = allowed_values
        self._foreign_key = None
        self.value = value

    @property
    def allowed_values(self):
        return self._allowed_values

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, value: Any):
        if self._allowed_values is None or value in self._allowed_values:
            self._value = value
        else:
            raise ValueError(
                f"Invalid value '{value}' for parameter '{self.name}'. Allowed values: {self.allowed_values}."
            )

    def __str__(self):
        """
        String describing the parameter.
        """
        return f"{self.name}: {self.value}"

    @property
    def foreign_key(self) -> str | None:
        """
        The foreign key is what parameter (ML or other) this is linked to in the frontend.
        """
        return self._foreign_key

    @foreign_key.setter
    def foreign_key(self, value: str | None) -> None:
        """
        Setter for the foreign key.
        """
        self._foreign_key = value


class NumericalParameter(ExperimentalParameter):
    """
    Defines and holds information over one numerical parameter of the experiment.
    The way this data is handled by the experiment depends entirely on its implementation.
    """

    name: str
    value: float
    units: str
    max_value: float | None
    min_value: float | None

    def __init__(
        self,
        name: str,
        value: float,
        units: str,
        max_value: float | None = None,
        min_value: float | None = None,
    ):
        """
        Constructor.

        @param name: str
            The name of the parameter. Required parameters are specified by name by an experiment.
            In case of chemicals, the name must correspond to a compound (which in turn must have a corresponding
            concentration column on the samples dataframe).
        @param value: float
            The value to use for the parameter.
        @param units: str
            Units for the parameter. Support for different units is dependent on the experiment implementation.
            Note: units are case-sensitive.
        @param max_value: float | None = None
            Do not allow setting this parameter above this value.
        @param min_value: float | None = None
            Do not allow setting this parameter below this value.
        """
        ExperimentalParameter.__init__(self, name=name, value=value)
        self.units = units
        self.max_value = max_value
        self.min_value = min_value

    def with_units(self, units: str) -> float:
        """
        Return the value in the requested units.

        @param units: str
            The units requested.
        @return: float
            The value of the parameter converted to the requested units.
        @raise: ValueError
            If the required conversion fails.
        """
        return self.value * conversion_factor(units_from=self.units, units_to=units)

    def set_units(self, units: str, round_to: int | None = None) -> float:
        """
        Convert the stored value to the requested units and return it.
        The units are updated and an optional parameter can be used to round the resulting value.

        @param units: str
            The units requested.
        @param round_to: int
            Decimal to round the result to. The value is also stored in the rounded form.
        @return: float
            The value of the parameter converted to the requested units.
        @raise: ValueError
            If the required conversion fails.
        """
        converted = self.with_units(units)
        if round_to is not None:
            converted = round(converted, round_to)
        self.value = converted
        self.units = units
        return converted

    def __str__(self):
        """
        String describing the parameter.
        """
        return f"{self.name}: {format_float(self.value)} {self.units}"


class ChemicalParameter(NumericalParameter):
    """
    Defines and holds information over the concentration of a chemical within a run.
    The way this data is handled by the experiment depends entirely on its implementation.
    """

    name: str
    value: float
    units: str
    sampling_priority: int

    def __init__(
        self, name: str, value: float, units: str, sampling_priority: int = 1000
    ):
        """
        Constructor.

        @param name: str
            The name of the parameter must correspond to a compound (which in turn must have a corresponding
            concentration column on the samples dataframe).
        @param value: float
            The concentration (or equivalents) value for this chemical.
        @param units: str
            Units for the parameter. Support for different units is dependent on the experiment implementation.
            Note 1: exactly one parameter per run must specify its value as a concentration ('M'
            or 'mM' are supported). The rest must be given as equivalents relative to this one concentration ('eq' or
            'mol%' are supported).
            Note 2: units are case-sensitive.
        @param sampling_priority: int = 1000
            Priority in the order of sampling this specific vial when putting together the recipy.
            Different slug sampling methods may enforce this differently, but higher numbers should correspond to
            earlier sampling. Default is 1000.
        """
        NumericalParameter.__init__(self, name=name, value=value, units=units)
        self.sampling_priority = sampling_priority

    def as_concentration(self) -> bool:
        """
        Is this Chemical parameter defined as concentration of the species, or by ratio (equivalents or mol%) based on
        another species?

        @return: bool
         True if the amount is specified in terms of concentration.
        """
        return self.units.strip()[-1] == "M"

    def __str__(self) -> str:
        """
        String describing the parameter.
        """
        return NumericalParameter.__str__(self) + f" [{self.sampling_priority}]"


class RunResult:
    """
    Holds information on a completed run.
    """

    run_id: str
    result: Any
    success: bool
    parameters: list[ExperimentalParameter]
    collection_vial_id: str | None
    exception: Exception | None

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.result = None
        self.success = False
        self.parameters = []
        self.collection_vial_id = None
        self.exception = None

    def __str__(self):
        success = "" if self.success else "[FAILED]\n"
        results = (
            dict_to_str(self.result)
            if isinstance(self.result, dict)
            else str(self.result)
        )
        error = "" if self.exception is None else str(self.exception) + "\n"
        return success + error + results

    def get_param_by_name(self, name: str) -> Optional[ExperimentalParameter]:
        """
        searches the parameters if name matches (equality) one of the names of the experimental parameters
        returns the first found experimental parameter.

        :param name: the name of the parmeter we want to search

        :returns: Experimental parameter deepcopy
        """
        param = None
        for parameter in self.parameters:
            if parameter.name == name:
                param = copy.deepcopy(parameter)
                break

        return param

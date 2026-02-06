"""
File: analytics_parameters.py
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: Analytical class parameter.
"""

from typing import Any


class AnalyticalParameter:
    """
    Defines and holds info about a parameter for the analysis.
    it is separate from the other parameters as i need to have a tag
    to know which mathematics this is necessary for. (i.e. LAMA is
    fully automated, integral needs the integration bounds)

    """

    def __init__(
        self,
        name: str,
        value: Any,
        min_value: Any = None,
        max_value: Any = None,
        discrete_values: list | set | None = None,
        units: str = "",
        tag: str | None = None,
        track: bool = False,
    ):
        """
        Constructor.

        @param name: str
            The name of the parameter. Required parameters are specified by name by an experiment.
            In case of chemicals, the name must correspond to a compound (which in turn must have a corresponding
            concentration column on the samples dataframe).
        @param value: float
            The value to use for the parameter.
            Note: despite max, min and discrete settings, any value can be set here. This class is just a container.
        @param min_value: any = None
            The minimum value for a numerical parameter.
        @param max_value: any = None
            The maximum value for a numerical parameter.
        @param discrete_values: list | set | None = None
            The set of allowed values for the parameter.
        @param units: str
            Units for the parameter. Support for different units is dependent on the experiment implementation.
            Note: units are case-sensitive.
        @param tag: str
            The tag for the parameter, this is used to determine which processing method this is necessary for.
            'all' is used for parameters which are always required.
            None (default) for parameters which are meant to be set by the user.
        """
        self.name = name
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        self.discrete_values = discrete_values
        self.units = units
        self.tag = tag
        self.track = track

    def __str__(self):
        return f"{self.name}: {self.value} {self.units}"

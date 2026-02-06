"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from typing import Any, Optional, List

from robrains.base_classes import BaseParamClass


class ExpParameter(BaseParamClass):
    """This is a class to represent an experimental
    parameter from the omniplatypus side. it is used to handle not
    numerical parameters that do not have to be optimised, but the user
    may want to use to change the behaviour of the experiment.

    Members:
        - tag: str: the tag of the class
        - name: str: the name of the parameter
        - value: str: the value of the parameter
        - allowed_values: list: the list of allowed values for the parameter
    """

    tag = "experimental_parameter"
    name: str
    value: Any
    allowed_values: list[Any] | None

    def __init__(self, name: str, value: Any, allowed_values: list[Any] = None):
        """Initialises the experimental parameter object
        :param name: str: the name of the parameter
        :param value: Any: the value of the parameter
        :param allowed_values: list: the list of allowed values for the parameter
        """
        self.name = name
        self.value = value
        self.allowed_values = allowed_values

    def __bool__(self):
        """returns true if name and value are not none"""
        return bool(self.name) and bool(self.value)

    @classmethod
    def from_json(cls, json_dict: dict):
        """
        function loads the parameter from json dictionary
        :param json_dict: dictionary of data from the json file

        :returns instance: instance of the parameter class
        """
        try:
            instance = cls(
                name=json_dict["name"],
                value=json_dict["value"],
                allowed_values=json_dict.get("allowed_values", None),
            )
            return instance

        except Exception as e:
            cls.log_mssg(
                cls,
                f"Experimental Parameter could not be loaded from dict: {e}",
                level="Warning",
            )

    def update_values(
        self, name: str, value: Any, allowed_values: Optional[List[Any]] = None
    ) -> None:
        """
        Update the property name and its current and permissible values.

        :param name: The name of the property being updated.
        :param value: The new value to assign.
        :param allowed_values: Optional list of permissible values for validation.
        """
        # Log before change
        self.log_mssg(
            f"Updating values: name='{name}', value={value}, allowed_values={allowed_values}",
        )

        # Assign
        self.name = name
        self.value = value
        self.allowed_values = allowed_values or []

        # Log after successful update
        self.log_mssg(
            f"update_values completed: name='{self.name}', value={self.value}",
            level="ok",
            indent=1,
        )

    def __repr__(self):
        """returns a string representation of the class"""
        return f"ExpParameter(name={self.name}, value={self.value}, allowed_values={self.allowed_values})"

"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from robrains.base_classes import BaseParamClass


class PhysicalParameter(BaseParamClass):
    """This is a class to represent a physical parameter,
    it mostly acts like a dictionary with some added functionalities,
    such as storing the parameter name, the min and max values and the unit.
    it also stores the tag: "physical_parameter"

    Members:
        - tag: str: the tag of the class
        - name: str: the name of the parameter
        - min_value: float: the minimum value of the parameter (from config)
        - max_value: float: the maximum value of the parameter (from config)
        - unit: str: the unit of the parameter
        - style: str: the style of the parameter
        - value: float: the value of the parameter (only for constants)

    Methods:
        - __init__(name: str, min_value: float, max_value: float, unit: str): initialises the physical parameter object
        - __bool__(): returns true if name, min_value, max_value and unit are not none
        - from_json(json_dict): classmethod, allows initialisation from json data.
    """

    tag = "physical_parameter"

    def __init__(
        self,
        name: str = None,
        min_value: float = None,
        max_value: float = None,
        unit: str = None,
    ):
        """Initialises the physical parameter object
        :param name: str: the name of the parameter
        :param min_value: float: the minimum value of the parameter
        :param max_value: float: the maximum value of the parameter
        :param unit: str: the unit of the parameter
        """

        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.unit = unit
        self.style = "variable"
        self.value = None

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
                min_value=json_dict["min_value"],
                max_value=json_dict["max_value"],
                unit=json_dict["unit"],
            )
            if json_dict.get("value", None) is not None:
                instance.value = json_dict["value"]
                instance.style = json_dict["style"]

            return instance

        except Exception as e:
            cls.log_mssg(
                cls,
                f"Physical Parameter could not be loaded from dict: {e}",
                level="Warning",
            )

    def update_values(
        self, name: str, min_value: float, max_value: float, unit: str, style: str
    ):
        """adds the values to the class"""
        self.name = name
        self.min_value = min_value
        self.max_value = max_value
        self.unit = unit
        self.style = style

    def __repr__(self):
        """returns a string representation of the class"""
        return f"PhysicalParameter(name={self.name}, min_value={self.min_value}, max_value={self.max_value}, unit={self.unit})"

"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr:

"""

from typing import Optional

from robrains.base_classes import BaseParamClass


class Chemical(BaseParamClass):
    """This is a class to represent a chemical, it mostly acts like a dictionary with some added functionalities,
    such as storing the chemical name, any identifier and it's purpose on the platform.
    it also stores the tag: 'chemical'

    Members:
        - tag: str: the tag of the class
        - name: str: the name of the chemical
        - identifier: str: the identifier of the chemical (CAS number, SMILES, etc)
        - identifier_type: str: the type of the identifier (CAS number, or SMILES, etc)
        - purpose: str: the purpose of the chemical in our system (Limiting Reagent, Excess Reagent, etc)
        - available_purposes: list: the list of available purposes for the chemical
        - available_IDs: list: the list of available identifier types for the chemical
        - value: float: the value of the chemical (only for constants)
        - price: float: the price of the chemical per unit mass,
        - ml_min: float: the minimum value of the chemical in the search space (input from user) defaults to None
        - ml_max: float: the maximum value of the chemical in the search space (input from user) defaults to None
        - unit: str: the unit of the chemical, in mM, for now does nothing, but later on i may want to implement a unit
            tracking system. we'll see


    Methods:
        - __init__(name: str, identifier_type: str, identifier: str, purpose: str): initialises the chemical object
        - update_values(name: str, identifier_type: str, identifier: str, purpose: str): updates the chemical object
        - __bool__(): returns true if name, id, idtype and purpose are not none
        - from_json(json_dict): classmethod that allows to initialise the class from data saved in a json file.
            for loading and saving the class to file.
    """

    available_purposes = [
        "Limiting Reagent",
        "Excess Reagent",
        "Other Reagent",
        "Other Reagent II",
        "Catalyst",
        "Co-catalyst",
        "Buchwald Ligand",
        "Ligand",
        "Solvent",
        "Solvent II",
        "Co-Solvent",
        "Sacrificial Reagent",
        "Additive",
        "Additive II",
        "Additive III",
        "Jerry",
        "Jerry II",
        "Technique",
        "Constant",
    ]
    available_IDs = ["CAS", "SMILES", "InChI", "InChIKey", "Internal_ID"]
    value = None
    price = None
    tag = "chemical_parameter"
    unit = "mM"

    @property
    def purpose(self) -> str:
        return self._purpose

    @purpose.setter
    def purpose(self, purpose: str) -> None:
        self.assertion_method(
            purpose,
            lambda x: x in self.available_purposes,
            f"purpose must be in {self.available_purposes}",
        )
        self._purpose = purpose

    def __init__(
        self,
        name: str = None,
        identifier_type: str = None,
        identifier: str = None,
        purpose: str = None,
    ):
        """inits the class, if no data is provided it will just set values to defaults Nones"""

        self.name = name
        self.identifier = identifier
        self.identifier_type = identifier_type
        if purpose is not None:
            self.purpose = purpose
        else:
            self._purpose = None

    def update_values(
        self,
        name: str,
        identifier_type: str,
        identifier: str,
        purpose: Optional[str] = None,
    ) -> None:
        """
        Update the core attributes of the chemical object.

        :param name: Human-readable name of the chemical.
        :param identifier_type: The type of identifier (e.g. 'CAS', 'InChI').
        :param identifier: The chemical’s unique identifier string.
        :param purpose: The intended use or role of the chemical.
        """
        self.log_mssg(
            f"Updating chemical: name='{name}', id_type='{identifier_type}', "
            f"id='{identifier}', purpose='{purpose}'",
        )
        self.name = name
        self.identifier_type = identifier_type
        self.identifier = identifier
        self.value = None
        if purpose is not None:
            self.purpose = purpose
        self.log_mssg("Chemical attributes updated successfully.", level="ok")

    def __bool__(self):
        """returns true if name, id, idtype and purpose are not none"""
        # let's be more deliberate on how we check if stuff is none or "":
        if self.name and self.identifier and self.identifier_type and self.purpose:
            return True
        else:
            return False

    def __repr__(self):
        """returns a string representation of the class"""
        return f"Chemical(name={self.name}, id_type={self.identifier_type}, id={self.identifier}, purpose={self.purpose})"

    @classmethod
    def from_json(cls, json_dict: dict):
        """
        function to load the data from json file
        :param: json_dict, dict, dict of data from json dictionary.

        return instance: instance of the chemical class initialised from the json

        """
        try:
            instance = cls(
                name=json_dict["name"],
                identifier=json_dict["identifier"],
                identifier_type=json_dict["identifier_type"],
                purpose=json_dict["_purpose"],
            )
            if json_dict.get("value", None) is not None:
                instance.value = json_dict.get("value", None)
            if json_dict.get("price", None) is not None:
                instance.price = json_dict.get("price", None)
            return instance

        except Exception as e:
            cls.log_mssg(
                cls, f"Error in loading the class from json: {e}", level="warning"
            )

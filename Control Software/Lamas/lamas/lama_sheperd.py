"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

Descr: class holds metadata about the lamas and figures out parameterisation

"""

from lamas.lama import Lama
from lamas.alpaca import Alpaca
from lamas.guanaco import Guanaco
from lamas.vicuna import Vicuna


class LamaSheperd:
    structure = {
        "Alpaca": Alpaca.structure,
        "Guanaco": Guanaco.structure,
        "Vicuna": Vicuna.structure,
        "Lama": Lama.structure,
    }

    @classmethod
    def get_structure(cls):
        return cls.structure

    @classmethod
    def load_stucture(cls, structure):
        cls.structure = structure
        return cls.structure

    @classmethod
    def get_args(cls, func_name, lama_name):
        substructure = cls.structure.get(lama_name, None)
        if substructure is None:
            return None
        func_struct = substructure.get(func_name, None)
        if func_struct is None:
            return None

        return func_struct.get_parameters(as_dict=True)

"""
Author: Elia Savino
github: github.com/EliaSavino

Happy Hacking!

TEST DESCR:
"""

import unittest
from unittest.mock import patch, MagicMock
from robrains.ml_modules import DevelopmentHITL
from robrains.communication_module import ML_Platform_HITL_Development
from robrains.parameter_backends import ML_parameter

# make a dummy class to test the inheritance


class ChildClass(DevelopmentHITL, ML_Platform_HITL_Development):
    def __init__(self):
        super().__init__()


class TestChild(unittest.TestCase):
    def setUp(self):
        self.child = ChildClass()
        self.ML_A = ML_parameter()


if __name__ == "__main__":
    unittest.main()

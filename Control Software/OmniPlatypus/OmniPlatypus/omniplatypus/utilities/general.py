"""
File: general.py
Author: Simone Pilon - Noël Research Group - 2023
GitHub: https://github.com/simone16

Description: General utilities for the devices of the Omniplatypus.
"""

import re
import os.path
from typing import Any, Union, Literal, List, Callable, Dict
import datetime
from inspect import getfile
from threading import Thread
from queue import Queue, Empty
import ast
import numpy as np
import inspect
import importlib

import omniplatypus as this_module


def path_to_configuration_folder():
    # todo this implementation is temporary. The final resting place for the configuration should be outside of the module.
    return os.path.join(os.path.dirname(getfile(this_module)), "config")


def path_to_img_folder():
    return os.path.join(os.path.dirname(getfile(this_module)), "..", "..", "imgs")


def none_if_empty(x: Any) -> Any:
    if len(x) > 0:
        return x
    return None


def time_of_completion(
    minutes: float = 0.0, seconds: float = 0.0, format: str = "%H:%M"
) -> str:
    """
    Returns a string representing the time of completion of a task lasting the given number of minutes and seconds.

    @param minutes: float = 0.0
        Minutes until the task is over.
    @param seconds: float = 0.0
        Seconds until the task is over.
    @param format: str = "%H:%M"
        Format for the return value.
    @return: str
        String representation for the time of completion.
    """
    return (
        datetime.datetime.now() + datetime.timedelta(minutes=minutes, seconds=seconds)
    ).strftime(format)


def format_float(value: float) -> str:
    """
    Format numerical values in a nice, human-readable way, while preserving reasonable accuracy.

    @param value: float
        The value to format as a number.
    @return: str
        Decimal representation as a string.
    """
    if np.isnan(value):
        return "NaN"
    return f"{value:.4g}"


def dict_to_str(value: dict, indentation: int = 0) -> str:
    """
    Format dictionaries in a more readable, multi-line string.

    @param value: dict
        The dictionary to print.
    @param indentation: int
        Sets the indentation level. This is used to convert nested dictionaries recursively.
    @return: str
        A human-readable representation of the dict.
    """
    string = ""
    indentation_str = "  " * indentation
    for key, key_value in value.items():
        if isinstance(key_value, dict):
            formatted_value = dict_to_str(key_value, indentation + 1)
        elif isinstance(key_value, float):
            formatted_value = format_float(key_value)
        else:
            formatted_value = str(key_value)
        spacer = " "
        if "\n" in formatted_value:
            spacer = "\n"
        string += f"{indentation_str}{key}:{spacer}{formatted_value}\n"
    return string


_unit_prefixes = {
    "P": 1.0e15,
    "T": 1.0e12,
    "G": 1.0e9,
    "M": 1.0e6,
    "k": 1.0e3,
    "h": 1.0e2,
    "da": 1.0e1,
    "d": 1.0e-1,
    "c": 1.0e-2,
    "m": 1.0e-3,
    "u": 1.0e-6,
    "n": 1.0e-9,
    "p": 1.0e-12,
    "f": 1.0e-15,
    None: 1.0,
    "": 1.0,
}


def conversion_factor(units_from: str, units_to: str) -> float:
    """
    Calculates the required conversion factor to go from one unit to another.
    The actual unit of measurement must match, what is returned is the prefix conversion factor
    (e.g.: going from mm to cm the conversion factor is 0.1).
    Note: prefixes are case-sensitive.
    Note: the prefix 'micro' is indicated with the letter 'u', rather than the greek letter 'mu'.

    @param units_from: str
        Units of the original value.
    @param units_to: str
        Units of the target value. This value is obtained by multiplying the original value by the value returned by
        this function.
    @return: float
        Conversion factor between units_from and units_to.
    @raise: ValueError
        If the units to convert are different (beyond the prefix) or if the prefix is not supported.

    # simo, it was being annoying with mM and M so i changed the implementation a bit

    # Would be nice to add conversion to SI from Non SI and conversion between similar categories of units. buuuut that
    # would be a lot of work and I'm not sure it's worth it.
    """
    # remove whitespace from units.
    units_from = units_from.strip()
    units_to = units_to.strip()

    # handle special case 'mol%' (1/100 of equivalent)
    if units_from == "mol%":
        units_from = "ceq"
    if units_to == "mol%":
        units_to = "ceq"

    # Extract the base units (e.g., 'm' from 'mm') by comparing from the end of the strings.
    # Reverse the strings for easier comparison from the end
    reversed_from = units_from[::-1]
    reversed_to = units_to[::-1]

    # Find the point where the characters stop matching
    divergence_point = 0
    for i in range(min(len(reversed_from), len(reversed_to))):
        if reversed_from[i] != reversed_to[i]:
            break
        divergence_point += 1

    # Check if the base units match
    if divergence_point == 0:
        raise ValueError(
            f"Incompatible units for conversion: {units_from} -> {units_to}."
        )

    # Extract prefixes
    units_from_prefix = units_from[:-divergence_point]
    units_to_prefix = units_to[:-divergence_point]

    # Check if the prefixes are valid
    if units_from_prefix not in _unit_prefixes:
        raise ValueError(f"Invalid unit prefix '{units_from_prefix}'.")
    if units_to_prefix not in _unit_prefixes:
        raise ValueError(f"Invalid unit prefix '{units_to_prefix}'.")

    # Calculate and return the conversion factor
    return _unit_prefixes[units_from_prefix] / _unit_prefixes[units_to_prefix]


_numerical_pattern = re.compile(
    r"^\s*(?P<number>[+\-]?\d+(?:\.\d*)?(?:[eE]?[+\-]?\d+)?)\s*(?P<units>[a-zA-Z]+)"
)


def parse_with_units(value: str, desired_units: str | None = None) -> tuple[float, str]:
    """
    Reads numerical values from strings which include units. If a desired unit is provided, the value
    is converted.
    Only SI prefixes are supported, from Peta- to femto- (case sensitive!).
    Note: greek letter mu is replaced with 'u'.

    @param value: str
        A string representing numerical values with units.
    @param desired_units: str | None
        The desired unit for the output value. If none is specified, the input unit is used.
    @return: tuple[float, str]
        A tuple with the numerical value and its units (as a string).
    @raise: ValueError
        If the units to convert are different (beyond the prefix) or if the prefix is not supported.
    """
    match = _numerical_pattern.match(value)
    if match is None:
        raise ValueError(f"Unable to parse '{value}' as a number with units.")
    numerical_value = float(match.group("number"))
    given_units = match.group("units")
    if desired_units is None:
        return numerical_value, given_units
    return (
        numerical_value * conversion_factor(given_units, desired_units),
        desired_units,
    )


class ThreadEx(Thread):
    """
    Overrides the default run method to capture any exception raised during execution.
    """

    def run(self):
        try:
            super().run()
        except Exception as e:
            self.exception = e

    def raise_exception(self):
        if hasattr(self, "exception") and self.exception is not None:
            raise self.exception


def run_all(*threads: Thread):
    """
    Run multiple threads in parallel.
    Returns when all threads are finished.
    If any of the threads fail (by raising an exception), the exception is raised in the caller thread.

    @param threads: *Thread
        The threads which should be started.
    """

    # Add an exception member to keep track of thread-specific exceptions
    for thread in threads:
        thread.__class__ = ThreadEx
        thread.exception = None
        thread.start()
    # Let all threads conclude
    for thread in threads:
        thread.join()
    # Raise the first exception we can find
    for thread in threads:
        if thread.exception is not None:
            thread.exception.add_note(f"Occurred in thread '{thread.name}'.")
            raise thread.exception


def flush_queue(queue_item: Queue) -> None:
    """
    Empties a queue completely, ignoring all items.

    @param queue_item: Queue
        The queue to empty.
    """
    while True:
        try:
            queue_item.get(block=False)
        except Empty:
            return
        else:
            queue_item.task_done()


def get_ast_structure(
    file_path: str, what_to_return: Literal["functions", "classes", "all"]
) -> Union[list, dict]:
    """
    Gets the Abstract Syntax Tree (AST) structure of a Python file, as a list or dictionary
    of functions and classes.

    @param file_path: str the path to the file to parse
    @param what_to_return: Literal['functions', 'classes', 'all'] what to return from the file

    @return: Union[list, dict] the AST structure of the file
        if 'functions' is what_to_return, a list of functions
        if 'classes' is what_to_return, a list of classes
        if 'all' is what_to_return, a dictionary of class names to lists of functions

    @raise: FileNotFoundError if the file does not exist, ValueError if the file is not a Python file, or if the value of what_to_return is invalid
    """
    # check the file exists and is a python file
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    if not file_path.endswith(".py"):
        raise ValueError(f"File {file_path} is not a Python file.")

    # parse the file
    with open(file_path, "r") as file:
        file_contents = file.read()
        ast_tree = ast.parse(file_contents)

    # Helper functions for recursion
    def get_top_level_functions(tree_body):
        """Extracts top-level function names."""
        return [node.name for node in tree_body if isinstance(node, ast.FunctionDef)]

    def get_classes_and_methods(tree_body):
        """Extracts class names and their methods."""
        classes = {}
        for node in tree_body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = [
                    sub_node.name
                    for sub_node in node.body
                    if isinstance(sub_node, ast.FunctionDef)
                ]
        return classes

    # Perform extraction based on the request
    match what_to_return:
        case "functions":
            return get_top_level_functions(ast_tree.body)  # Top-level functions
        case "classes":
            return list(get_classes_and_methods(ast_tree.body).keys())  # Class names
        case "all":
            classes = get_classes_and_methods(
                ast_tree.body
            )  # Classes and their methods
            functions = get_top_level_functions(ast_tree.body)  # Top-level functions
            return {"classes": classes, "functions": functions}
        case _:
            raise ValueError(f"Invalid value for what_to_return: {what_to_return}")


def get_module_structure(
    module_name: str, what_to_return: Literal["functions", "classes", "all"]
) -> Union[List[str], Dict[str, List[str]]]:
    """
    Gets the structure of a module, as a list or dictionary of functions and classes.

    @param module_name: str The full name of the module (e.g., 'package.subpackage.file').
    @param what_to_return: Literal['functions', 'classes', 'all'] What to return from the module.

    @return: Union[List[str], Dict[str, List[str]]] The structure of the module:
        - If 'functions', returns a list of function names.
        - If 'classes', returns a list of class names.
        - If 'all', returns a dictionary of class names to lists of methods and a list of top-level functions.

    @raise: ImportError if the module cannot be imported, or ValueError if what_to_return is invalid.
    """
    # Dynamically import the module
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_name}'.") from e

    # Helper functions to get functions and classes
    def get_functions(module):
        """Extracts top-level function names."""
        return [
            name
            for name, obj in inspect.getmembers(module, inspect.isfunction)
            if obj.__module__ == module.__name__
        ]

    def get_classes(module):
        """Extracts class names and their methods."""
        classes = {}
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__:  # Only classes defined in this module
                classes[name] = [
                    method_name
                    for method_name, method_obj in inspect.getmembers(
                        obj, inspect.isfunction
                    )
                ]
        return classes

    # Perform extraction based on the request
    match what_to_return:
        case "functions":
            return get_functions(module)  # Top-level functions
        case "classes":
            return list(get_classes(module).keys())  # Class names
        case "all":
            classes = get_classes(module)  # Classes and their methods
            functions = get_functions(module)  # Top-level functions
            return {"classes": classes, "functions": functions}
        case _:
            raise ValueError(f"Invalid value for what_to_return: {what_to_return}")


def get_function_from_globals(function_name: str, globals) -> Callable:
    """
    Gets a function if it has been imported in the global namespace.
    @param function_name: str the name of the function to get
    @parma globals: globals, of the namespace where you call this function
    @return: Callable the function with the given name
    @raise: ValueError if the function is not found in the namespace or if the variable is not a function
    """
    if function_name in globals:
        func = globals[function_name]
        if callable(func):
            return func
        else:
            raise ValueError(f"Variable {function_name} is not a function.")
    else:
        raise ValueError(f"Function {function_name} not found in the global namespace.")


def get_imported_structure(
    module_name: str,
    what_to_return: Literal["functions", "classes", "all"],
    caller_globals: Dict[str, Any],
) -> Union[List[str], Dict[str, List[str]]]:
    """
    Gets the structure of functions and classes specifically defined in the specified module.

    @param module_name: str The name of the module to filter by (e.g., 'package.subpackage.file').
    @param what_to_return: Literal['functions', 'classes', 'all'] What to return from the imported namespace.
    @param caller_globals: Dict[str,Any] The globals() dictionary of the calling module.

    @return: Union[List[str], Dict[str, List[str]]] The structure of the specified module:
        - If 'functions', returns a list of function names.
        - If 'classes', returns a list of class names.
        - If 'all', returns a dictionary of class names to lists of methods and a list of top-level functions.

    @raise: ValueError if what_to_return is invalid.
    """

    # Helper functions to filter functions and classes from the global namespace
    def get_functions():
        """Extracts function names from the global namespace for the specified module."""
        return [
            name
            for name, obj in caller_globals.items()
            if inspect.isfunction(obj) and obj.__module__ == module_name
        ]

    def get_classes():
        """Extracts class names and their methods from the global namespace for the specified module."""
        classes = {}
        for name, obj in caller_globals.items():
            if inspect.isclass(obj) and obj.__module__ == module_name:
                classes[name] = [
                    method_name
                    for method_name, method_obj in inspect.getmembers(
                        obj, inspect.isfunction
                    )
                ]
        return classes

    # Perform extraction based on the request
    match what_to_return:
        case "functions":
            return get_functions()  # Top-level functions
        case "classes":
            return list(get_classes().keys())  # Class names
        case "all":
            classes = get_classes()  # Classes and their methods
            functions = get_functions()  # Top-level functions
            return {"classes": classes, "functions": functions}
        case _:
            raise ValueError(f"Invalid value for what_to_return: {what_to_return}")

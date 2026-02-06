"""
File: test_run_all.py
Author: Simone Pilon - Noël Research Group - 2024
GitHub: https://github.com/simone16

Description: unit test for Gpio array using mock interface.
"""

import unittest
from time import sleep
from threading import Thread

from omniplatypus.utilities.general import run_all


def foo(name: str, delay: float):
    print(f"{name} started.")
    sleep(delay)
    print(f"{name} finished.")


def foo_raise(name: str, delay: float):
    print(f"{name} started.")
    sleep(delay)
    raise RuntimeError(f"{name} raised me.")


class TestRunAll(unittest.TestCase):
    def test_success(self):
        thread_1 = Thread(
            target=foo, name="foo1", kwargs={"name": "foo1", "delay": 1.0}
        )
        thread_2 = Thread(
            target=foo, name="foo2", kwargs={"name": "foo2", "delay": 1.0}
        )
        run_all(thread_1, thread_2)

    def test_fail(self):
        thread_1 = Thread(
            target=foo, name="foo1", kwargs={"name": "foo1", "delay": 1.0}
        )
        thread_2 = Thread(
            target=foo_raise,
            name="foo_raise",
            kwargs={"name": "foo_raise", "delay": 1.0},
        )
        self.assertRaises(RuntimeError, run_all, thread_1, thread_2)

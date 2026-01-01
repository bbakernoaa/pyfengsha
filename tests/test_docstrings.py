import unittest
import inspect
from pyfengsha import fengsha


class TestDocstrings(unittest.TestCase):
    def test_docstrings(self):
        """
        Test that all functions in fengsha.py have NumPy-style docstrings.
        """
        functions = inspect.getmembers(fengsha, inspect.isfunction)
        for name, func in functions:
            if func.__module__ == "pyfengsha.fengsha":
                with self.subTest(name=name):
                    self.assertIsNotNone(func.__doc__, f"{name} has no docstring.")
                    self.assertIn(
                        "Parameters",
                        func.__doc__,
                        f"{name} is missing Parameters section.",
                    )
                    self.assertIn(
                        "Returns", func.__doc__, f"{name} is missing Returns section."
                    )


if __name__ == "__main__":
    unittest.main()

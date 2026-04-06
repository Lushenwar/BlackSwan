"""
Test fixture: function that always raises an exception.

Used to test that the CLI reliably reports exit code 1 and populates
shatter_points when every iteration produces a failure.
"""


def compute_value(x=1.0):
    """Always raises ArithmeticError. Every iteration is a critical failure."""
    raise ArithmeticError("Intentional failure — this fixture always fails")

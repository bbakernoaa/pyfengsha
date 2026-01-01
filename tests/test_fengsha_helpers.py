"""
Unit tests for helper functions in the fengsha module using pytest.
"""

import numpy as np
import pytest
from pyfengsha.fengsha import kok_aerosol_distribution

def test_kok_aerosol_distribution():
    """
    Test the Kok aerosol distribution function for correctness.

    Validates the function's output against a known, pre-calculated
    result and ensures that the distribution is correctly normalized.
    """
    # Test with a standard case
    radius = np.array([0.1, 0.5, 1.0])
    r_low = np.array([0.05, 0.45, 0.95])
    r_up = np.array([0.15, 0.55, 1.05])
    expected = np.array([0.01204987, 0.29441288, 0.69353726])

    result = kok_aerosol_distribution(radius, r_low, r_up)

    np.testing.assert_allclose(result, expected, rtol=1e-6)
    assert np.sum(result) == pytest.approx(1.0, abs=1e-6)

def test_kok_aerosol_distribution_zero_sum():
    """
    Test the Kok aerosol distribution function with a zero-sum case.

    Ensures that if the underlying (un-normalized) distribution is all
    zeros, the function returns an array of zeros instead of NaNs from
    a division-by-zero error.
    """
    # A case that will result in a zero or near-zero distribution
    radius = np.array([1e-9, 2e-9])
    r_low = np.array([0.5e-9, 1.5e-9])
    r_up = np.array([1.5e-9, 2.5e-9])
    expected = np.array([0.0, 0.0])

    result = kok_aerosol_distribution(radius, r_low, r_up)

    np.testing.assert_allclose(result, expected)

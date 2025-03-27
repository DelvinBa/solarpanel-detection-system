"""
Test utilities for the Solar Panel Detection project.
This file contains simple test examples for demonstration.
"""

import os
import pytest


def test_environment_setup():
    """Test that the environment is set up correctly."""
    assert True, "Environment should be set up"


def test_project_structure():
    """Test that the project has the expected directory structure."""
    # Test for some crucial directories
    assert os.path.exists("src"), "src directory should exist"


def test_simple_calculation():
    """Test a simple calculation to demonstrate testing."""
    assert 2 + 2 == 4, "Basic arithmetic should work"


@pytest.mark.parametrize("input_val,expected", [
    (1, 1),
    (2, 4),
    (3, 9),
    (4, 16),
])
def test_square_function(input_val, expected):
    """Test a square function with parameterized tests."""
    def square(x):
        return x * x
    
    result = square(input_val)
    assert result == expected, f"Square of {input_val} should be {expected}" 
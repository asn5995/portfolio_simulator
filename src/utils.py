"""
Common helper utilities.
"""

from typing import List, Dict
import numpy as np


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a number as currency."""
    return f"${value:,.{decimals}f}"


def format_percent(value: float, decimals: int = 2) -> str:
    """Format a number as percentage."""
    return f"{value * 100:.{decimals}f}%"


def validate_probabilities(probabilities: List[float], tolerance: float = 1e-6) -> bool:
    """Validate that probabilities sum to 1.0."""
    return abs(sum(probabilities) - 1.0) < tolerance


def validate_positive(value: float, name: str = "value") -> None:
    """Validate that a value is positive."""
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")


def validate_non_negative(value: float, name: str = "value") -> None:
    """Validate that a value is non-negative."""
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")


"""Shared utilities for the QKD dual-capacity routing simulation."""

import math

# Floating-point tolerance for capacity comparisons
EPS = 1e-9


def canonical_edge(u: int, v: int) -> tuple:
    """Return canonical undirected edge key as (min, max)."""
    return (min(u, v), max(u, v))


def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Safe division returning default when denominator is near zero."""
    if abs(denominator) < EPS:
        return default
    return numerator / denominator

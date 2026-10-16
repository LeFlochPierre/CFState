"""Shared, non-mutating scientific computations for CF-State."""
from .imodulons import (
    calibrate_with_operator,
    counts_to_logtpm,
    project_calibrated,
    project_residual,
    residualize_controls,
)

__all__ = [
    "calibrate_with_operator",
    "counts_to_logtpm",
    "project_calibrated",
    "project_residual",
    "residualize_controls",
]

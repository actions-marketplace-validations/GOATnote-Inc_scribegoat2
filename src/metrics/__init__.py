"""
ScribeGoat2 Metrics Module

Provides statistical computation with uncertainty quantification (BR-3).
"""

from .confidence_intervals import (
    IntervalResult,
    bootstrap_pass_k_interval,
    compute_metrics_with_ci,
    wilson_score_interval,
)

__all__ = [
    "wilson_score_interval",
    "bootstrap_pass_k_interval",
    "compute_metrics_with_ci",
    "IntervalResult",
]

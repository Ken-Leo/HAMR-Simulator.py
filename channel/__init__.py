"""Channel module for HAMR Simulator.

Provides channel models (Longitudinal, Perpendicular, HAMR),
low-pass filters, FIR filters, media noise, and math utilities.
"""

from channel.math_utils import (
    PI,
    uniform_random,
    gaussian_random,
    autocorr,
    cross_corr,
    mat_inverse,
    solve_linear_system,
)
from channel.fir import non_causal_fir, causal_fir
from channel.lpf import lpf
from channel.channel import longitudinal_channel, perpendicular_channel, hamr_channel
from channel.media_noise import media_noise_filter

__all__ = [
    "PI",
    "uniform_random",
    "gaussian_random",
    "autocorr",
    "cross_corr",
    "mat_inverse",
    "solve_linear_system",
    "non_causal_fir",
    "causal_fir",
    "lpf",
    "longitudinal_channel",
    "perpendicular_channel",
    "hamr_channel",
    "media_noise_filter",
]

"""Low-pass filter for the HAMR simulator.

Matches the C LPF function in MagneticDisk.c: designs a Hamming-windowed
sinc low-pass filter and applies it via non-causal FIR.
"""

from __future__ import annotations

import math

import numpy as np

from channel.fir import non_causal_fir
from channel.math_utils import PI


def lpf(channel_output: np.ndarray, filter_order: int,
        cutoff: float) -> np.ndarray:
    """Design and apply a Hamming-windowed sinc low-pass filter.

    The cutoff frequency is a fraction of the Nyquist rate.
    A cut-off of ``1/OSR`` gives a filter with bandwidth ``1/OSR`` samples.

    Args:
        channel_output: Input signal.
        filter_order: Filter order (number of taps = filter_order + 1).
        cutoff: Normalised cutoff frequency (e.g. 1/OSR).

    Returns:
        Filtered signal of length len(channel_output) + filter_order.
    """
    channel_output = np.asarray(channel_output, dtype=np.float64)
    cutoff_rad = cutoff * PI

    # Design sinc filter with Hamming window
    h = np.zeros(filter_order + 1, dtype=np.float64)
    half = filter_order / 2.0

    for i in range(filter_order + 1):
        if i == half:
            h[i] = cutoff_rad / PI
        else:
            h[i] = (
                math.sin(cutoff_rad * (i - half))
                / (PI * (i - half))
            )
        # Hamming window
        h[i] *= (0.54 - 0.46 * math.cos(2 * PI * i / filter_order))

    # Normalize DC gain to 1
    h = h / np.sum(h)

    # Apply filter
    return non_causal_fir(channel_output, h)

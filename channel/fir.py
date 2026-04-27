"""FIR filters for the HAMR simulator.

Provides non-causal and causal FIR filtering, matching the C
implementations of NonCausalFIR and CausalFIR in MagneticDisk.c.
"""

from __future__ import annotations

import numpy as np


def non_causal_fir(data: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply a non-causal FIR filter.

    The impulse response centre h[0] is assumed to be at
    h[floor(len(h)/2)].

    The output has length ``len(data) + floor(len(h)/2)``.

    Args:
        data: Input signal.
        h: Impulse response (centred at len(h)//2).

    Returns:
        Filtered signal.
    """
    data = np.asarray(data, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    num_taps = len(h)
    front_pad = num_taps // 2
    back_pad = num_taps - 1

    padded = np.zeros(
        len(data) + front_pad + back_pad, dtype=np.float64
    )
    padded[front_pad: front_pad + len(data)] = data

    output_len = len(data) + front_pad
    output = np.zeros(output_len, dtype=np.float64)

    for i in range(output_len):
        for j in range(num_taps):
            output[i] += h[j] * padded[i + j]

    return output


def causal_fir(data: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply a causal FIR filter.

    Unlike the non-causal variant, the output has length
    ``len(data) + len(h) - 1``.

    Args:
        data: Input signal.
        h: Impulse response (h[0] is the first tap).

    Returns:
        Filtered signal of length len(data) + len(h) - 1.
    """
    data = np.asarray(data, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    return np.convolve(data, h, mode="full")

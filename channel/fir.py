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

    # Pad with front_pad zeros at front, back_pad zeros at back
    padded = np.zeros(
        len(data) + front_pad + back_pad, dtype=np.float64
    )
    padded[front_pad: front_pad + len(data)] = data

    # Original: output[i] = sum_j h[j] * padded[i+j]
    # This is correlation. np.convolve with reversed h gives correlation.
    # np.convolve(padded, h[::-1])[offset:offset+output_len] matches at offset=len(h)-1
    full_conv = np.convolve(padded, h[::-1], mode="full")

    output_len = len(data) + front_pad
    start = len(h) - 1
    output = full_conv[start: start + output_len]

    return output


def causal_fir(data: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Apply a causal FIR filter.

    Unlike the non-causal variant, the output has the **same length** as
    the input (``len(data)``).  Transient samples at the start are
    discarded, matching the C ``CausalFIR`` behaviour in MagneticDisk.c.

    Args:
        data: Input signal.
        h: Impulse response (h[0] is the first tap).

    Returns:
        Filtered signal of length ``len(data)``.
    """
    data = np.asarray(data, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    num_taps = len(h)
    pad_len = num_taps - 1

    # Pad with zeros on both sides, then compute valid-region output
    padded = np.zeros(len(data) + 2 * pad_len, dtype=np.float64)
    padded[pad_len: pad_len + len(data)] = data

    output = np.zeros(len(data), dtype=np.float64)
    for i in range(pad_len, len(data) + pad_len):
        val = 0.0
        for j in range(num_taps):
            val += h[j] * padded[i - j]
        output[i - pad_len] = val

    return output

"""Media noise models for the HAMR simulator.

Provides jitter and pulse-broadening noise models matching the
C MediaNoiseFilter() function in MagneticDisk.c (lines 1898-2081).

When media noise is enabled (sigma_jitter > 0 or sigma_pulse_broad > 0),
the channel output is re-computed from scratch using per-transition
noise rather than a simple FIR convolution.
"""

from __future__ import annotations

import math

import numpy as np

from channel.math_utils import PI, LCG, gaussian_random, erf


def media_noise_filter(
    os_padded_bits: np.ndarray,
    sector_length: int,
    read_channel_coeffs: np.ndarray,
    num_channel_taps: int,
    time_index: np.ndarray,
    nd: float,
    sigma_jitter: float,
    sigma_pulse_broad: float,
    osr: int,
    channel_type: str = "Longitudinal",
    seed_jitter: int = -200,
    seed_pulse: int = -100,
) -> np.ndarray:
    """Apply media noise (jitter + pulse broadening) to channel output.

    Exact translation of C ``MediaNoiseFilter()`` from MagneticDisk.c
    (lines 1898-2081).  Re-computes the channel output from scratch
    using per-transition noise rather than a simple FIR convolution.

    Algorithm:
    1. Differentiate bipolar bits to find transitions.
    2. For each transition, generate jitter (DeltaX) and pulse-broadening
       (DeltaND) noise samples.
    3. Compute channel output by evaluating the pulse shape at each tap
       position, with per-transition noise applied.

    Parameters
    ----------
    os_padded_bits : np.ndarray
        Oversampled bipolar bits (+1/-1), length ``sector_length``.
    sector_length : int
        Number of oversampled samples.
    read_channel_coeffs : np.ndarray
        Channel impulse response coefficients (unused in media-noise
        mode; kept for API compatibility with the C signature).
    num_channel_taps : int
        Number of channel filter taps.
    time_index : np.ndarray
        Normalised time vector of length ``num_channel_taps``.
    nd : float
        Normalised density.
    sigma_jitter : float
        Jitter noise std-dev as % of bit period.
    sigma_pulse_broad : float
        Pulse broadening noise std-dev as % of bit period.
    osr : int
        Oversampling rate.
    channel_type : str
        "Longitudinal" or "Perpendicular".
    seed_jitter : int
        RNG seed for jitter noise.
    seed_pulse : int
        RNG seed for pulse-broadening noise.

    Returns
    -------
    np.ndarray
        Channel output with media noise applied, length
        ``sector_length + num_channel_taps // 2``.
    """
    # --- Step 1: Differentiate bipolar input ---
    # diff[i] = bits[i] - bits[i-1], values are {-2, 0, +2}
    diff = np.zeros(sector_length, dtype=np.int64)
    diff[0] = 0
    for i in range(1, sector_length):
        diff[i] = int(os_padded_bits[i]) - int(os_padded_bits[i - 1])

    # --- Step 2: Find transitions and generate noise per transition ---
    lcg_jitter = LCG(seed_jitter)
    lcg_pulse = LCG(seed_pulse)

    # Collect transition indices
    transition_indices: list[int] = []
    for i in range(sector_length):
        if diff[i] != 0:
            transition_indices.append(i)

    num_trans = len(transition_indices)

    # Generate DeltaX (truncated Gaussian) and DeltaND (single-sided Gaussian)
    delta_x = np.zeros(num_trans, dtype=np.float64)
    delta_nd = np.zeros(num_trans, dtype=np.float64)

    sigma_j = sigma_jitter / 100.0
    sigma_pb = sigma_pulse_broad / 100.0

    for t in range(num_trans):
        # DeltaX: truncated Gaussian, |DeltaX| < 0.5
        dx = 0.5
        while abs(dx) >= 0.5:
            dx = sigma_j * gaussian_random(lcg_jitter)
        delta_x[t] = dx

        # DeltaND: single-sided Gaussian, DeltaND >= 0
        dnd = -1.0
        while dnd < 0:
            dnd = sigma_pb * gaussian_random(lcg_pulse)
        delta_nd[t] = dnd

    # --- Step 3: Pad differentiated data for FIR-style filtering ---
    front_pad = num_channel_taps // 2
    back_pad = num_channel_taps - 1
    total_pad = sector_length + front_pad + back_pad

    padded_data = np.zeros(total_pad, dtype=np.int64)
    for i in range(total_pad):
        if i <= front_pad - 1 or i >= front_pad + sector_length:
            padded_data[i] = 0
        else:
            padded_data[i] = diff[i - front_pad]

    # --- Step 4: Compute channel output with per-transition noise ---
    output_length = sector_length + front_pad
    output = np.zeros(output_length, dtype=np.float64)

    current_data_tap_loc = num_channel_taps // 2  # Location of h(0)

    # Build a lookup: for each padded_data index, which transition number is it?
    # Transition numbers are 1-indexed in the C code
    # Map from padded_data index to transition index (0-based into delta_x/delta_nd)
    trans_map: dict[int, int] = {}
    trans_counter = 0
    for idx in range(total_pad):
        if padded_data[idx] != 0:
            trans_map[idx] = trans_counter
            trans_counter += 1

    for i in range(output_length):
        # Check if current tap position (corresponding to h(0)) is a transition
        center_idx = i + current_data_tap_loc
        if padded_data[center_idx] != 0:
            # This is a transition — compute h(0) * current_bit
            t_num = trans_map[center_idx]
            current_tap_value = _pulse_value(
                time_index[current_data_tap_loc],
                delta_x[t_num],
                delta_nd[t_num],
                nd,
                channel_type,
            )
            output[i] += padded_data[center_idx] * current_tap_value

        # Future taps: h(1) to h(end)
        for j in range(current_data_tap_loc + 1, num_channel_taps):
            if padded_data[i + j] != 0:
                t_num = trans_map[i + j]
                current_tap_value = _pulse_value(
                    time_index[j],
                    delta_x[t_num],
                    delta_nd[t_num],
                    nd,
                    channel_type,
                )
                output[i] += padded_data[i + j] * current_tap_value

        # Past taps: h(-1) to h(start)
        for j in range(current_data_tap_loc - 1, -1, -1):
            if padded_data[i + j] != 0:
                t_num = trans_map[i + j]
                current_tap_value = _pulse_value(
                    time_index[j],
                    delta_x[t_num],
                    delta_nd[t_num],
                    nd,
                    channel_type,
                )
                output[i] += padded_data[i + j] * current_tap_value

    return output


def _pulse_value(
    time_val: float,
    delta_x: float,
    delta_nd: float,
    nd: float,
    channel_type: str,
) -> float:
    """Compute the pulse shape value at a given time with noise applied.

    Matches the C code pulse shape formulas in MediaNoiseFilter.

    Parameters
    ----------
    time_val : float
        Normalised time index value.
    delta_x : float
        Jitter offset for this transition.
    delta_nd : float
        Pulse broadening offset for this transition.
    nd : float
        Normalised density.
    channel_type : str
        "Longitudinal" or "Perpendicular".

    Returns
    -------
    float
        Pulse shape value.
    """
    t_noisy = time_val + delta_x

    if channel_type in ("perpendicular", "Perpendicular"):
        # Perpendicular: erf pulse shape
        # erf(2/ND * sqrt(log(2)) * (TimeIndex + DeltaX))
        return erf(2.0 / nd * math.sqrt(math.log(2)) * t_noisy)
    else:
        # Longitudinal: Lorentzian derivative pulse shape
        # sqrt(2/pi) / (1 + 4*(TimeIndex+DeltaX)^2 / (ND+DeltaND)^2)
        denom = 1.0 + 4.0 * t_noisy ** 2 / (nd + delta_nd) ** 2
        return math.sqrt(2.0 / PI) / denom

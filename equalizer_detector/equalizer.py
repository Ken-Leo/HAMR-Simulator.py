"""FIR filters, low-pass filter, and adaptive equalizer.

Translates: NonCausalFIR, CausalFIR, LPF, AdaptEqualizer
from MagneticDisk.c (lines 1128-1333).
"""

from typing import Tuple

import numpy as np

LMS_STEP_SIZE: float = 0.005
MIN_MSE: float = 0.005


def non_causal_fir(
    data: np.ndarray,
    data_length: int,
    h: np.ndarray,
) -> np.ndarray:
    """Non-causal (symmetric) FIR filter. Centers the filter response.

    The filter coefficients are laid out as:
        h: ... h[-2] h[-1] h[0] h[1] h[2] ...
    with h[0] located at index floor(NumTaps / 2).

    Parameters
    ----------
    data : np.ndarray
        Input signal of length ``data_length``.
    data_length : int
        Number of valid samples in *data*.
    h : np.ndarray
        Filter taps. ``h[floor(NumTaps/2)]`` is the centre tap.

    Returns
    -------
    np.ndarray
        Filtered output of length ``data_length + floor(NumTaps / 2)``.
    """
    num_taps = len(h)
    front_pad_length = num_taps // 2
    back_pad_length = num_taps - 1
    total_length = data_length + front_pad_length + back_pad_length

    padded_data = np.zeros(total_length)
    padded_data[front_pad_length : front_pad_length + data_length] = (
        data[:data_length]
    )

    output_length = data_length + front_pad_length
    output = np.zeros(output_length)

    for i in range(output_length):
        for j in range(num_taps):
            output[i] += h[j] * padded_data[i + j]

    return output


def causal_fir(
    data: np.ndarray,
    data_length: int,
    h: np.ndarray,
) -> np.ndarray:
    """Causal FIR filter.

    Unlike the non-causal version, the output vector has length
    ``data_length`` (valid samples only; transient samples are discarded).

    Parameters
    ----------
    data : np.ndarray
        Input signal of length ``data_length``.
    data_length : int
        Number of valid samples in *data*.
    h : np.ndarray
        Filter taps.

    Returns
    -------
    np.ndarray
        Filtered output of length ``data_length``.
    """
    num_taps = len(h)
    pad_len = num_taps - 1
    total_length = data_length + 2 * pad_len

    padded_data = np.zeros(total_length)
    padded_data[pad_len : pad_len + data_length] = data[:data_length]

    output = np.zeros(data_length)

    for i in range(pad_len, data_length + pad_len):
        channel_output = 0.0
        for j in range(num_taps):
            channel_output += h[j] * padded_data[i - j]
        output[i - pad_len] = channel_output

    return output


def lpf(
    channel_output: np.ndarray,
    filter_order: int,
    cutoff: float,
) -> np.ndarray:
    """Low-pass FIR filter using the windowed-sinc method with a Hamming window.

    Parameters
    ----------
    channel_output : np.ndarray
        Input signal to filter.
    filter_order : int
        Order of the filter (number of taps = ``filter_order + 1``).
    cutoff : float
        Normalised cutoff frequency (0 < cutoff < 1).

    Returns
    -------
    np.ndarray
        Filtered output.
    """
    num_taps = filter_order + 1
    cutoff_rad = cutoff * np.pi
    mid = filter_order / 2.0

    h = np.zeros(num_taps)
    for i in range(filter_order, -1, -1):
        if i == mid:
            h[i] = cutoff_rad / np.pi
        else:
            h[i] = np.sin(cutoff_rad * (i - mid)) / (np.pi * (i - mid))
        # Hamming window
        h[i] *= 0.54 - 0.46 * np.cos(2.0 * np.pi * i / filter_order)

    # Normalise DC gain to 1
    h = h / np.sum(h)

    return non_causal_fir(channel_output, len(channel_output), h)


def adapt_equalizer(
    equalized_output: np.ndarray,
    padded_eq_output: np.ndarray,
    eq_coeff: np.ndarray,
    num_eq_taps: int,
    sector_length: int,
    pri_imp_res: np.ndarray,
    pri_imp_res_length: int,
    start_flag: int = 1,
) -> Tuple[float, float]:
    """LMS adaptive equalizer.

    Updates ``eq_coeff`` in-place and returns the MSE and LMS error
    for the current sector.

    Parameters
    ----------
    equalized_output : np.ndarray
        Raw channel output (DSChannelOutput equivalent).
    padded_eq_output : np.ndarray
        Already-equalized output used to compute the PR target.
    eq_coeff : np.ndarray
        Equalizer coefficients (updated in-place).
    num_eq_taps : int
        Number of equalizer taps.
    sector_length : int
        Number of samples in the sector.
    pri_imp_res : np.ndarray
        PR target impulse response.
    pri_imp_res_length : int
        Length of the PR impulse response.
    start_flag : int
        If non-zero, re-initialise equalizer coefficients.

    Returns
    -------
    tuple[float, float]
        (MSE, average_LMS_error) for the sector.
    """
    # Initialise equalizer coefficients on start
    if start_flag:
        eq_coeff[:] = 0.0

    # --- PR target output (causal FIR of equalized_output) ---
    pr_actual_output = causal_fir(equalized_output, sector_length, pri_imp_res)

    # --- Equalizer output (non-causal FIR of equalized_output) ---
    eq_output_temp = non_causal_fir(equalized_output, sector_length, eq_coeff)

    # --- LMS coefficient update ---
    # Pad the channel output for the equaliser
    front_pad = num_eq_taps // 2
    back_pad = num_eq_taps - 1
    padded_len = sector_length + front_pad + back_pad
    padded_ds = np.zeros(padded_len)
    padded_ds[front_pad : front_pad + sector_length] = equalized_output[
        :sector_length
    ]

    lmse_total = 0.0
    for i in range(sector_length):
        eq_output = 0.0
        for j in range(num_eq_taps):
            eq_output += eq_coeff[j] * padded_ds[i + j]

        eq_error = eq_output - pr_actual_output[i]
        lmse_total += eq_error * eq_error

        for j in range(num_eq_taps):
            eq_coeff[j] -= 2.0 * LMS_STEP_SIZE * eq_error * padded_ds[i + j]

    # --- MSE computation ---
    mse = 0.0
    for i in range(sector_length):
        diff = pr_actual_output[i] - eq_output_temp[i]
        mse += diff * diff

    avg_lmse = lmse_total / sector_length if sector_length > 0 else 0.0

    return float(mse), float(avg_lmse)


def apply_equalizer(
    data: np.ndarray,
    eq_coeff: np.ndarray,
    num_eq_taps: int,
) -> np.ndarray:
    """Apply a fixed FIR equalizer with given coefficients.

    This is the equalizer used during the main simulation loop (not adaptive).

    Args:
        data: Input signal samples.
        eq_coeff: Equalizer FIR coefficients.
        num_eq_taps: Number of taps.

    Returns:
        Equalized signal of same length as input.
    """
    # Use non-causal FIR with the equalizer coefficients
    data_length = len(data)
    num_taps = len(eq_coeff)
    front_pad = num_taps // 2
    back_pad = num_taps - 1
    total_length = data_length + front_pad + back_pad

    padded = np.zeros(total_length, dtype=np.float64)
    padded[front_pad : front_pad + data_length] = data

    output = np.zeros(data_length, dtype=np.float64)
    for i in range(data_length):
        for j in range(num_taps):
            output[i] += eq_coeff[j] * padded[i + j]

    return output


def find_gpr_target(
    eq_output: np.ndarray,  # equalized output (received)
    pr_output: np.ndarray,  # desired PR target output
    num_taps: int,
    gpr_target_length: int = 4,
) -> tuple:
    """Find optimal GPR (Generalized Partial Response) equalizer coefficients.

    Solves the normal equations: (R + alpha*I) * a = p
    where R is the autocorrelation matrix of the input,
    a is the equalizer coefficient vector,
    p is the cross-correlation between input and desired output.

    This matches the C FindGPRTarget() function.

    Args:
        eq_output: Actual equalized output (received signal).
        pr_output: Desired PR target output.
        num_taps: Number of equalizer taps.
        gpr_target_length: Length of PR target for energy constraint.

    Returns:
        (gpr_target, eq_coeff) where:
        - gpr_target: GPR target coefficients (length gpr_target_length)
        - eq_coeff: Equalizer coefficients (length num_taps)
    """
    data_length = len(eq_output)

    # Energy of PR target (for normalisation)
    # For EPR4: [1, 1, -1, -1], energy = 4
    # We use a default GPR target shape
    if gpr_target_length == 4:
        gpr_target = np.array([1.0, 1.0, 0.0, -1.0, -1.0], dtype=np.float64)
    elif gpr_target_length == 5:
        gpr_target = np.array([1.0, 1.0, 0.0, -1.0, -1.0], dtype=np.float64)
    else:
        gpr_target = np.zeros(gpr_target_length, dtype=np.float64)
        gpr_target[0] = 1.0
        gpr_target[1] = 1.0
        gpr_target[-1] = -1.0
        gpr_target[-2] = -1.0

    # Compute autocorrelation matrix R of eq_output
    # R[i][j] = corr(eq_output, j - i)
    R = np.zeros((num_taps, num_taps), dtype=np.float64)
    for i in range(num_taps):
        for j in range(num_taps):
            lag = j - i
            n = data_length - abs(lag)
            if n > 0:
                if lag >= 0:
                    R[i, j] = np.sum(eq_output[:n] * eq_output[lag : lag + n]) / n
                else:
                    R[i, j] = np.sum(eq_output[:n] * eq_output[-lag : -lag + n]) / n

    # Add small regularization (diagonal loading) for numerical stability
    alpha = 1e-6
    R += alpha * np.eye(num_taps)

    # Compute cross-correlation vector p between eq_output and pr_output
    p = np.zeros(num_taps, dtype=np.float64)
    min_len = min(data_length, len(pr_output))
    if min_len > 0:
        for i in range(num_taps):
            if i < min_len:
                p[i] = np.sum(eq_output[:min_len - i] * pr_output[i:min_len]) / (min_len - i)

    # Solve R * a = p for equalizer coefficients
    try:
        eq_coeff = np.linalg.solve(R, p)
    except np.linalg.LinAlgError:
        eq_coeff = np.linalg.lstsq(R, p, rcond=None)[0]

    return gpr_target, eq_coeff

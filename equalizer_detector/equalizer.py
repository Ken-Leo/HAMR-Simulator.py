"""FIR filters, low-pass filter, and adaptive equalizer.

Translates: NonCausalFIR, CausalFIR, LPF, AdaptEqualizer
from MagneticDisk.c (lines 1128-1333).
"""

from typing import Tuple

import numpy as np

LMS_STEP_SIZE: float = 0.005
MIN_MSE: float = 0.005
NLMS_EPSILON: float = 1e-6


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
    pri_imp_res: np.ndarray,
    eq_coeff: np.ndarray,
    num_eq_taps: int,
    clean_bits: np.ndarray,
    channel_output: np.ndarray,
    sector_length: int,
    start_flag: int = 1,
) -> Tuple[float, float]:
    """LMS adaptive equalizer (translation of C ``AdaptEqualizer``).

    The equalizer learns to invert the channel by comparing its output
    against a desired PR-target signal computed from *clean_bits*.

    In the C code::

        CausalFIR(PaddedEncodedBits, SectorLength, PRActualOutput,
                  PRImpRes, PRImpResLength);
        // then LMS: EqCoeff[j] -= 2*STEP*EqError*PaddedDS[i+j]

    Parameters
    ----------
    pri_imp_res : np.ndarray
        PR impulse response (target shape, e.g. EPR4 ``[1,1,-1,-1]``).
    eq_coeff : np.ndarray
        Equalizer coefficients array (modified **in-place**).
    num_eq_taps : int
        Number of equalizer taps.
    clean_bits : np.ndarray
        Clean bipolar bits (0/1 → ±1), same length as the sector.
        These are convolved with *pri_imp_res* to form the desired
        PR-target output.
    channel_output : np.ndarray
        Received channel samples (after LPF + downsampling).
    sector_length : int
        Number of samples to process.
    start_flag : int
        If non-zero, re-initialise coefficients to zero.

    Returns
    -------
    tuple[float, float]
        (MSE, average_LMS_error) for the sector.
    """
    pri_imp_res_length = len(pri_imp_res)

    # Initialise equalizer coefficients on start
    if start_flag:
        eq_coeff[:] = 0.0

    # --- Desired PR target output = causal_fir(clean_bits, pri_imp_res) ---
    pr_actual_output = causal_fir(clean_bits, sector_length, pri_imp_res)

    # --- Pad channel output for the equaliser ---
    front_pad = num_eq_taps // 2
    back_pad = num_eq_taps - 1
    total_pad = sector_length + front_pad + back_pad
    padded_ds = np.zeros(total_pad, dtype=np.float64)
    padded_ds[front_pad: front_pad + sector_length] = channel_output[
        :sector_length
    ]

    # --- LMS coefficient update ---
    lmse_total = 0.0
    for i in range(sector_length):
        eq_output = 0.0
        for j in range(num_eq_taps):
            eq_output += eq_coeff[j] * padded_ds[i + j]

        eq_error = eq_output - pr_actual_output[i]
        lmse_total += eq_error * eq_error

        for j in range(num_eq_taps):
            eq_coeff[j] -= 2.0 * LMS_STEP_SIZE * eq_error * padded_ds[i + j]

    # --- MSE: difference between PR output and equalized output ---
    mse = 0.0
    for i in range(sector_length):
        eq_out_val = 0.0
        for j in range(num_eq_taps):
            eq_out_val += eq_coeff[j] * padded_ds[i + j]
        diff = pr_actual_output[i] - eq_out_val
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


def _corr(a: np.ndarray, a_len: int,
          b: np.ndarray, b_len: int,
          shift: int) -> float:
    """Cross-correlation at a single lag, matching C ``Corr()``.

    Computes ``sum_k A[k] * B[k - shift]`` over the overlapping region.
    B is shifted to the right with respect to A (positive shift delays B).

    Parameters
    ----------
    a : np.ndarray
        First array.
    a_len : int
        Number of valid samples in *a*.
    b : np.ndarray
        Second array.
    b_len : int
        Number of valid samples in *b*.
    shift : int
        Lag (can be negative).

    Returns
    -------
    float
        Correlation value.
    """
    if shift >= 0:
        start = shift
    else:
        start = 0
    if a_len - 1 <= shift + b_len - 1:
        finish = a_len - 1
    else:
        finish = shift + b_len - 1
    if start > finish:
        return 0.0
    result = 0.0
    for k in range(start, finish + 1):
        result += a[k] * b[k - shift]
    return result


def _matrix_inv(A: np.ndarray) -> np.ndarray:
    """Gauss-Jordan matrix inverse, matching C ``MatrixInv()``.

    Parameters
    ----------
    A : np.ndarray
        Square matrix to invert.

    Returns
    -------
    np.ndarray
        Inverse of A.
    """
    m = A.shape[0]
    A = A.astype(np.float64).copy()
    B = np.eye(m, dtype=np.float64)

    for c in range(m):
        # Swap row c with any row that has a non-zero element in column c
        if A[c, c] == 0:
            for i in range(m):
                if A[i, c] != 0:
                    A[[c, i]] = A[[i, c]]
                    B[[c, i]] = B[[i, c]]
                    break

        # Scale row c so that A[c][c] == 1
        if A[c, c] != 1:
            temp = A[c, c]
            A[c] /= temp
            B[c] /= temp

        # Eliminate column c in all other rows
        for i in range(m):
            if i != c:
                temp = A[i, c]
                A[i] -= temp * A[c]
                B[i] -= temp * B[c]

    return B


def find_gpr_target(
    channel_output: np.ndarray,  # s: downsampled channel output
    bipolar_input: np.ndarray,  # a: clean bipolar bits (0/1 -> +/-1)
    num_taps: int,  # N: equalizer tap count
    gpr_target_length: int = 4,  # L: GPR target length
) -> tuple:
    """Compute GPR (Generalized Partial Response) equalizer coefficients.

    Exact translation of the C ``FindGPRTarget()`` from MagneticDisk.c
    (lines 855-1116).  Solves the optimality conditions derived from
    Jaekyun Moon et al., "Equalization for Maximum Likelihood Detectors",
    IEEE Trans. Mag., Vol 31, No. 2, Mar 1995.

    Algorithm (matching C exactly):

    1. Build Toeplitz autocorrelation matrices **R** (from channel
       output ``s``) and **A** (from bipolar input ``a``), plus
       cross-correlation matrix **T** (between ``s`` and ``a``).
    2. Compute ``R_inv = inv(R)`` via Gauss-Jordan elimination.
    3. Compute ``Temp2 = inv(A - T' * R_inv * T)``.
    4. **GPR target**: ``G = Lambda * Temp2[:, 0]`` where
       ``Lambda = 1 / Temp2[0][0]`` (monic constraint ``G[0] = 1``).
    5. **Equalizer taps**: ``F = R_inv * T * G`` (non-causal FIR
       stored in reverse order).

    Parameters
    ----------
    channel_output : np.ndarray
        Downsampled channel output (received samples), length >= N+L.
    bipolar_input : np.ndarray
        Clean bipolar input bits (0/1 → ±1), same length as
        ``channel_output``.
    num_taps : int
        Number of equalizer taps (N, typically 21).
    gpr_target_length : int
        Length of the GPR target (L, e.g. 4 for EPR4).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(gpr_target, eq_coeff)`` where ``gpr_target`` has length
        ``gpr_target_length`` and ``eq_coeff`` has length ``num_taps``.
    """
    data_length = len(channel_output)
    N = num_taps
    L = gpr_target_length
    K = N // 2  # Will work for both N even and odd

    # --- Compute first row of R (Toeplitz autocorrelation of s) ---
    r_row = np.zeros(N, dtype=np.float64)
    for i in range(N):
        r_row[i] = _corr(channel_output, data_length,
                         channel_output, data_length, i)

    # Fill R using Toeplitz symmetry
    R = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            R[i, j] = r_row[abs(j - i)]

    # --- Compute first row of A (Toeplitz autocorrelation of a) ---
    a_row = np.zeros(L, dtype=np.float64)
    for i in range(L):
        a_row[i] = _corr(bipolar_input, data_length,
                         bipolar_input, data_length, i)

    # Fill A using Toeplitz symmetry
    A = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            A[i, j] = a_row[abs(j - i)]

    # --- Compute T (cross-correlation between s and a) ---
    # T[i][j] = Corr(s, a, K - i + j)
    T = np.zeros((N, L), dtype=np.float64)
    for i in range(N):
        for j in range(L):
            T[i, j] = _corr(channel_output, data_length,
                            bipolar_input, data_length,
                            K - i + j)

    # --- Compute R_inv via Gauss-Jordan ---
    R_inv = _matrix_inv(R)

    # --- Compute Temp1 = R_inv * T ---
    temp1 = R_inv @ T  # [N x L]

    # --- Compute T' (transpose) ---
    T_T = T.T  # [L x N]

    # --- Compute Temp2 = T' * Temp1 = T' * R_inv * T ---
    temp2 = T_T @ temp1  # [L x L]

    # --- Compute A_temp = A - Temp2 ---
    a_temp = A - temp2

    # --- Compute A_temp_inv via Gauss-Jordan (stored back in temp2) ---
    # Note: C reuses Temp2 variable
    temp2 = _matrix_inv(a_temp)

    # --- Compute Lambda = 1 / Temp2[0][0] ---
    Lambda = 1.0 / temp2[0, 0]

    # --- Compute GPR target G = Lambda * Temp2[:, 0] ---
    gpr_target = Lambda * temp2[:, 0]

    # Monic constraint check (matches C: fabs(G[0] - 1.0) > PRE)
    if abs(gpr_target[0] - 1.0) > 1e-4:
        # Normalize to enforce monic constraint
        gpr_target /= gpr_target[0]

    # --- Compute equalizer coefficients F = R_inv * T * G ---
    # Compute T * G first
    t_g = T @ gpr_target  # [N]

    # Compute R_inv * (T * G)
    rinv_tg = R_inv @ t_g  # [N]

    # F is stored in reverse order (C: F[N-1-i] = Temp1[i][0])
    eq_coeff = np.zeros(N, dtype=np.float64)
    for i in range(N):
        eq_coeff[N - 1 - i] = rinv_tg[i]

    return gpr_target, eq_coeff

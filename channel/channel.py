"""Channel models for the HAMR simulator.

Provides Longitudinal, Perpendicular, and HAMR channel implementations
matching the C LongPerp() and Hamr() functions in MagneticDisk.c.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from channel.fir import non_causal_fir
from channel.math_utils import PI


def _compute_channel_coeffs(
    num_taps: int,
    time_index: np.ndarray,
    nd: float,
    channel_type: str,
) -> np.ndarray:
    """Compute channel impulse response coefficients.

    Args:
        num_taps: Number of taps.
        time_index: Normalised time vector.
        nd: Normalised density.
        channel_type: "longitudinal", "perpendicular", or "hamr".

    Returns:
        Channel coefficient array of length ``num_taps``.
    """
    coeffs = np.zeros(num_taps, dtype=np.float64)

    if channel_type in ("perpendicular", "Perpendicular"):
        vp = 0.5 * math.pow(PI / (2 * math.log(2)), 0.25)
        for i in range(num_taps):
            coeffs[i] = (
                vp * 2.0 / nd
                * math.sqrt(math.log(2) / PI)
                * math.exp(
                    -math.log(2) * 4 * time_index[i] ** 2 / nd ** 2
                )
            )
            coeffs[i] /= nd  # / OSR equivalent
    elif channel_type in ("longitudinal", "Longitudinal"):
        vp = math.sqrt(2.0 / PI)
        for i in range(num_taps):
            coeffs[i] = (
                vp * -8.0 * time_index[i] / nd ** 2
                * 1.0 / (1.0 + 4.0 * time_index[i] ** 2 / nd ** 2) ** 2
            )
            coeffs[i] /= nd
    else:
        raise ValueError(
            f"Unknown channel type '{channel_type}'. "
            "Use 'Longitudinal' or 'Perpendicular' for analytic channels."
        )

    return coeffs


def _make_time_index(num_taps: int, osr: int) -> np.ndarray:
    """Create the normalised time index vector.

    Equivalent to the C code:
        TimeIndex[0] = floor(NumChannelTaps/2) / OSR
        TimeIndex[i] = TimeIndex[i-1] - 1/OSR
    """
    centre = math.floor(num_taps / 2) / osr
    return np.array(
        [centre - i / osr for i in range(num_taps)],
        dtype=np.float64,
    )


def _bipolar(bits: np.ndarray) -> np.ndarray:
    """Map 0/1 bits to bipolar: 0 -> -1, 1 -> +1."""
    return 2.0 * bits.astype(np.float64) - 1.0


def longitudinal_channel(
    bits: np.ndarray,
    nd: float,
    num_taps: int = 201,
    osr: int = 10,
    sigma_jitter: float = 0.0,
    sigma_pulse_broad: float = 0.0,
) -> np.ndarray:
    """Longitudinal recording channel (AWGN only, no media noise).

    Args:
        bits: Binary bit array (0/1).
        nd: Normalised density.
        num_taps: Number of channel filter taps.
        osr: Oversampling rate.
        sigma_jitter: Jitter noise std-dev (ignored if 0).
        sigma_pulse_broad: Pulse broadening noise std-dev (ignored if 0).

    Returns:
        Channel output signal.
    """
    oss_len = len(bits) * osr
    bipolar = _bipolar(np.repeat(bits, osr))
    time_index = _make_time_index(num_taps, osr)
    coeffs = _compute_channel_coeffs(num_taps, time_index, nd, "longitudinal")
    output = non_causal_fir(bipolar, coeffs)
    return output[:oss_len]


def perpendicular_channel(
    bits: np.ndarray,
    nd: float,
    num_taps: int = 201,
    osr: int = 10,
    sigma_jitter: float = 0.0,
    sigma_pulse_broad: float = 0.0,
) -> np.ndarray:
    """Perpendicular recording channel (AWGN only, no media noise).

    Args:
        bits: Binary bit array (0/1).
        nd: Normalised density.
        num_taps: Number of channel filter taps.
        osr: Oversampling rate.
        sigma_jitter: Jitter noise std-dev (ignored if 0).
        sigma_pulse_broad: Pulse broadening noise std-dev (ignored if 0).

    Returns:
        Channel output signal.
    """
    oss_len = len(bits) * osr
    bipolar = _bipolar(np.repeat(bits, osr))
    time_index = _make_time_index(num_taps, osr)
    coeffs = _compute_channel_coeffs(num_taps, time_index, nd, "perpendicular")
    output = non_causal_fir(bipolar, coeffs)
    return output[:oss_len]


def hamr_channel(
    bipolar_bits: np.ndarray,
    osr: int,
    hamr_params: dict,
    sector_index: int = 0,
    nlts_k: np.ndarray | None = None,
    nlts_rho: np.ndarray | None = None,
    over_sampled_bit_length: float = 0.0,
) -> np.ndarray:
    """Simplified HAMR channel model.

    This is a streamlined approximation of the C Hamr() function.
    For production use, a full HAMR implementation with all magnetic
    and thermal effects should be integrated.

    The channel models the readback of bipolar transitions through
    a HAMR write head and GMR reader.

    Args:
        bipolar_bits: Oversampled bipolar signal (+1/-1).
        osr: Oversampling rate.
        hamr_params: Dict with HAMR parameters (sigma_t, T_peak, etc.).
        sector_index: Sector number (for NLTS tracking).
        nlts_k: NLTS K parameters per microtrack.
        nlts_rho: NLTS rho parameters per microtrack.
        over_sampled_bit_length: Bit period in oversampled domain.

    Returns:
        Readback signal.
    """
    length = len(bipolar_bits)
    output = np.zeros(length, dtype=np.float64)

    # Find transitions
    transitions: list[int] = []
    for i in range(1, length):
        if bipolar_bits[i] != bipolar_bits[i - 1]:
            transitions.append(i)

    if not transitions:
        return output

    # Simplified HAMR readback: model each transition as a Gaussian pulse
    sigma_t = hamr_params.get("sigma_t", 90.0)
    gmr_sigma = hamr_params.get("reader_sigma_r", 1000.0)
    hgm = hamr_params.get("hamr_hg", 1.6e6)

    # Effective Gaussian width combining writer and reader profiles
    sigma_eff = math.sqrt(sigma_t ** 2 + gmr_sigma ** 2)

    # Amplitude scaling factor
    amplitude = hgm * 1e-6  # Simple scaling

    # Build readback signal
    x = np.arange(length, dtype=np.float64)
    peak_index = transitions[0] if transitions else length // 2

    for t in transitions:
        # Transition contribution as a derivative-Gaussian pulse
        pulse = amplitude * (x - t) / (sigma_eff ** 2) * np.exp(
            -((x - t) ** 2) / (2 * sigma_eff ** 2)
        )
        output += pulse

    return output


def channel(
    bits: np.ndarray,
    channel_type: str = "Longitudinal",
    nd: float = 2.5,
    num_taps: int = 201,
    osr: int = 10,
    sigma_jitter: float = 0.0,
    sigma_pulse_broad: float = 0.0,
    hamr_params: dict | None = None,
    sector_index: int = 0,
    nlts_k: np.ndarray | None = None,
    nlts_rho: np.ndarray | None = None,
    over_sampled_bit_length: float = 0.0,
) -> np.ndarray:
    """Unified channel dispatcher.

    Args:
        bits: Binary bit array (0/1) or bipolar array (+1/-1).
        channel_type: "Longitudinal", "Perpendicular", or "Hamr".
        nd: Normalised density (for analytic channels).
        num_taps: Number of channel filter taps.
        osr: Oversampling rate.
        sigma_jitter: Jitter noise std-dev.
        sigma_pulse_broad: Pulse broadening noise std-dev.
        hamr_params: HAMR parameters dict (for HAMR channel).
        sector_index: Sector index for NLTS.
        nlts_k: NLTS K parameters.
        nlts_rho: NLTS rho parameters.
        over_sampled_bit_length: Bit period in oversampled domain.

    Returns:
        Channel output signal.

    Raises:
        ValueError: If channel_type is not recognised.
    """
    if channel_type in ("Hamr", "HAMR", "hamr"):
        if hamr_params is None:
            hamr_params = {}
        if len(bits.shape) == 0 or bits.dtype in (np.int8, np.int16, np.int32, np.int64):
            bipolar = _bipolar(bits)
        else:
            bipolar = bits.copy()
        return hamr_channel(
            bipolar, osr, hamr_params, sector_index,
            nlts_k, nlts_rho, over_sampled_bit_length,
        )
    elif channel_type in ("Longitudinal", "longitudinal"):
        return longitudinal_channel(
            bits, nd, num_taps, osr, sigma_jitter, sigma_pulse_broad,
        )
    elif channel_type in ("Perpendicular", "perpendicular"):
        return perpendicular_channel(
            bits, nd, num_taps, osr, sigma_jitter, sigma_pulse_broad,
        )
    else:
        raise ValueError(
            f"Unknown channel type '{channel_type}'. "
            "Use 'Longitudinal', 'Perpendicular', or 'Hamr'."
        )

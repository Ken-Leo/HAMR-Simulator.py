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
from channel.media_noise import media_noise_filter


def _compute_channel_coeffs(
    num_taps: int,
    time_index: np.ndarray,
    nd: float,
    osr: int,
    channel_type: str,
) -> np.ndarray:
    """Compute channel impulse response coefficients.

    Matches C ReadChannelCoeff computation in MagneticDisk.c:3726-3741.
    Coefficients are divided by OSR (not ND) to account for oversampling.

    Args:
        num_taps: Number of taps.
        time_index: Normalised time vector.
        nd: Normalised density.
        osr: Oversampling rate.
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
            coeffs[i] /= osr
    elif channel_type in ("longitudinal", "Longitudinal"):
        vp = math.sqrt(2.0 / PI)
        for i in range(num_taps):
            coeffs[i] = (
                vp * -8.0 * time_index[i] / nd ** 2
                * 1.0 / (1.0 + 4.0 * time_index[i] ** 2 / nd ** 2) ** 2
            )
            coeffs[i] /= osr
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
    """Longitudinal recording channel.

    Matches C LongPerp() → NonCausalFIR (clean) or MediaNoiseFilter (noisy).

    Args:
        bits: Binary bit array (0/1) of length PaddedSectorLength.
        nd: Normalised density.
        num_taps: Number of channel filter taps.
        osr: Oversampling rate.
        sigma_jitter: Jitter noise std-dev (% of T, 0 = disabled).
        sigma_pulse_broad: Pulse broadening noise std-dev (% of T, 0 = disabled).

    Returns:
        Channel output of length len(bits) * osr + num_taps // 2.
    """
    oss_len = len(bits) * osr
    bipolar = _bipolar(np.repeat(bits, osr))
    time_index = _make_time_index(num_taps, osr)
    coeffs = _compute_channel_coeffs(num_taps, time_index, nd, osr, "longitudinal")
    if sigma_jitter == 0.0 and sigma_pulse_broad == 0.0:
        output = non_causal_fir(bipolar, coeffs)
    else:
        output = media_noise_filter(
            bipolar, oss_len, coeffs, num_taps, time_index,
            nd, sigma_jitter, sigma_pulse_broad, osr,
            channel_type="Longitudinal",
        )
    return output


def perpendicular_channel(
    bits: np.ndarray,
    nd: float,
    num_taps: int = 201,
    osr: int = 10,
    sigma_jitter: float = 0.0,
    sigma_pulse_broad: float = 0.0,
) -> np.ndarray:
    """Perpendicular recording channel.

    Matches C LongPerp() → NonCausalFIR (clean) or MediaNoiseFilter (noisy).

    Args:
        bits: Binary bit array (0/1) of length PaddedSectorLength.
        nd: Normalised density.
        num_taps: Number of channel filter taps.
        osr: Oversampling rate.
        sigma_jitter: Jitter noise std-dev (% of T, 0 = disabled).
        sigma_pulse_broad: Pulse broadening noise std-dev (% of T, 0 = disabled).

    Returns:
        Channel output of length len(bits) * osr + num_taps // 2.
    """
    oss_len = len(bits) * osr
    bipolar = _bipolar(np.repeat(bits, osr))
    time_index = _make_time_index(num_taps, osr)
    coeffs = _compute_channel_coeffs(num_taps, time_index, nd, osr, "perpendicular")
    if sigma_jitter == 0.0 and sigma_pulse_broad == 0.0:
        output = non_causal_fir(bipolar, coeffs)
    else:
        output = media_noise_filter(
            bipolar, oss_len, coeffs, num_taps, time_index,
            nd, sigma_jitter, sigma_pulse_broad, osr,
            channel_type="Perpendicular",
        )
    return output


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


# ---------------------------------------------------------------------------
# Full HAMR Channel (physics-based, matches C Hamr() function)
# ---------------------------------------------------------------------------


class FullHamrChannel:
    """Full HAMR channel with PW50 initialization and microtrack physics.

    Wraps ``hamr_channel.hamr_channel()`` from the detailed physics module.
    Handles the initial single-transition PW50 calculation (C code lines
    3756-3839) and provides a simple call interface for the main simulation
    loop.

    Usage::

        ch = FullHamrChannel(config, nd, osr)
        # ... in main loop:
        output = ch(padded_bits, sector_index, disable_media_noise=False)
    """

    def __init__(
        self,
        config: "SimulatorConfig",  # type: ignore[name-defined]
        nd: float,
        osr: int,
    ) -> None:
        # Lazy import to avoid hard dependency if not using HAMR
        from channel.hamr_channel import (
            Mag_Param,
            Physical_Param,
            Reader_Param,
            hamr_channel as full_hamr,
        )

        self._full_hamr = full_hamr
        self._osr = osr
        self._nd = nd

        # Build parameter dataclasses from config
        self._mp = Mag_Param(
            sigma_t=config.hamr_sigma_t,
            T_Peak=config.hamr_t_peak,
            c=config.hamr_c,
            d=config.hamr_d,
            z0=config.hamr_z0,
            Hc=config.hamr_hc[:],
            Mr=config.hamr_mr[:],
            S=config.hamr_s[:],
            Hg=config.hamr_hg,
        )
        self._pp = Physical_Param(
            g=config.phys_g,
            d=config.phys_d,
            t=config.phys_t,
            y=config.phys_y,
            wt=config.phys_wt,
        )
        self._rp = Reader_Param(
            C=config.reader_c,
            gr=config.reader_gr,
            tr=config.reader_tr,
            wr=config.reader_wr,
            sigma_r=config.reader_sigma_r,
        )
        self._N = config.hamr_n

        # NLTS parameters
        self._k = config.nlts_k[:]
        self._rho = config.nlts_rho[:]

        # Flags
        self._temperature_variation = config.temperature_variation
        self._write_hd_cross_track_mov = config.write_hd_cross_track_mov
        self._hmd = config.hmd
        self._temperature_modulation = config.temperature_modulation
        self._modulated_peak_temp = config.hamr_t_peak  # default
        if config.temperature_modulation:
            self._modulated_peak_temp = 450.0  # C code default

        # Perform initial single-transition PW50 calculation
        self._pw50, self._over_sampled_bit_length, self._norm_factor = self._init_pw50()

    def _init_pw50(self) -> tuple[float, float, float]:
        """Run a single transition to determine PW50 and bit length.

        Matches C code lines 3756-3839.
        """
        # Create a single transition: -1 for first half, +1 for second half
        oss_len = 40960  # Large enough buffer for initial calculation
        os_bipolar = [-1] * (oss_len // 2) + [1] * (oss_len // 2)

        # X array: -1000 to 1000 nm sampled at 1 nm
        x = [float(i - 1000) for i in range(2001)]
        x_length = len(x)

        # Disable media noise for initial calculation
        k_dummy = [0.0] * self._N
        rho_dummy = [0.0] * self._N

        # Call with trans_to_calc_trans_param=1
        readback, pw50, peak_idx, norm_factor = self._full_hamr(
            mp=self._mp,
            pp=self._pp,
            rp=self._rp,
            N=self._N,
            oversampled_input_bits=os_bipolar,
            length_padded=oss_len,
            oversampled_bit_length=0.0,
            OSR=self._osr,
            num_sectors=-1,
            k=k_dummy,
            rho=rho_dummy,
            x=x,
            length=x_length,
            NLTS_compensation=0,
            num_nlts_influencing=0,
            temperature_variation=0,
            sigma_temp_variation=0.0,
            peak_temp_trunc_value=0.1,
            write_hd_cross_track_mov=0,
            max_write_hd_cr_tr_mov=0.0,
            mean_write_hd_cr_tr_mov=0.0,
            sigma_jitter=0.0,
            HMD=0,
            sigma_hmd_variation=0.0,
            num_ar_coeff=0,
            ar_model_coeff=[],
            temperature_modulation=0,
            modulated_peak_temp=self._modulated_peak_temp,
            trans_to_calc_trans_param=1,
            seed=-500,
        )

        # Compute oversampled bit length from PW50
        over_sampled_bit_length = pw50 / (self._nd * self._osr)

        return pw50, over_sampled_bit_length, norm_factor

    @property
    def pw50(self) -> float:
        return self._pw50

    @property
    def over_sampled_bit_length(self) -> float:
        return self._over_sampled_bit_length

    def __call__(
        self,
        padded_bits: np.ndarray,
        sector_index: int,
        disable_media_noise: bool = False,
    ) -> np.ndarray:
        """Run the full HAMR channel for one sector.

        Args:
            padded_bits: Padded bit array (0/1).
            sector_index: Current sector number.
            disable_media_noise: If True, set K/Rho to 0 (adaptive mode).

        Returns:
            Channel output signal.
        """
        # Convert to oversampled bipolar
        oss_len = len(padded_bits) * self._osr
        os_bits = np.repeat(padded_bits, self._osr)
        os_bipolar = (2 * os_bits.astype(np.float64) - 1.0).tolist()

        # X array: centered around 0, spanning ~10*PW50
        half_span = int(10 * self._pw50 / self._over_sampled_bit_length) + 1
        x_length = 2 * half_span + 1
        x = [
            i * self._over_sampled_bit_length - half_span * self._over_sampled_bit_length
            for i in range(x_length)
        ]

        # NLTS parameters
        k = [0.0] * self._N if disable_media_noise else self._k[:]
        rho = [0.0] * self._N if disable_media_noise else self._rho[:]

        readback, pw50, peak_idx, norm_factor = self._full_hamr(
            mp=self._mp,
            pp=self._pp,
            rp=self._rp,
            N=self._N,
            oversampled_input_bits=os_bipolar,
            length_padded=oss_len,
            oversampled_bit_length=self._over_sampled_bit_length,
            OSR=self._osr,
            num_sectors=sector_index,
            k=k,
            rho=rho,
            x=x,
            length=x_length,
            NLTS_compensation=0,
            num_nlts_influencing=0,
            temperature_variation=self._temperature_variation,
            sigma_temp_variation=2.0,
            peak_temp_trunc_value=20.0,
            write_hd_cross_track_mov=self._write_hd_cross_track_mov,
            max_write_hd_cr_tr_mov=20.0,
            mean_write_hd_cr_tr_mov=10.0,
            sigma_jitter=0.0,
            HMD=self._hmd,
            sigma_hmd_variation=0.0,
            num_ar_coeff=0,
            ar_model_coeff=[],
            temperature_modulation=self._temperature_modulation,
            modulated_peak_temp=self._modulated_peak_temp,
            trans_to_calc_trans_param=0,
            normalization_factor=self._norm_factor,
            seed=-500,
        )

        return np.array(readback, dtype=np.float64)


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

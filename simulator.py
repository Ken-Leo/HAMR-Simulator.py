"""HAMR Receiver Simulator.

Main orchestrator that ties together encoding, channel modelling,
filtering, equalization, detection, and decoding to simulate
a complete magnetic recording receiver.

Pipeline:
    UserBits -> Encoder -> Channel -> LPF -> Downsampler ->
    Equalizer -> Detector -> Decoder -> Error Counting

Based on the C main() function in MagneticDisk.c (lines 3356-4688).
"""

from __future__ import annotations


import csv
import math
import os
import pathlib
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from channel.channel import channel, FullHamrChannel
from channel.lpf import lpf
from channel.math_utils import LCG, CachedGaussian, gaussian_random, uniform_random
from decoders.mtr_6_7 import dec_6by7mtr_code
from decoders.rll_4_5 import dec_4by5rll_code
from decoders.tmtr_8_9 import dec_8by9tmtr_code
from encoders.mtr_6_7 import enc_6by7mtr_code
from encoders.rll_4_5 import enc_4by5rll_code
from encoders.tmtr_8_9 import enc_8by9tmtr_code
from equalizer_detector.detector import classical_viterbi, classical_sova
from equalizer_detector.constrained_detectors import (
    viterbi_6by7mtr_code,
    viterbi_8by9tmtr_code,
)
from equalizer_detector.equalizer import (
    adapt_equalizer,
    apply_equalizer,
    find_gpr_target,
)

# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class SimulatorConfig:
    """Simulation parameters.

    Matches the C main() variable declarations in MagneticDisk.c.
    """

    # General Parameters
    snr_db: list[float] = field(default_factory=lambda: [21.0])
    max_num_sectors: int = 10
    min_num_sectors: int = 10
    max_num_bit_err: int = 100
    osr: int = 10  # Oversampling rate
    sector_length: int = 4096
    pre_padding_length: int = 20
    post_padding_length: int = 20

    # Channel Options
    channel_type: str = "Hamr"  # "Longitudinal", "Perpendicular", "Hamr"
    user_density: float = 2.5  # Density before coding
    num_channel_taps: int = 201

    # Noise Parameters
    sigma_jitter: float = 0.0
    sigma_pulse_broad: float = 0.0

    # LPF
    num_lpf_taps: int = 201

    # Equalizer
    equalizer_type: str = "GPRTarget"  # "FixedPRTarget", "GPRTarget"
    num_eq_taps: int = 21
    num_eq_sectors: int = 10

    # PR Target
    pri_imp_res: list[float] = field(default_factory=lambda: [1, 1, -1, -1])

    # Detector
    detector_type: str = "Viterbi"  # "Viterbi", "SOVA"
    viterbi_delay: int = 20

    # Encoding
    use_encoding: bool = False
    encoder_type: str = "rll_4_5"  # "none", "rll_4_5", "mtr_6_7", "tmtr_8_9"

    # HAMR-specific
    hamr_n: int = 16  # Number of microtracks
    hamr_sigma_t: float = 90.0
    hamr_t_peak: float = 350.0
    hamr_c: float = 0.0
    hamr_d: float = 0.0
    hamr_z0: float = 0.0
    hamr_hc: list[float] = field(default_factory=lambda: [-1000.0, 1600000.0])
    hamr_mr: list[float] = field(default_factory=lambda: [-600.0, 1200000.0])
    hamr_s: list[float] = field(default_factory=lambda: [0.0003, 0.5])
    hamr_hg: float = 1.6e6
    phys_g: float = 100.0
    phys_d: float = 19.0
    phys_t: float = 2.0
    phys_y: float = 28.0
    phys_wt: float = 160.0
    reader_c: float = 1.0
    reader_gr: float = 5.0
    reader_tr: float = 1.0
    reader_wr: float = 1000.0
    reader_sigma_r: float = 1000.0

    # NLTS
    nlts_k: list[float] = field(default_factory=lambda: [4000.0] * 16)
    nlts_rho: list[float] = field(default_factory=lambda: [2.0] * 16)

    # Temperature effects
    temperature_modulation: int = 0
    temperature_variation: int = 0
    write_hd_cross_track_mov: int = 0
    hmd: int = 0

    # Output
    results_dir: str = "results"
    figures_dir: str = "figures"

    def __post_init__(self) -> None:
        """Set default values if None."""
        if self.snr_db is None:
            self.snr_db = [21.0]
        if self.pri_imp_res is None:
            self.pri_imp_res = [1, 1, -1, -1]
        if self.hamr_hc is None:
            self.hamr_hc = [-1000.0, 1600000.0]
        if self.hamr_mr is None:
            self.hamr_mr = [-600.0, 1200000.0]
        if self.hamr_s is None:
            self.hamr_s = [0.0003, 0.5]
        if self.nlts_k is None:
            self.nlts_k = [4000.0] * 16
        if self.nlts_rho is None:
            self.nlts_rho = [2.0] * 16


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_code_params(config: SimulatorConfig) -> tuple[int, int, float]:
    """Compute sector lengths and code rate based on encoding config.

    Returns:
        ``(user_sector_length, sector_length, code_rate)``

    ``sector_length`` is the coded sector length used throughout the
    simulation pipeline.  ``user_sector_length`` is only used for
    BER/SER comparison (decoder output is compared against the first
    ``user_sector_length`` bits of the generated random ``user_bits``).

    The sector_length is passed to the encoder which may further
    adjust it to satisfy its internal constraint (e.g. 4Z+1 for RLL).
    """
    if not config.use_encoding:
        return config.sector_length, config.sector_length, 1.0

    if config.encoder_type in ("rll_4_5",):
        code_rate = 4.0 / 5.0
    elif config.encoder_type in ("mtr_6_7",):
        code_rate = 6.0 / 7.0
    elif config.encoder_type in ("tmtr_8_9",):
        code_rate = 8.0 / 9.0
    else:
        return 1, 1, 1.0

    sector_length = config.sector_length
    user_sector_length = max(1, int(sector_length * code_rate))
    return user_sector_length, sector_length, code_rate


def _encoder_fn(config: SimulatorConfig):
    """Return the encoder function based on config."""
    if not config.use_encoding:
        return None
    encoders = {
        "rll_4_5": enc_4by5rll_code,
        "mtr_6_7": enc_6by7mtr_code,
        "tmtr_8_9": enc_8by9tmtr_code,
    }
    return encoders.get(config.encoder_type)


def _decoder_fn(config: SimulatorConfig):
    """Return the decoder function based on config."""
    if not config.use_encoding:
        return None
    decoders = {
        "rll_4_5": dec_4by5rll_code,
        "mtr_6_7": dec_6by7mtr_code,
        "tmtr_8_9": dec_8by9tmtr_code,
    }
    return decoders.get(config.encoder_type)


def _bipolar(bits: np.ndarray) -> np.ndarray:
    """Map 0/1 bits to bipolar: 0 -> -1, 1 -> +1."""
    return 2.0 * bits.astype(np.float64) - 1.0


def _causal_fir_simple(data: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Causal FIR filter producing len(data) output samples.

    Computes the middle portion of the full convolution to match
    the causal FIR behavior used in the C code for PR target shaping.

    Args:
        data: Input signal.
        h: Filter coefficients (e.g., PR target).

    Returns:
        Output of same length as input.
    """
    pad_len = len(h) - 1
    total_length = len(data) + 2 * pad_len
    padded = np.zeros(total_length, dtype=np.float64)
    padded[pad_len: pad_len + len(data)] = data[: len(data)]

    output = np.zeros(len(data), dtype=np.float64)
    for i in range(pad_len, len(data) + pad_len):
        channel_output = 0.0
        for j in range(len(h)):
            channel_output += h[j] * padded[i - j]
        output[i - pad_len] = channel_output

    return output


# ---------------------------------------------------------------------------
# Main simulation
# ---------------------------------------------------------------------------


def run_simulation(config: SimulatorConfig) -> dict[str, Any]:
    """Run the full HAMR receiver simulation.

    Pipeline:
        UserBits -> Encoder -> Channel -> LPF -> Downsampler ->
        Equalizer -> Detector -> Decoder -> Error Counting

    Args:
        config: Simulation configuration.

    Returns:
        Dictionary with keys:
        - snr_values: list of SNR values tested
        - results: list of per-SNR result dicts
        - ber_per_snr: BER at each SNR point
        - ser_per_snr: SER at each SNR point
    """
    pri_imp_res = np.array(config.pri_imp_res, dtype=np.float64)
    nd = config.user_density / (4.0 / 5.0) if config.use_encoding and config.encoder_type == "rll_4_5" else config.user_density

    # Compute sector parameters
    user_sector_length, sector_length, code_rate = _compute_code_params(config)

    # Prepare padded sector length.
    # For rate < 1 encoders the encoded output is larger than sector_length.
    # The encoder may adjust sector_length upward by at most +1 (RLL: 4096->4097).
    # Upper bound: ceil((sector_length) / code_rate) + 1
    if code_rate < 1:
        encoded_len = int(math.ceil(sector_length / code_rate)) + 1
    else:
        encoded_len = sector_length
    padded_sector_length = encoded_len + config.pre_padding_length + config.post_padding_length

    # Compute signal power for noise calculation
    # Signal power = S^2 * E where E = sum over PR target autocorrelation
    # For bipolar: S = 2 (mapped to +/-1)
    # E = sum_j sum_k pri_imp_res[j] * pri_imp_res[k] * R(j-k)
    # where R(lag) is autocorrelation of channel coeffs (approximated as 1 for now)
    signal_energy = float(np.sum(pri_imp_res ** 2))
    signal_power = 4.0 * signal_energy  # S=2 for bipolar

    # HAMR parameters dict
    hamr_params: dict[str, float] = {
        "sigma_t": config.hamr_sigma_t,
        "T_peak": config.hamr_t_peak,
        "c": config.hamr_c,
        "d": config.hamr_d,
        "z0": config.hamr_z0,
        "hc_a": config.hamr_hc[0],
        "hc_b": config.hamr_hc[1],
        "mr_a": config.hamr_mr[0],
        "mr_b": config.hamr_mr[1],
        "s_a": config.hamr_s[0],
        "s_b": config.hamr_s[1],
        "hg": config.hamr_hg,
        "reader_sigma_r": config.reader_sigma_r,
    }

    # NLTS arrays
    nlts_k = np.array(config.nlts_k, dtype=np.float64)
    nlts_rho = np.array(config.nlts_rho, dtype=np.float64)

    # Equalizer coefficients
    eq_coeff = np.zeros(config.num_eq_taps, dtype=np.float64)

    # Encoder/decoder functions
    enc_fn = _encoder_fn(config)
    dec_fn = _decoder_fn(config)

    # Initialize full HAMR channel if selected (does PW50 calculation)
    full_hamr: FullHamrChannel | None = None
    over_sampled_bit_length = 0.0
    if config.channel_type.lower() == "hamr":
        full_hamr = FullHamrChannel(config, nd, config.osr)
        over_sampled_bit_length = full_hamr.over_sampled_bit_length
        print(f"HAMR channel initialized: PW50={full_hamr.pw50:.1f} nm, "
              f"OSBL={over_sampled_bit_length:.2f} nm")

    results: list[dict[str, Any]] = []

    start_time = time.time()

    for snr_idx, snr_db in enumerate(config.snr_db):
        # Convert SNR (dB) to noise sigma
        # noise_sigma = sqrt(10^(-SNR/10) * signal_power)
        # Match C: NoiseSigma = sqrt(OSR) * 10^(-SNR/20) * 2 * sqrt(ND/2)
        noise_sigma = (
            math.sqrt(config.osr)
            * math.pow(10, -snr_db / 20.0)
            * 2.0
            * math.sqrt(nd / 2.0)
        )

        num_sectors = 0
        total_bit_errors = 0
        num_error_sectors = 0
        error_sectors: list[dict[str, Any]] = []

        # RNG seeds (matching C: idum1=-500, idum2=-600)
        lcg_bits = uniform_random(-500)
        lcg_noise = uniform_random(-600)
        cached_gauss = CachedGaussian(lcg_noise)  # matches C static cache

        # Equalizer adaptation mode (use high SNR first)
        eq_adapted = False
        gpr_target: np.ndarray | None = None

        # Phase 1: Equalizer adaptation (if GPR target)
        if config.equalizer_type == "GPRTarget":
            concat_input: list[np.ndarray] = []
            concat_output: list[np.ndarray] = []

            adapt_snr = 50.0  # High SNR for clean adaptation
            adapt_noise_sigma = (
                math.sqrt(config.osr)
                * math.pow(10, -adapt_snr / 20.0)
                * 2.0
                * math.sqrt(nd / 2.0)
            )

            for adapt_sector in range(config.num_eq_sectors):
                # Generate bits (enough to cover encoder's adjusted sector_length)
                user_bits = np.array(
                    [int(lcg_bits.random() > 0.5) for _ in range(sector_length + 4)],
                    dtype=np.int64,
                )

                # Encode
                if enc_fn is not None:
                    user_bits[0] = 0  # Force first bit to 0
                    encoded = enc_fn(user_bits, sector_length)
                else:
                    encoded = user_bits[:sector_length].copy()

                # Pad
                padded = np.zeros(padded_sector_length, dtype=np.int64)
                padded[
                    config.pre_padding_length: config.pre_padding_length + len(encoded)
                ] = encoded

                # Channel: pass unipolar bits; channel() does its own bipolar+oversampling
                # Matches C: NonCausalFIR(OSBipolarBits, OSSectorLength, ...)
                oss_len = len(padded) * config.osr
                if full_hamr is not None:
                    # Use full physics-based HAMR channel
                    ch_output = full_hamr(padded, adapt_sector,
                                          disable_media_noise=True)
                else:
                    # C main(): during adaptation (j<0), LongPerp passes
                    # sigma_jitter=0, sigma_pulse_broad=0 to disable media noise
                    ch_output = channel(
                        padded,
                        config.channel_type,
                        nd,
                        config.num_channel_taps,
                        config.osr,
                        0.0,  # sigma_jitter = 0 (adaptive mode)
                        0.0,  # sigma_pulse_broad = 0 (adaptive mode)
                        hamr_params,
                        adapt_sector,
                        nlts_k,
                        nlts_rho,
                        over_sampled_bit_length,
                    )

                # Add noise: C code adds noise only to inner region
                # from PrePaddingLength*OSR to OSSectorLength - PostPaddingLength*OSR
                # of the first OSSectorLength elements (not the FIR tail).
                inner_start = config.pre_padding_length * config.osr
                inner_count = oss_len - (config.pre_padding_length +
                                         config.post_padding_length) * config.osr
                # Generate noise one at a time for the inner region (matches C count)
                noise_vec = np.array(
                    [cached_gauss() for _ in range(inner_count)],
                    dtype=np.float64,
                )
                ch_output[inner_start: inner_start + inner_count] += noise_vec * adapt_noise_sigma

                # LPF: C code calls LPF(ChannelOutput, OSSectorLength, ...)
                # meaning LPF only sees first OSSectorLength elements of channel output
                ch_for_lpf = ch_output[:oss_len]
                lpf_output = lpf(ch_for_lpf, config.num_lpf_taps - 1, 1.0 / config.osr)

                # Downsample: C code does DSOutput[i] = LPFOutput[i * OSR]
                ds_output = lpf_output[:: config.osr][:padded_sector_length]

                # Store for GPR target computation (bipolar, matching C code)
                concat_input.append(
                    _bipolar(padded[config.pre_padding_length: config.pre_padding_length + len(encoded)])
                )
                concat_output.append(ds_output[config.pre_padding_length: config.pre_padding_length + len(encoded)])

            # Compute GPR target (fallback - LMS uses desired_signal instead)
            if concat_input:
                full_input = np.concatenate(concat_input)
                full_output = np.concatenate(concat_output)
                gpr_target, eq_coeff = find_gpr_target(
                    full_output, full_input, config.num_eq_taps,
                    gpr_target_length=len(config.pri_imp_res),
                )
                eq_adapted = True

        # Phase 1b: LMS equalizer adaptation for FixedPRTarget
        # (C code uses AdaptEqualizer for both FixedPRTarget and GPRTarget)
        if config.equalizer_type == "FixedPRTarget" and config.num_eq_sectors > 0:
            adapt_snr = 50.0  # High SNR for clean adaptation
            adapt_noise_sigma = (
                math.sqrt(config.osr)
                * math.pow(10, -adapt_snr / 20.0)
                * 2.0
                * math.sqrt(nd / 2.0)
            )

            pri_imp_res = np.array(config.pri_imp_res, dtype=np.float64)
            eq_coeff = np.zeros(config.num_eq_taps, dtype=np.float64)
            start_flag = 1

            for adapt_sector in range(config.num_eq_sectors):
                user_bits = np.array(
                    [int(lcg_bits.random() > 0.5) for _ in range(sector_length + 4)],
                    dtype=np.int64,
                )
                if enc_fn is not None:
                    user_bits[0] = 0
                    encoded = enc_fn(user_bits, sector_length)
                else:
                    encoded = user_bits[:sector_length].copy()

                padded = np.zeros(padded_sector_length, dtype=np.int64)
                padded[config.pre_padding_length: config.pre_padding_length + len(encoded)] = encoded
                oss_len = len(padded) * config.osr

                if full_hamr is not None:
                    ch_output = full_hamr(padded, adapt_sector,
                                          disable_media_noise=True)
                else:
                    ch_output = channel(
                        padded, config.channel_type, nd, config.num_channel_taps,
                        config.osr, 0.0, 0.0,
                    )
                inner_start = config.pre_padding_length * config.osr
                inner_count = oss_len - (config.pre_padding_length +
                                         config.post_padding_length) * config.osr
                noise_vec = np.array(
                    [cached_gauss() for _ in range(inner_count)],
                    dtype=np.float64,
                )
                ch_output[inner_start: inner_start + inner_count] += noise_vec * adapt_noise_sigma

                lpf_output = lpf(ch_output[:oss_len], config.num_lpf_taps - 1, 1.0 / config.osr)
                ds_output = lpf_output[:: config.osr][:padded_sector_length]

                # Clean bipolar bits for desired signal
                clean_bipolar = _bipolar(padded)[config.pre_padding_length: config.pre_padding_length + len(encoded)]
                ds_inner = ds_output[config.pre_padding_length: config.pre_padding_length + len(encoded)]

                mse, lmse = adapt_equalizer(
                    pri_imp_res, eq_coeff, config.num_eq_taps,
                    clean_bipolar, ds_inner, len(clean_bipolar),
                    start_flag=start_flag,
                )
                start_flag = 0

            eq_adapted = True
            print(f"  FixedPRTarget LMS adapted: MSE={mse:.4f}, LMS={lmse:.4f}")

        # Phase 2: Main simulation loop
        for _ in range(config.max_num_sectors):
            num_sectors += 1

            # Generate random user bits.
            # The encoder truncates user_bits to its adjusted sector_length
            # (e.g. 4Z+1 for RLL).  Generate enough bits to cover the
            # adjusted length (up to sector_length + 3 for the worst case).
            user_bits = np.array(
                [int(lcg_bits.random() > 0.5) for _ in range(sector_length + 4)],
                dtype=np.int64,
            )

            # Encode
            if enc_fn is not None:
                user_bits[0] = 0
                encoded = enc_fn(user_bits, sector_length)
            else:
                encoded = user_bits[:sector_length].copy()

            # Pad with zeros
            padded = np.zeros(padded_sector_length, dtype=np.int64)
            padded[
                config.pre_padding_length: config.pre_padding_length + len(encoded)
            ] = encoded

            # Channel: pass unipolar bits; channel() does its own bipolar+oversampling
            oss_len = len(padded) * config.osr
            if full_hamr is not None:
                # Use full physics-based HAMR channel
                ch_output = full_hamr(padded, num_sectors,
                                      disable_media_noise=False)
            else:
                ch_output = channel(
                    padded,
                    config.channel_type,
                    nd,
                    config.num_channel_taps,
                    config.osr,
                    config.sigma_jitter,
                    config.sigma_pulse_broad,
                    hamr_params,
                    num_sectors,
                    nlts_k,
                    nlts_rho,
                    over_sampled_bit_length,
                )

            # Add AWGN noise: C code adds noise only to inner region
            inner_start = config.pre_padding_length * config.osr
            inner_count = oss_len - (config.pre_padding_length +
                                     config.post_padding_length) * config.osr
            noise_vec = np.array(
                [gaussian_random(lcg_noise) for _ in range(inner_count)],
                dtype=np.float64,
            )
            ch_output[inner_start: inner_start + inner_count] += noise_vec * noise_sigma

            # LPF: C code calls LPF(ChannelOutput, OSSectorLength, ...)
            ch_for_lpf = ch_output[:oss_len]
            lpf_output = lpf(ch_for_lpf, config.num_lpf_taps - 1, 1.0 / config.osr)

            # Downsample: C code does DSOutput[i] = LPFOutput[i * OSR]
            ds_output = lpf_output[:: config.osr][:padded_sector_length]

            # Equalization
            if config.equalizer_type == "FixedPRTarget":
                equalized = apply_equalizer(ds_output, eq_coeff, config.num_eq_taps)
            elif config.equalizer_type == "GPRTarget":
                target = gpr_target if gpr_target is not None else pri_imp_res
                eq_coeff_gpr = eq_coeff if eq_coeff is not None else np.zeros(config.num_eq_taps)
                equalized = apply_equalizer(ds_output, eq_coeff_gpr, config.num_eq_taps)
            else:
                equalized = ds_output.copy()

            # Detection
            det_sector_length = len(equalized)
            if config.detector_type == "Viterbi":
                # Use constrained Viterbi if MTR/TMTR encoding is active
                if config.use_encoding:
                    if config.encoder_type == "mtr_6_7":
                        detected_hard, detected_soft = viterbi_6by7mtr_code(
                            config.viterbi_delay, equalized, config.pre_padding_length,
                            det_sector_length,
                            gpr_target if config.equalizer_type == "GPRTarget" else pri_imp_res,
                        )
                    elif config.encoder_type == "tmtr_8_9":
                        detected_hard, detected_soft = viterbi_8by9tmtr_code(
                            config.viterbi_delay, equalized, config.pre_padding_length,
                            det_sector_length,
                            gpr_target if config.equalizer_type == "GPRTarget" else pri_imp_res,
                        )
                    else:
                        detected_hard, detected_soft = classical_viterbi(
                            config.viterbi_delay, equalized, det_sector_length,
                            gpr_target if config.equalizer_type == "GPRTarget" else pri_imp_res,
                        )
                else:
                    detected_hard, detected_soft = classical_viterbi(
                        config.viterbi_delay, equalized, det_sector_length,
                        gpr_target if config.equalizer_type == "GPRTarget" else pri_imp_res,
                    )
            elif config.detector_type == "SOVA":
                # Currently, constrained SOVA is not implemented, fallback to classical
                detected_hard, detected_soft, _ = classical_sova(
                    config.viterbi_delay, equalized, det_sector_length,
                    gpr_target if config.equalizer_type == "GPRTarget" else pri_imp_res,
                    noise_sigma,
                )
            else:
                raise ValueError(f"Unknown detector type: {config.detector_type}")

            # Decode: the Viterbi/SOVA output has det_sector_length elements
            # (matching the equalized signal length).  Pass the full output
            # to the decoder with pre_padding=0.
            # Decode
            if dec_fn is not None:
                decoded, invalid_cw = dec_fn(
                    detected_hard, 0, det_sector_length
                )
            else:
                decoded = detected_hard.copy()

            # Count errors
            # The detector output corresponds to the full padded signal.
            # Only compare the inner region (excluding padding), matching
            # the C code which compares ViterbiHardOutput[i] with PaddedBits[i]
            # for i in [PrePaddingLength, PrePaddingLength + SectorLength).
            det_inner = detected_hard[
                config.pre_padding_length: config.pre_padding_length + len(encoded)
            ]
            compare_len = min(len(det_inner), len(encoded))
            bit_errors = int(np.sum(det_inner[:compare_len] != encoded[:compare_len]))
            total_bit_errors += bit_errors

            if bit_errors > 0:
                num_error_sectors += 1
                error_sectors.append({
                    "sector": num_sectors,
                    "bit_errors": bit_errors,
                    "snr_db": snr_db,
                })

            # Early termination
            if num_sectors >= config.max_num_sectors:
                break
            if (total_bit_errors >= config.max_num_bit_err
                    and num_sectors >= config.min_num_sectors):
                break

        # Compute BER and SER
        ber = (
            total_bit_errors
            / (num_sectors * user_sector_length)
            if num_sectors > 0
            else 0.0
        )
        ser = num_error_sectors / num_sectors if num_sectors > 0 else 0.0

        result = {
            "snr_db": snr_db,
            "ber": ber,
            "ser": ser,
            "num_sectors": num_sectors,
            "num_bit_errors": total_bit_errors,
            "num_error_sectors": num_error_sectors,
        }
        results.append(result)

        print(
            f"SNR={snr_db:.1f}dB: BER={ber:.2e} SER={ser:.4f} "
            f"Sectors={num_sectors} BitErrs={total_bit_errors}"
        )

    elapsed = time.time() - start_time

    # Save results
    _save_results(results, config, elapsed)

    # Generate BER vs SNR figure
    _plot_ber_snr(results, config)

    return {
        "snr_values": [r["snr_db"] for r in results],
        "results": results,
        "ber_per_snr": [r["ber"] for r in results],
        "ser_per_snr": [r["ser"] for r in results],
        "elapsed_seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def _save_results(results: list[dict], config: SimulatorConfig,
                  elapsed: float) -> None:
    """Save simulation summary to CSV."""
    results_path = pathlib.Path(config.results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    csv_path = results_path / "simulation_summary.csv"
    fieldnames = [
        "snr_db", "ber", "ser", "num_sectors",
        "num_bit_errors", "num_error_sectors",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow(r)

    print(f"Results saved to {csv_path}")


def _plot_ber_snr(results: list[dict], config: SimulatorConfig) -> None:
    """Plot BER vs SNR and save to figures directory."""
    try:
        import matplotlib
        matplotlib.use("Agg")  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available; skipping BER plot.")
        return

    fig_dir = pathlib.Path(config.figures_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    snr_vals = [r["snr_db"] for r in results]
    ber_vals = [r["ber"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Replace 0 BER with minimum detectable BER for log-scale display
    min_detectable = 1.0
    for r in results:
        nbits = r["num_sectors"] * config.sector_length
        detectable = 0.5 / nbits if nbits > 0 else 1.0
        if detectable < min_detectable:
            min_detectable = detectable
    min_detectable = max(min_detectable, 1e-15)
    plot_ber = [max(r["ber"], min_detectable * 0.1) for r in results]

    ax.semilogy(snr_vals, plot_ber, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title("HAMR Receiver Simulation: BER vs SNR")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)

    # Add channel type annotation
    ax.text(
        0.02, 0.98,
        f"Channel: {config.channel_type}\n"
        f"Detector: {config.detector_type}\n"
        f"Encoding: {config.encoder_type if config.use_encoding else 'None'}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    fig.savefig(fig_dir / "ber_vs_snr.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"BER plot saved to {fig_dir / 'ber_vs_snr.png'}")

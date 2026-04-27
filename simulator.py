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

from channel.channel import channel
from channel.lpf import lpf
from channel.math_utils import LCG, gaussian_random, uniform_random
from decoders.mtr_6_7 import dec_6by7mtr_code
from decoders.rll_4_5 import dec_4by5rll_code
from decoders.tmtr_8_9 import dec_8by9tmtr_code
from encoders.mtr_6_7 import enc_6by7mtr_code
from encoders.rll_4_5 import enc_4by5rll_code
from encoders.tmtr_8_9 import enc_8by9tmtr_code
from equalizer_detector.detector import classical_viterbi, classical_sova
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
        return 1, 1, 1.0

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

    results: list[dict[str, Any]] = []
    over_sampled_bit_length = 0.0  # Will be computed for HAMR channel

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
                    config.pre_padding_length: config.pre_padding_length + encoded_len
                ] = encoded

                # Bipolar
                bipolar = _bipolar(padded)

                # Channel
                ch_output = channel(
                    bipolar,
                    config.channel_type,
                    nd,
                    config.num_channel_taps,
                    config.osr,
                    config.sigma_jitter,
                    config.sigma_pulse_broad,
                    hamr_params,
                    adapt_sector,
                    nlts_k,
                    nlts_rho,
                    over_sampled_bit_length,
                )

                # Add noise
                oss_len = len(bipolar) * config.osr
                noise = np.array(
                    [
                        gaussian_random(lcg_noise)
                        for _ in range(oss_len)
                    ],
                    dtype=np.float64,
                )
                ch_output = ch_output + noise * adapt_noise_sigma

                # LPF
                lpf_output = lpf(ch_output, config.num_lpf_taps - 1, 1.0 / config.osr)

                # Downsample: extract exactly encoded_len elements.
                # The LPF extends the signal, so we compute stop from encoded_len * OSR.
                ds_start = config.pre_padding_length * config.osr
                ds_stop = ds_start + encoded_len * config.osr
                ds_output = lpf_output[ds_start: ds_stop: config.osr]

                # Store for GPR target computation
                concat_input.append(padded[config.pre_padding_length: config.pre_padding_length + encoded_len])
                concat_output.append(ds_output[config.pre_padding_length: config.pre_padding_length + encoded_len])

            # Compute GPR target
            if concat_input:
                full_input = np.concatenate(concat_input)
                full_output = np.concatenate(concat_output)
                gpr_target, eq_coeff = find_gpr_target(
                    full_output, full_input, config.num_eq_taps,
                    gpr_target_length=len(config.pri_imp_res),
                )
                eq_adapted = True

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
                config.pre_padding_length: config.pre_padding_length + encoded_len
            ] = encoded
            padded_bipolar = _bipolar(padded)

            # Channel
            ch_output = channel(
                padded_bipolar,
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

            # Add AWGN noise
            ch_noise = np.array(
                [gaussian_random(lcg_noise) for _ in range(len(ch_output))],
                dtype=np.float64,
            )
            ch_output = ch_output + ch_noise * noise_sigma

            # LPF
            lpf_output = lpf(ch_output, config.num_lpf_taps - 1, 1.0 / config.osr)

            # Downsampling: extract exactly encoded_len elements.
            ds_start = config.pre_padding_length * config.osr
            ds_stop = ds_start + encoded_len * config.osr
            ds_output = lpf_output[ds_start: ds_stop: config.osr]

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
                detected_hard, detected_soft = classical_viterbi(
                    config.viterbi_delay, equalized, det_sector_length,
                    gpr_target if config.equalizer_type == "GPRTarget" else pri_imp_res,
                )
            elif config.detector_type == "SOVA":
                detected_hard, detected_soft = classical_sova(
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
            compare_len = min(len(decoded), user_sector_length, len(user_bits))
            bit_errors = int(np.sum(decoded[:compare_len] != user_bits[:compare_len]))
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
    ax.semilogy(snr_vals, ber_vals, "bo-", linewidth=2, markersize=8)
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

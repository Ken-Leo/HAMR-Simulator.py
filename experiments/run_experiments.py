"""HAMR Receiver Simulator -- Comprehensive Experiment Suite.

Runs a battery of experiments to validate the Python translation,
measure performance, and collect data for the academic test report.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Add project root to path
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Imports from project modules
# ---------------------------------------------------------------------------
from channel.channel import channel
from channel.lpf import lpf
from channel.fir import non_causal_fir, causal_fir
from channel.math_utils import LCG, gaussian_random, uniform_random
from encoders.rll_4_5 import enc_4by5rll_code, Codewords as rll_cw
from encoders.mtr_6_7 import enc_6by7mtr_code, Codewords as mtr_cw
from encoders.tmtr_8_9 import enc_8by9tmtr_code, Codewords as tmtr_cw
from decoders.rll_4_5 import dec_4by5rll_code
from decoders.mtr_6_7 import dec_6by7mtr_code
from decoders.tmtr_8_9 import dec_8by9tmtr_code
from equalizer_detector.viterbi import classical_viterbi
from equalizer_detector.sova import classical_sova
from equalizer_detector.equalizer import (
    adapt_equalizer,
    apply_equalizer,
    find_gpr_target,
)

RESULTS = pathlib.Path(__file__).parent / "results"
ASSETS = pathlib.Path(__file__).parent / "assets"
RESULTS.mkdir(parents=True, exist_ok=True)
ASSETS.mkdir(parents=True, exist_ok=True)

# Valid sector lengths for each encoder type (4Z+1, 6Z+1, 8Z+1)
SECTOR_RLL = 101
SECTOR_MTR = 211
SECTOR_TMTR = 361

# Small sector lengths for fast iteration in BER experiments
SECTOR_FAST = 101

# 512-byte sector (4096 bits), adjusted to 4Z+1 for RLL(4/5) constraint
SECTOR_512B = 4097  # 4×1024+1, ≈512 bytes

PRI_IMP = np.array([1, 1, -1, -1], dtype=np.float64)  # EPR4 (matches C code)
NUM_EQ_TAPS = 21


def save_fig(name: str, dpi: int = 200) -> None:
    plt.tight_layout()
    plt.savefig(ASSETS / name, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  -> saved {ASSETS / name}")


def make_bipolar(bits: np.ndarray) -> np.ndarray:
    """0/1 -> bipolar (+1/-1)."""
    return 2.0 * bits.astype(np.float64) - 1.0


def _causal_fir_simple(data: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Causal FIR filter producing len(data) output samples (PR target shaping)."""
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


def run_single_sector_sweep(
    ch_type: str,
    encoder_fn,
    sector_len: int,
    snr_db: float,
    num_sectors: int = 100,
    eq_coeff_init: np.ndarray | None = None,
    gpr_target_init: np.ndarray | None = None,
) -> tuple[float, float, dict]:
    """Run one SNR point, return (ber, ser, info_dict).

    Full pipeline matches C GPRTarget mode:
    1. One-time GPR target computation (done externally)
    2. Fixed FIR equalizer (no per-sector LMS)
    3. GPR target passed to Viterbi detector
    """
    PRE = 20  # preamble length (matches C PREAMBLE_LENGTH)
    OSR = 10
    lcg_bits = uniform_random(-500)
    lcg_noise = uniform_random(-600)
    noise_sigma = math.sqrt(OSR) * 10 ** (-snr_db / 20) * 2 * math.sqrt(2.5 / 2)
    total_errors = 0
    total_bits = 0
    error_sectors = 0

    # Use provided GPR target/coeffs, or fall back to PR target
    gpr_target = gpr_target_init if gpr_target_init is not None else PRI_IMP
    eq_coeff = eq_coeff_init if eq_coeff_init is not None else np.zeros(NUM_EQ_TAPS)

    for s in range(num_sectors):
        bits = np.array(
            [int(lcg_bits.random() > 0.5) for _ in range(sector_len)],
            dtype=np.int64,
        )
        bits[0] = 0

        if encoder_fn is not None:
            bits[0] = 0
            encoded = encoder_fn(bits, sector_len)
        else:
            encoded = bits[:sector_len].copy()

        encoded_len = len(encoded)
        oss_len = (PRE + encoded_len + PRE) * OSR

        # Pad with pre/post padding (matches C code)
        padded = np.zeros(PRE + encoded_len + PRE, dtype=np.int64)
        padded[PRE: PRE + encoded_len] = encoded
        bipolar = make_bipolar(padded)

        # Channel
        ch_out = channel(bipolar, ch_type, 2.5, 201, OSR, 0.0, 0.0, {}, s)

        # Add noise to inner region only (matches C)
        noise = np.array([gaussian_random(lcg_noise) for _ in range(oss_len)],
                         dtype=np.float64)
        inner_start = PRE * OSR
        inner_stop = oss_len - PRE * OSR
        ch_out[inner_start: inner_stop] += noise[inner_start: inner_stop] * noise_sigma

        # LPF: C code calls LPF(ChannelOutput, OSSectorLength, ...)
        ch_for_lpf = ch_out[:oss_len]
        lpf_out = lpf(ch_for_lpf, 20, 1.0 / OSR)

        # Downsample: C code does DSOutput[i] = LPFOutput[i * OSR]
        ds = lpf_out[::OSR][:PRE + encoded_len + PRE]

        # Apply fixed GPR equalizer (no per-sector LMS - matches C GPRTarget mode)
        eq_out = apply_equalizer(ds[PRE: PRE + encoded_len], eq_coeff, NUM_EQ_TAPS)

        # Viterbi detection with GPR target
        detected, _ = classical_viterbi(20, eq_out, encoded_len, gpr_target)

        # Decode if needed
        if encoder_fn is not None:
            decoder_fn, _ = _get_decoder_for_encoder(encoder_fn)
            decoded, _ = decoder_fn(detected, 0, encoded_len)
            compare = min(len(decoded), sector_len)
            errors = int(np.sum(decoded[:compare] != bits[:compare]))
        else:
            compare = encoded_len
            errors = int(np.sum(detected[:compare] != bits[:compare]))

        total_errors += errors
        total_bits += compare
        if errors > 0:
            error_sectors += 1

    ber = total_errors / total_bits if total_bits > 0 else 1.0
    ser = error_sectors / num_sectors if num_sectors > 0 else 1.0
    return ber, ser, {
        "total_errors": total_errors,
        "total_bits": total_bits,
        "error_sectors": error_sectors,
    }


def run_single_sector_sweep_with_gpr(
    ch_type: str,
    encoder_fn,
    sector_len: int,
    snr_db: float,
    num_sectors: int = 100,
    eq_coeff_init: np.ndarray | None = None,
    gpr_target_init: np.ndarray | None = None,
    decoder_fn=None,
) -> tuple[float, float, dict]:
    """Run one SNR point with custom decoder, return (ber, ser, info_dict).

    Extended version of run_single_sector_sweep that accepts a custom decoder
    function for experiments with multiple encoder types.
    """
    PRE = 20
    OSR = 10
    lcg_bits = uniform_random(-500)
    lcg_noise = uniform_random(-600)
    noise_sigma = math.sqrt(OSR) * 10 ** (-snr_db / 20) * 2 * math.sqrt(2.5 / 2)
    total_errors = 0
    total_bits = 0
    error_sectors = 0

    gpr_target = gpr_target_init if gpr_target_init is not None else PRI_IMP
    eq_coeff = eq_coeff_init if eq_coeff_init is not None else np.zeros(NUM_EQ_TAPS)

    for s in range(num_sectors):
        bits = np.array(
            [int(lcg_bits.random() > 0.5) for _ in range(sector_len)],
            dtype=np.int64,
        )
        bits[0] = 0

        if encoder_fn is not None:
            encoded = encoder_fn(bits, sector_len)
        else:
            encoded = bits[:sector_len].copy()

        encoded_len = len(encoded)
        oss_len = (PRE + encoded_len + PRE) * OSR

        padded = np.zeros(PRE + encoded_len + PRE, dtype=np.int64)
        padded[PRE: PRE + encoded_len] = encoded
        bipolar = make_bipolar(padded)

        ch_out = channel(bipolar, ch_type, 2.5, 201, OSR, 0.0, 0.0, {}, s)

        noise = np.array([gaussian_random(lcg_noise) for _ in range(oss_len)],
                         dtype=np.float64)
        inner_start = PRE * OSR
        inner_stop = oss_len - PRE * OSR
        ch_out[inner_start: inner_stop] += noise[inner_start: inner_stop] * noise_sigma

        ch_for_lpf = ch_out[:oss_len]
        lpf_out = lpf(ch_for_lpf, 20, 1.0 / OSR)

        ds = lpf_out[::OSR][:PRE + encoded_len + PRE]

        eq_out = apply_equalizer(ds[PRE: PRE + encoded_len], eq_coeff, NUM_EQ_TAPS)

        detected, _ = classical_viterbi(20, eq_out, encoded_len, gpr_target)

        if decoder_fn is not None:
            decoded, _ = decoder_fn(detected, 0, encoded_len)
            compare = min(len(decoded), sector_len)
            errors = int(np.sum(decoded[:compare] != bits[:compare]))
        elif encoder_fn is not None:
            # Fallback: use default RLL decoder
            decoded, _ = dec_4by5rll_code(detected, 0, encoded_len)
            compare = min(len(decoded), sector_len)
            errors = int(np.sum(decoded[:compare] != bits[:compare]))
        else:
            compare = encoded_len
            errors = int(np.sum(detected[:compare] != bits[:compare]))

        total_errors += errors
        total_bits += compare
        if errors > 0:
            error_sectors += 1

    ber = total_errors / total_bits if total_bits > 0 else 1.0
    ser = error_sectors / num_sectors if num_sectors > 0 else 1.0
    return ber, ser, {
        "total_errors": total_errors,
        "total_bits": total_bits,
        "error_sectors": error_sectors,
    }


def compute_gpr_target_for_sweep(
    ch_type: str,
    encoder_fn,
    sector_len: int,
    num_eq_sectors: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute GPR target and equalizer coefficients from training sectors.

    Matches C FindGPRTarget() usage in GPRTarget mode.
    Returns (gpr_target, eq_coeff).
    """
    PRE = 20
    OSR = 10
    lcg_bits = uniform_random(-500)
    lcg_noise = uniform_random(-600)
    adapt_snr = 50.0
    nd = 2.5
    noise_sigma = math.sqrt(OSR) * 10 ** (-adapt_snr / 20) * 2 * math.sqrt(nd / 2)

    concat_input: list[np.ndarray] = []
    concat_output: list[np.ndarray] = []

    for s in range(num_eq_sectors):
        bits = np.array(
            [int(lcg_bits.random() > 0.5) for _ in range(sector_len)],
            dtype=np.int64,
        )
        bits[0] = 0

        if encoder_fn is not None:
            bits[0] = 0
            encoded = encoder_fn(bits, sector_len)
        else:
            encoded = bits[:sector_len].copy()

        encoded_len = len(encoded)
        oss_len = (PRE + encoded_len + PRE) * OSR

        padded = np.zeros(PRE + encoded_len + PRE, dtype=np.int64)
        padded[PRE: PRE + encoded_len] = encoded
        bipolar = make_bipolar(padded)

        ch_out = channel(bipolar, ch_type, 2.5, 201, OSR, 0.0, 0.0, {}, s)

        noise = np.array([gaussian_random(lcg_noise) for _ in range(oss_len)],
                         dtype=np.float64)
        inner_start = PRE * OSR
        inner_stop = oss_len - PRE * OSR
        ch_out[inner_start: inner_stop] += noise[inner_start: inner_stop] * noise_sigma

        ch_for_lpf = ch_out[:oss_len]
        lpf_out = lpf(ch_for_lpf, 20, 1.0 / OSR)

        ds = lpf_out[::OSR][:PRE + encoded_len + PRE]

        concat_input.append(make_bipolar(encoded))
        concat_output.append(ds[PRE: PRE + encoded_len])

    full_input = np.concatenate(concat_input)
    full_output = np.concatenate(concat_output)
    gpr_target, eq_coeff = find_gpr_target(
        full_output, full_input, NUM_EQ_TAPS,
        gpr_target_length=4,  # EPR4 [1,1,-1,-1] (matches C code)
    )
    return gpr_target, eq_coeff


# ===========================================================================
# Experiment 1: Channel Impulse Response Shapes
# ===========================================================================

def exp1_channel_impulse_response():
    """Experiment 1: Compare Longitudinal vs Perpendicular channel impulse
    responses for different normalized densities."""
    print("\n[EXP1] Channel Impulse Response")
    t = time.time()

    num_taps = 201
    osr = 10
    nd_values = [1.5, 2.5, 3.5]
    time_index = np.linspace(-(num_taps - 1) / (2 * osr),
                              (num_taps - 1) / (2 * osr), num_taps)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Longitudinal
    ax = axes[0]
    for nd in nd_values:
        coeffs = []
        vp = math.sqrt(2.0 / np.pi)
        for ti in time_index:
            c = (vp * -8.0 * ti / nd**2
                 * 1.0 / (1.0 + 4.0 * ti**2 / nd**2)**2)
            c /= nd
            coeffs.append(c)
        ax.plot(time_index, coeffs, label=f"ND = {nd}")
    ax.set_xlabel("Time (normalized)")
    ax.set_ylabel("Channel Coefficient")
    ax.set_title("Longitudinal Channel Impulse Response")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Perpendicular
    ax = axes[1]
    for nd in nd_values:
        coeffs = []
        vp = 0.5 * (np.pi / (2 * math.log(2)))**0.25
        for ti in time_index:
            c = (vp * 2.0 / nd
                 * math.sqrt(math.log(2) / np.pi)
                 * math.exp(-math.log(2) * 4 * ti**2 / nd**2))
            c /= nd
            coeffs.append(c)
        ax.plot(time_index, coeffs, label=f"ND = {nd}")
    ax.set_xlabel("Time (normalized)")
    ax.set_ylabel("Channel Coefficient")
    ax.set_title("Perpendicular Channel Impulse Response")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_fig("exp1_channel_impulse_response.png")
    print(f"  Done in {time.time()-t:.2f}s")
    return {}


# ===========================================================================
# Experiment 2: LPF Frequency Response
# ===========================================================================

def exp2_lpf_frequency_response():
    """Experiment 2: Low-pass filter frequency response for different orders."""
    print("\n[EXP2] LPF Frequency Response")
    t = time.time()

    filter_orders = [20, 50, 100]
    cutoff = 0.5

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Time domain
    ax = axes[0]
    for fo in filter_orders:
        cutoff_rad = cutoff * np.pi
        mid = fo / 2.0
        h = np.zeros(fo + 1)
        for i in range(fo + 1):
            if i == int(mid):
                h[i] = cutoff_rad / np.pi
            else:
                h[i] = np.sin(cutoff_rad * (i - mid)) / (np.pi * (i - mid))
            h[i] *= 0.54 - 0.46 * np.cos(2 * np.pi * i / fo)
        h = h / np.sum(h)
        ax.plot(h, label=f"Order = {fo}")
    ax.set_xlabel("Tap Index")
    ax.set_ylabel("Coefficient Value")
    ax.set_title("LPF Impulse Response (Hamming Windowed Sinc)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Frequency domain
    ax = axes[1]
    for fo in filter_orders:
        cutoff_rad = cutoff * np.pi
        mid = fo / 2.0
        h = np.zeros(fo + 1)
        for i in range(fo + 1):
            if i == int(mid):
                h[i] = cutoff_rad / np.pi
            else:
                h[i] = np.sin(cutoff_rad * (i - mid)) / (np.pi * (i - mid))
            h[i] *= 0.54 - 0.46 * np.cos(2 * np.pi * i / fo)
        h = h / np.sum(h)
        freq_response = np.abs(np.fft.fft(h, 1024))
        freq_db = 20 * np.log10(freq_response + 1e-12)
        freqs = np.linspace(0, 1, 1024)
        ax.plot(freqs, freq_db, label=f"Order = {fo}")
    ax.set_xlabel("Normalized Frequency (0 to 1)")
    ax.set_ylabel("Magnitude (dB)")
    ax.set_title("LPF Frequency Response")
    ax.set_ylim(-60, 5)
    ax.axvline(cutoff, color="r", linestyle="--", alpha=0.5, label="Cutoff")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_fig("exp2_lpf_frequency_response.png")
    print(f"  Done in {time.time()-t:.2f}s")
    return {}


# ===========================================================================
# Experiment 3: FIR Filter Verification
# ===========================================================================

def exp3_fir_filters():
    """Experiment 3: Verify causal and non-causal FIR filters against
    numpy convolution reference."""
    print("\n[EXP3] FIR Filter Verification")
    t = time.time()

    np.random.seed(42)
    n_signals = 100
    signal_len = 512

    # Non-causal FIR
    num_taps = 21
    h_nc = np.random.randn(num_taps)
    h_nc = h_nc / np.sum(np.abs(h_nc))

    errors_nc = []
    for _ in range(n_signals):
        x = np.random.randn(signal_len)
        result = non_causal_fir(x, h_nc)
        # Reference: manual convolution matching non_causal_fir implementation
        pad = num_taps // 2
        back_pad = num_taps - 1
        x_pad = np.zeros(signal_len + pad + back_pad)
        x_pad[pad: pad + signal_len] = x
        ref = np.zeros_like(result)
        for i in range(signal_len + pad):
            for j in range(num_taps):
                ref[i] += h_nc[j] * x_pad[i + j]
        errors_nc.append(float(np.max(np.abs(result - ref))))

    # Causal FIR
    h_c = np.random.randn(11)
    errors_c = []
    for _ in range(n_signals):
        x = np.random.randn(signal_len)
        result = causal_fir(x, h_c)
        ref = np.convolve(x[:signal_len], h_c, mode="full")
        errors_c.append(float(np.max(np.abs(result - ref))))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.hist(errors_nc, bins=50, density=True, alpha=0.7, color="steelblue")
    ax.set_xlabel("Max Absolute Error")
    ax.set_ylabel("Density")
    ax.set_title(f"Non-Causal FIR Error\nMean max error = {np.mean(errors_nc):.2e}")
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.hist(errors_c, bins=50, density=True, alpha=0.7, color="coral")
    ax.set_xlabel("Max Absolute Error")
    ax.set_ylabel("Density")
    ax.set_title(f"Causal FIR Error\nMean max error = {np.mean(errors_c):.2e}")
    ax.grid(True, alpha=0.3)

    save_fig("exp3_fir_filters.png")
    print(f"  Done in {time.time()-t:.2f}s")
    return {"nc_mean_error": float(np.mean(errors_nc)),
            "nc_max_error": float(np.max(errors_nc)),
            "c_mean_error": float(np.mean(errors_c)),
            "c_max_error": float(np.max(errors_c))}


# ===========================================================================
# Experiment 4: RLL(4/5) Code Characteristics
# ===========================================================================

def exp4_rll_code():
    """Experiment 4: Analyze RLL(4/5) codeword distribution and transition
    density."""
    print("\n[EXP4] RLL(4/5) Code Analysis")
    t = time.time()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Codeword table as text
    ax = axes[0]
    ax.axis("off")
    cw_bits = rll_cw.astype(int)  # shape (16, 5)
    lines = ["RLL(4/5) Codebook  (rate 4/5, RLL(0,2))",
             " " + "-" * 38]
    lines.append(f"{'Src':>4s}  |  {'Codeword':>5s}")
    lines.append(f"{'----':>4s}  |  {'------':>5s}")
    for i in range(16):
        cw = cw_bits[i]
        src = f"{i:04b}"
        row = f"  {src}   |  {''.join(str(b) for b in cw)}"
        lines.append(row)
    ax.text(0.1, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=11, verticalalignment="top", family="monospace",
            bbox=dict(boxstyle="round", facecolor="#e8f4e8", alpha=0.8))
    ax.set_title("RLL(4/5) Codeword Table")

    # Transition density comparison
    ax = axes[1]
    np.random.seed(42)
    num_trials = 10000
    user_trans = []
    enc_trans = []
    for _ in range(num_trials):
        bits = np.random.randint(0, 2, SECTOR_RLL, dtype=np.int64)
        bits[0] = 0
        encoded = enc_4by5rll_code(bits, SECTOR_RLL)
        user_t = np.sum(np.abs(np.diff(bits))) / max(len(bits) - 1, 1)
        enc_t = np.sum(np.abs(np.diff(encoded))) / max(len(encoded) - 1, 1)
        user_trans.append(user_t)
        enc_trans.append(enc_t)

    user_trans, enc_trans = np.array(user_trans), np.array(enc_trans)
    ax.bar(["User bits", "RLL(4/5) encoded"],
           [user_trans.mean(), enc_trans.mean()],
           color=["steelblue", "coral"], yerr=[user_trans.std(), enc_trans.std()],
           capsize=4, error_kw={"alpha": 0.5, "capsize": 4})
    ax.set_ylabel("Transition Density")
    ax.set_title(f"Transition Density\nRLL(4/5) increases transitions by {(enc_trans.mean()/user_trans.mean()-1)*100:.1f}%")

    save_fig("exp4_rll_code.png")
    print(f"  Done in {time.time()-t:.2f}s")
    return {}


# ===========================================================================
# Experiment 5: BER vs SNR - Viterbi Detector
# ===========================================================================

def _get_decoder_for_encoder(enc_fn):
    """Return the matching decoder function and its codeword length."""
    if enc_fn is None:
        return None, 0
    if enc_fn is enc_4by5rll_code:
        return dec_4by5rll_code, 4
    if enc_fn is enc_6by7mtr_code:
        return dec_6by7mtr_code, 6
    if enc_fn is enc_8by9tmtr_code:
        return dec_8by9tmtr_code, 8
    return dec_4by5rll_code, 4


def exp5_ber_snr_viterbi():
    """Experiment 5: BER vs SNR curves for different channel/encoding
    combinations using Viterbi detector with GPR fixed equalizer."""
    print("\n[EXP5] BER vs SNR (Viterbi)")
    t = time.time()

    snr_range = np.arange(24, 41, 2)  # 24, 26, ..., 40 dB
    num_sectors = 50
    num_eq_sectors = 20
    sector_len = SECTOR_512B  # 4097 bits (~512 bytes, 4Z+1 for RLL(4/5))

    configs = [
        ("Perpendicular", "No Encoding", False, None),
        ("Perpendicular", "RLL(4/5)", True, enc_4by5rll_code),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_ber = {}
    all_ser = {}

    # Pre-compute GPR target/coeffs for each config (one-time, matches C GPRTarget mode)
    gpr_cache: dict = {}
    for ch_type, enc_name, use_enc, enc_fn in configs:
        key = (ch_type, enc_fn)
        if key not in gpr_cache:
            print(f"  Computing GPR target for {enc_name} ...")
            gpr_target, eq_coeff = compute_gpr_target_for_sweep(
                ch_type, enc_fn, sector_len, num_eq_sectors=num_eq_sectors)
            gpr_cache[key] = (gpr_target, eq_coeff)

    for ax_idx, (ch_type, enc_name, use_enc, enc_fn) in enumerate(configs):
        if ax_idx >= 1:
            continue
        ax = axes[ax_idx]
        label = f"{ch_type} {'(' + enc_name + ')' if enc_name != 'None' else ''}"
        gpr_target, eq_coeff = gpr_cache[(ch_type, enc_fn)]
        ber_points = []
        for snr in snr_range:
            ber, _, _ = run_single_sector_sweep(
                ch_type, enc_fn, sector_len, snr, num_sectors,
                eq_coeff_init=eq_coeff, gpr_target_init=gpr_target)
            ber_points.append(ber)
            print(f"    {ch_type}/{enc_name}: SNR={snr}dB BER={ber:.2e}")
        ax.semilogy(snr_range, ber_points, "o-", label=label,
                    linewidth=2, markersize=6)
        all_ber[label] = ber_points

    axes[0].set_xlabel("SNR (dB)")
    axes[0].set_ylabel("Bit Error Rate (BER)")
    axes[0].set_title("BER vs SNR - Viterbi Detector")
    axes[0].legend()
    axes[0].grid(True, which="both", alpha=0.3)

    # SER curve
    for idx, (ch_type, enc_name, use_enc, enc_fn) in enumerate(configs):
        label = f"{ch_type} {'(' + enc_name + ')' if enc_name != 'None' else ''}"
        gpr_target, eq_coeff = gpr_cache[(ch_type, enc_fn)]
        ser_points = []
        for snr in snr_range:
            _, ser, _ = run_single_sector_sweep(
                ch_type, enc_fn, sector_len, snr, num_sectors,
                eq_coeff_init=eq_coeff, gpr_target_init=gpr_target)
            ser_points.append(ser)
        axes[1].semilogy(snr_range, ser_points, "s-", label=label,
                         linewidth=2, markersize=6)
        all_ser[label] = ser_points

    axes[1].set_xlabel("SNR (dB)")
    axes[1].set_ylabel("Block Error Rate (SER)")
    axes[1].set_title("SER vs SNR - Viterbi Detector")
    axes[1].legend()
    axes[1].grid(True, which="both", alpha=0.3)

    # Save GPR target and equalizer coefficients as text
    for (ch_type, enc_fn), (gpr_target, eq_coeff) in gpr_cache.items():
        enc_label = "NoEncoding" if enc_fn is None else enc_fn.__name__.replace("_", "-")
        with open(RESULTS / f"exp5_gpr_target_{enc_label}.txt", "w") as f:
            f.write(f"# GPR Target - Channel: {ch_type}, Encoding: {enc_label}\n")
            f.write(f"# Length: {len(gpr_target)}\n")
            f.write(" ".join(f"{v:.10f}" for v in gpr_target) + "\n")
        with open(RESULTS / f"exp5_eq_coeff_{enc_label}.txt", "w") as f:
            f.write(f"# Equalizer Coefficients - Channel: {ch_type}, Encoding: {enc_label}\n")
            f.write(f"# Num taps: {len(eq_coeff)}\n")
            for i, v in enumerate(eq_coeff):
                f.write(f"{i:>3d}: {v:.10f}\n")
        print(f"  Saved exp5_gpr_target_{enc_label}.txt, exp5_eq_coeff_{enc_label}.txt")

    save_fig("exp5_ber_snr_viterbi.png")
    print(f"  Done in {time.time()-t:.2f}s")
    return {"ber_points": all_ber, "ser_points": all_ser}


# ===========================================================================
# Experiment 6: SOVA Soft Output Analysis
# ===========================================================================

def exp6_sova_soft_output():
    """Experiment 6: Analyze SOVA soft output quality at different SNR levels.

    Uses GPR fixed equalizer (one-time computation) matching C GPRTarget mode.
    """
    print("\n[EXP6] SOVA Soft Output Analysis")
    t = time.time()

    snr_values = [28, 31, 34, 37]
    sector_len = SECTOR_RLL
    PRE = 20
    OSR = 10
    lcg_base = 0
    num_eq_sectors = 20

    # Pre-compute GPR target/coeffs (one-time, matches C GPRTarget mode)
    print("  Computing GPR target for SOVA analysis ...")
    gpr_target, eq_coeff = compute_gpr_target_for_sweep(
        "Perpendicular", enc_4by5rll_code, sector_len, num_eq_sectors)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, snr in enumerate(snr_values):
        ax = axes[idx // 2, idx % 2]
        lcg_bits = uniform_random(-500 - lcg_base)
        lcg_noise = uniform_random(-600 - lcg_base)
        lcg_base += 100
        noise_sigma = math.sqrt(OSR) * 10 ** (-snr / 20) * 2 * math.sqrt(2.5 / 2)

        bits = np.array(
            [int(lcg_bits.random() > 0.5) for _ in range(sector_len)],
            dtype=np.int64,
        )
        bits[0] = 0
        encoded = enc_4by5rll_code(bits, sector_len)
        encoded_len = len(encoded)

        # Pre/post padding before channel
        padded = np.zeros(PRE + encoded_len + PRE, dtype=np.int64)
        padded[PRE: PRE + encoded_len] = encoded
        bipolar = make_bipolar(padded)

        oss_len = (PRE + encoded_len + PRE) * OSR
        ch_out = channel(bipolar, "Perpendicular", 2.5, 201, OSR, 0.0, 0.0, {}, 0)

        # Noise added to inner region only (matches C)
        noise = np.array([gaussian_random(lcg_noise) for _ in range(oss_len)],
                         dtype=np.float64)
        inner_start = PRE * OSR
        inner_stop = oss_len - PRE * OSR
        ch_out[inner_start: inner_stop] += noise[inner_start: inner_stop] * noise_sigma

        # LPF: only first OSSectorLength samples
        lpf_out = lpf(ch_out[:oss_len], 20, 1.0 / OSR)

        # Downsample: DSOutput[i] = LPFOutput[i * OSR]
        ds = lpf_out[::OSR][:PRE + encoded_len + PRE]

        # Apply fixed GPR equalizer
        eq_out = apply_equalizer(ds[PRE: PRE + encoded_len], eq_coeff, NUM_EQ_TAPS)

        # SOVA detection with GPR target
        hard, soft, _ = classical_sova(10, eq_out, encoded_len, gpr_target, noise_sigma)

        # Decode
        decoded, _ = dec_4by5rll_code(hard, 0, encoded_len)
        compare = min(len(decoded), sector_len)
        errors = int(np.sum(decoded[:compare] != bits[:compare]))
        mean_conf = float(np.mean(soft))
        std_conf = float(np.std(soft))

        # Plot soft outputs color-coded by correctness
        correct = decoded[:compare] == bits[:compare]
        for i in range(compare):
            color = "green" if correct[i] else "red"
            ax.scatter(i, soft[i], s=3, c=color, alpha=0.4)

        ax.set_xlabel("Bit Index")
        ax.set_ylabel("Soft Output (Confidence)")
        ax.set_title(f"SNR = {snr}dB | Errors: {errors}/{sector_len} | "
                     f"Mean soft: {mean_conf:.2f}, Std: {std_conf:.2f}")
        ax.grid(True, alpha=0.3)

    save_fig("exp6_sova_soft_output.png")
    print(f"  Done in {time.time()-t:.2f}s")
    return {}


# ===========================================================================
# Experiment 7: LMS Adaptive Equalizer Convergence
# ===========================================================================

def exp7_equalizer_convergence():
    """Experiment 7: Track LMS adaptive equalizer MSE decrease over
    iterations using real channel signals with proper timing alignment.

    NOTE: This experiment specifically tests LMS convergence behavior.
    It uses per-sector LMS adaptation (FixedPRTarget mode) to demonstrate
    the MSE curve shape. The user should be aware that in the actual
    GPRTarget mode used in production, a fixed GPR equalizer is used
    instead of per-sector LMS.
    """
    print("\n[EXP7] LMS Equalizer Convergence")
    t = time.time()

    np.random.seed(42)
    pri_imp_res = np.array([1, 1, -1, -1], dtype=np.float64)
    num_eq_taps = 21
    sector_len = SECTOR_RLL
    PRE = 20
    OSR = 10
    num_iterations = 30
    # Moderate SNR for LMS convergence demonstration
    snr_demo = 30.0
    noise_sigma = math.sqrt(OSR) * 10 ** (-snr_demo / 20) * 2 * math.sqrt(2.5 / 2)

    eq_coeff = np.zeros(num_eq_taps)
    mse_history = []
    lmse_history = []

    # Save final iteration data for plots
    final_ds_output = None
    final_desired = None

    for iteration in range(num_iterations):
        start = 1 if iteration == 0 else 0

        # Generate a real channel signal with proper preamble/postamble
        bits = np.random.randint(0, 2, sector_len).astype(np.int64)
        bits[0] = 0
        padded = np.zeros(PRE + sector_len + PRE, dtype=np.int64)
        padded[PRE: PRE + sector_len] = bits
        bipolar = make_bipolar(padded)

        oss_len = (PRE + sector_len + PRE) * OSR
        ch_out = channel(bipolar, "Perpendicular", 2.5, 201, OSR, 0.0, 0.0, {}, iteration)

        # Add noise to inner region only (matches C)
        np.random.seed(42 + iteration)
        noise = np.array([gaussian_random(uniform_random(-600 - iteration))
                          for _ in range(oss_len)], dtype=np.float64)
        inner_start = PRE * OSR
        inner_stop = oss_len - PRE * OSR
        ch_out[inner_start: inner_stop] += noise[inner_start: inner_stop] * noise_sigma

        # LPF: only first OSSectorLength samples
        lpf_out = lpf(ch_out[:oss_len], 20, 1.0 / OSR)

        # Downsample: DSOutput[i] = LPFOutput[i * OSR]
        ds_full = lpf_out[::OSR][:PRE + sector_len + PRE]

        # Extract inner region (strip pre/post padding)
        ds_output = ds_full[PRE * OSR: PRE * OSR + sector_len * OSR: OSR][:sector_len]

        # Desired output = PR-shaped signal from clean bits through PR FIR
        bits_bipolar = make_bipolar(padded[PRE: PRE + sector_len])
        desired = _causal_fir_simple(bits_bipolar, pri_imp_res)

        mse, avg_lmse = adapt_equalizer(
            ds_output, desired, eq_coeff, num_eq_taps,
            sector_len, pri_imp_res, pri_imp_res_length=4,
            start_flag=start,
        )
        mse_history.append(mse)
        lmse_history.append(avg_lmse)

        if iteration == num_iterations - 1:
            final_ds_output = ds_output
            final_desired = desired

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # MSE convergence
    ax = axes[0, 0]
    ax.semilogy(range(1, num_iterations + 1), mse_history, "bo-",
                linewidth=2, markersize=6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total MSE")
    ax.set_title("LMS Adaptive Equalizer MSE Convergence")
    ax.grid(True, alpha=0.3)

    # Average LMS error
    ax = axes[0, 1]
    ax.semilogy(range(1, num_iterations + 1), lmse_history, "ro-",
                linewidth=2, markersize=6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Average LMS Error")
    ax.set_title("LMS Adaptive Equalizer Average Error")
    ax.grid(True, alpha=0.3)

    # Equalizer Output vs Desired (final iteration)
    ax = axes[1, 0]
    if final_ds_output is not None:
        applied = apply_equalizer(final_ds_output, eq_coeff, num_eq_taps)
        # Apply delay compensation for visualization
        eq_delay = (num_eq_taps - 1) // 2  # 10
        pr_delay = len(pri_imp_res) // 2   # 2
        shift = eq_delay - pr_delay        # 8
        if shift > 0 and shift < len(applied):
            aligned_eq = np.zeros_like(applied)
            aligned_eq[:len(applied) - shift] = applied[shift:]
            applied = aligned_eq
        x = np.arange(len(applied))
        ax.plot(x, applied, "b-", label="Equalizer Output", linewidth=1.5, alpha=0.7)
        ax.plot(x, final_desired, "r--", label="Desired PR Response", linewidth=2)
        ax.set_title("Equalizer Output vs Desired PR (delay-compensated)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # PR Target vs GPR Target
    ax = axes[1, 1]
    gpr_target, _ = find_gpr_target(ds_output, desired, num_eq_taps,
                                     gpr_target_length=len(pri_imp_res))
    ax.bar(range(len(pri_imp_res)), pri_imp_res, color="steelblue",
           alpha=0.7, width=0.4, label="PR Target")
    ax.bar(range(len(gpr_target)) + 0.4, gpr_target, color="coral",
           alpha=0.7, width=0.4, label="GPR Target")
    ax.set_xlabel("Coefficient Index")
    ax.set_ylabel("Coefficient Value")
    ax.set_title("PR Target vs GPR Target Coefficients")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_fig("exp7_equalizer_convergence.png")
    print(f"  Done in {time.time()-t:.2f}s")
    return {"mse_initial": float(mse_history[0]),
            "mse_final": float(mse_history[-1]),
            "improvement_ratio": float(mse_history[0] / mse_history[-1])}


# ===========================================================================
# Experiment 8: GPR Target Adaptation
# ===========================================================================

def exp8_gpr_target():
    """Experiment 8: Compare GPR target vs PR target equalizer
    coefficients."""
    print("\n[EXP8] GPR Target Adaptation")
    t = time.time()

    np.random.seed(42)
    seq_len = SECTOR_FAST
    pri_imp_res = np.array([1, 1, -1, -1], dtype=np.float64)
    num_eq_taps = 21

    user_bits = np.random.randint(0, 2, seq_len).astype(np.float64)
    user_bits = 2.0 * user_bits - 1.0

    ch_output = user_bits.copy()
    noise = np.random.randn(seq_len) * 0.01
    ch_output += noise

    gpr_target, eq_coeff = find_gpr_target(
        ch_output, user_bits, num_eq_taps,
        gpr_target_length=len(pri_imp_res),
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # GPR target coefficients
    ax = axes[0]
    ax.bar(range(len(gpr_target)), gpr_target, color="steelblue", edgecolor="black")
    ax.set_xlabel("Coefficient Index")
    ax.set_ylabel("Coefficient Value")
    ax.set_title(f"GPR Target (energy = {np.sum(gpr_target ** 2):.2f})")
    ax.grid(True, alpha=0.3)

    # Equalizer coefficients
    ax = axes[1]
    ax.plot(eq_coeff, "bo-", linewidth=2, markersize=4)
    ax.axhline(0, color="k", linewidth=0.5)
    ax.set_xlabel("Coefficient Index")
    ax.set_ylabel("Coefficient Value")
    ax.set_title(f"Adapted Equalizer Coefficients\n({num_eq_taps} taps)")
    ax.grid(True, alpha=0.3)

    # Input vs desired vs equalized output
    ax = axes[2]
    eq_out_temp = non_causal_fir(ch_output, eq_coeff)
    # Take first seq_len samples
    eq_out_temp = eq_out_temp[:seq_len]
    ax.plot(ch_output[:100], "k--", alpha=0.3, label="Input (noisy)")
    ax.plot(user_bits[:100], "r--", alpha=0.3, label="Desired (PR)")
    ax.plot(eq_out_temp[:100], "b-", linewidth=1.5, label="Equalized")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.set_title("Equalizer Output vs Desired (first 100 samples)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_fig("exp8_gpr_target.png")
    print(f"  Done in {time.time()-t:.2f}s")
    return {}


# ===========================================================================
# Experiment 9: Encoding Overhead Analysis
# ===========================================================================

def exp9_encoding_overhead():
    """Experiment 9: Analyze encoding overhead for different code
    rates."""
    print("\n[EXP9] Encoding Overhead")
    t = time.time()

    codes = [
        ("4/5 RLL(0,2)", 4, 5, enc_4by5rll_code, SECTOR_RLL),
        ("6/7 MTR(2;8)", 6, 7, enc_6by7mtr_code, SECTOR_MTR),
        ("8/9 TMTR(2/3;11)", 8, 9, enc_8by9tmtr_code, SECTOR_TMTR),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx, (name, uwl, cwl, enc_fn, default_sl) in enumerate(codes):
        ax = axes[idx]
        # Generate valid sector lengths for this code
        if cwl == 5:  # RLL: 4Z+1
            sl_list = list(range(41, 401, 40))
            sl_list = [s for s in sl_list if s % 4 == 1]
        elif cwl == 7:  # MTR: 6Z+1
            sl_list = list(range(61, 481, 40))
            sl_list = [s for s in sl_list if s % 6 == 1]
        else:  # TMTR: 8Z+1
            sl_list = list(range(81, 561, 40))
            sl_list = [s for s in sl_list if s % 8 == 1]

        overhead = []
        for sl in sl_list:
            bits = np.random.randint(0, 2, sl, dtype=np.int64)
            bits[0] = 0
            encoded = enc_fn(bits, sl)
            rate = len(bits) / len(encoded)
            overhead.append(rate)

        ax.plot(sl_list, overhead, "o-", linewidth=2, markersize=6)
        ax.axhline(uwl / cwl, color="r", linestyle="--",
                   alpha=0.5, label=f"Rate = {uwl}/{cwl} = {uwl / cwl:.3f}")
        ax.set_xlabel("Sector Length (bits)")
        ax.set_ylabel("User/Encoded Ratio")
        ax.set_title(f"{name} Encoding Rate")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Encoding Rate vs Sector Length")
    save_fig("exp9_encoding_overhead.png")

    # Summary table
    fig2, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    table_data = []
    for name, uwl, cwl, _, _ in codes:
        table_data.append([name, str(uwl), str(cwl), f"{uwl / cwl:.4f}"])
    table = ax.table(
        cellText=table_data,
        colLabels=["Code", "User Bits", "Code Bits", "Rate"],
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2)
    ax.set_title("Encoding Rate Summary", fontsize=14, fontweight="bold")
    save_fig("exp9_encoding_rate_table.png")

    print(f"  Done in {time.time()-t:.2f}s")
    return {}


# ===========================================================================
# Experiment 10: LCG RNG Statistical Properties
# ===========================================================================

def exp10_rng_analysis():
    """Experiment 10: Statistical analysis of the LCG random number
    generator."""
    print("\n[EXP10] LCG RNG Analysis")
    t = time.time()

    lcg = uniform_random(-500)
    n_samples = 100000

    samples = np.array([lcg.random() for _ in range(n_samples)],
                       dtype=np.float64)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Histogram
    ax = axes[0, 0]
    ax.hist(samples, bins=50, density=True, alpha=0.7, color="steelblue",
            edgecolor="black")
    ax.axhline(1.0, color="r", linestyle="--", alpha=0.5, label="Uniform(0,1)")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(
        f"LCG Output Distribution\n"
        f"Mean = {np.mean(samples):.4f} (exp=0.5), "
        f"Std = {np.std(samples):.4f} (exp={1 / np.sqrt(12):.4f})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Autocorrelation
    ax = axes[0, 1]
    sc = samples - np.mean(samples)
    acf = np.correlate(sc, sc, mode="full")[len(sc) - 1:]
    acf = acf / acf[0]
    lags = np.arange(len(acf))
    ax.plot(lags[:100], acf[:100], "b-", linewidth=1.5)
    threshold = 2.58 / np.sqrt(n_samples)
    ax.axhline(threshold, color="r", linestyle="--", alpha=0.5, label="99% conf")
    ax.axhline(-threshold, color="r", linestyle="--", alpha=0.5)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title("LCG Autocorrelation Function")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Spectral density
    ax = axes[1, 0]
    nfft = 1024
    spec = np.abs(np.fft.fft(samples - np.mean(samples), nfft))
    spec_db = 20 * np.log10(spec[:nfft // 2] + 1e-12)
    freqs = np.linspace(0, 1, nfft // 2)
    ax.plot(freqs, spec_db, "b-", linewidth=1)
    ax.set_xlabel("Normalized Frequency")
    ax.set_ylabel("Spectral Density (dB)")
    ax.set_title("LCG Spectral Density")
    ax.grid(True, alpha=0.3)

    # Gaussian RNG
    ax = axes[1, 1]
    lcg_g = uniform_random(-600)
    gauss = np.array([gaussian_random(lcg_g) for _ in range(10000)],
                     dtype=np.float64)
    ax.hist(gauss, bins=50, density=True, alpha=0.7, color="coral",
            edgecolor="black")
    x = np.linspace(-4, 4, 200)
    ax.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), "r-", linewidth=2,
            label="N(0,1)")
    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Gaussian RNG (Box-Muller)\n"
        f"Mean = {np.mean(gauss):.4f}, Std = {np.std(gauss):.4f}"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_fig("exp10_rng_analysis.png")
    print(f"  Done in {time.time()-t:.2f}s")
    return {}


# ===========================================================================
# Experiment 11: Viterbi vs SOVA Comparison
# ===========================================================================

def exp11_detector_comparison():
    """Experiment 11: Compare Viterbi and SOVA performance with GPR fixed
    equalizer matching C GPRTarget mode."""
    print("\n[EXP11] Viterbi vs SOVA Comparison")
    t = time.time()

    snr_range = np.arange(24, 41, 2)  # 24-40 dB
    sector_len = SECTOR_RLL  # 101 bits (4Z+1)
    PRE = 20
    OSR = 10
    num_trials = 50
    num_eq_sectors = 20

    # Pre-compute GPR target/coeffs (one-time)
    print("  Computing GPR target for Exp11 ...")
    gpr_target, eq_coeff = compute_gpr_target_for_sweep(
        "Perpendicular", enc_4by5rll_code, sector_len, num_eq_sectors)

    viterbi_ber = []
    sova_ber = []
    sova_conf = []

    for snr in snr_range:
        v_errors = 0
        s_errors = 0
        total_bits = 0
        conf_sum = 0.0
        noise_sigma = math.sqrt(OSR) * 10 ** (-snr / 20) * 2 * math.sqrt(2.5 / 2)

        for trial in range(num_trials):
            lcg_v = uniform_random(-500 - trial)
            lcg_s = uniform_random(-501 - trial)
            lcg_n = uniform_random(-600 - trial)

            bits = np.array(
                [int(lcg_v.random() > 0.5) for _ in range(sector_len)],
                dtype=np.int64,
            )
            bits[0] = 0
            encoded = enc_4by5rll_code(bits, sector_len)
            encoded_len = len(encoded)

            # Pre/post padding before channel (matches C code)
            padded = np.zeros(PRE + encoded_len + PRE, dtype=np.int64)
            padded[PRE: PRE + encoded_len] = encoded
            bipolar = make_bipolar(padded)

            oss_len = (PRE + encoded_len + PRE) * OSR
            ch_out = channel(bipolar, "Perpendicular", 2.5, 201, OSR, 0.0, 0.0, {}, trial)

            # Noise added to inner region only
            noise = np.array([gaussian_random(lcg_n) for _ in range(oss_len)],
                             dtype=np.float64)
            inner_start = PRE * OSR
            inner_stop = oss_len - PRE * OSR
            ch_out[inner_start: inner_stop] += noise[inner_start: inner_stop] * noise_sigma

            # LPF: only first OSSectorLength samples
            lpf_out = lpf(ch_out[:oss_len], 20, 1.0 / OSR)

            # Downsample: DSOutput[i] = LPFOutput[i * OSR]
            ds = lpf_out[::OSR][:PRE + encoded_len + PRE]

            # Apply fixed GPR equalizer
            eq_out = apply_equalizer(ds[PRE: PRE + encoded_len], eq_coeff, NUM_EQ_TAPS)

            # Viterbi
            v_hard, _ = classical_viterbi(PRE, eq_out, encoded_len, gpr_target)
            v_errors += int(np.sum(v_hard[:encoded_len] != bits[:encoded_len]))

            # SOVA (use same equalized output)
            s_hard, s_soft, _ = classical_sova(
                PRE, eq_out, encoded_len, gpr_target, noise_sigma)
            s_errors += int(np.sum(s_hard[:encoded_len] != bits[:encoded_len]))
            conf_sum += float(np.mean(s_soft))
            total_bits += encoded_len

        viterbi_ber.append(v_errors / total_bits)
        sova_ber.append(s_errors / total_bits)
        sova_conf.append(conf_sum / max(total_bits, 1))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.semilogy(snr_range, viterbi_ber, "bo-", label="Viterbi",
                linewidth=2, markersize=8)
    ax.semilogy(snr_range, sova_ber, "rs-", label="SOVA",
                linewidth=2, markersize=8)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("BER")
    ax.set_title("Viterbi vs SOVA BER Performance")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    ax = axes[1]
    ax.plot(snr_range, sova_conf, "rs-", label="SOVA Mean Confidence",
            linewidth=2, markersize=8)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Mean Soft Output")
    ax.set_title("SOVA Mean Confidence vs SNR")
    ax.grid(True, alpha=0.3)

    save_fig("exp11_detector_comparison.png")
    print(f"  Done in {time.time()-t:.2f}s")
    return {}


# ===========================================================================
# Experiment 12: Full Pipeline with Different Encoders
# ===========================================================================

def exp12_full_pipeline():
    """Experiment 12: End-to-end pipeline with different encoder types
    at fixed SNR, using GPR fixed equalizer (C GPRTarget mode)."""
    print("\n[EXP12] Full Pipeline Comparison")
    t = time.time()

    snr = 30
    configs = [
        ("No Encoding", None, None, SECTOR_RLL),
        ("RLL(4/5)", enc_4by5rll_code, dec_4by5rll_code, SECTOR_RLL),
        ("MTR(6/7)", enc_6by7mtr_code, dec_6by7mtr_code, SECTOR_MTR),
        ("TMTR(8/9)", enc_8by9tmtr_code, dec_8by9tmtr_code, SECTOR_TMTR),
    ]
    ch_type = "Perpendicular"
    num_sectors = 100
    num_eq_sectors = 20

    # Pre-compute GPR target/coeffs for each config
    gpr_cache: dict = {}
    for enc_name, enc_fn, dec_fn, sector_len in configs:
        print(f"  Computing GPR target for {enc_name} ...")
        gpr_target, eq_coeff = compute_gpr_target_for_sweep(
            ch_type, enc_fn, sector_len, num_eq_sectors=num_eq_sectors)
        gpr_cache[(enc_fn, sector_len)] = (gpr_target, eq_coeff)

    labels = []
    bers = []
    sers = []

    fig, ax = plt.subplots(figsize=(10, 6))

    for enc_name, enc_fn, dec_fn, sector_len in configs:
        gpr_target, eq_coeff = gpr_cache[(enc_fn, sector_len)]
        decoder_fn, _ = _get_decoder_for_encoder(dec_fn) if dec_fn else (None, 0)

        ber, ser, info = run_single_sector_sweep_with_gpr(
            ch_type, enc_fn, sector_len, snr, num_sectors,
            eq_coeff_init=eq_coeff, gpr_target_init=gpr_target,
            decoder_fn=decoder_fn)

        labels.append(enc_name)
        bers.append(ber)
        sers.append(ser)
        print(f"    {enc_name}: BER={ber:.2e} SER={ser:.4f}")

    ax.bar(range(len(labels)), bers, color=["steelblue", "coral", "seagreen", "orchid"])
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title(f"Full Pipeline BER at SNR = {snr}dB")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for i, v in enumerate(bers):
        ax.text(i, v, f"{v:.1e}", ha="center", va="bottom", fontsize=9)

    save_fig("exp12_full_pipeline.png")
    print(f"  Done in {time.time()-t:.2f}s")
    return {"bers": dict(zip(labels, bers)), "sers": dict(zip(labels, sers))}


# ===========================================================================
# Experiment 13: Code Comparison - All Three Encoders
# ===========================================================================

def exp13_all_codes_roundtrip():
    """Experiment 13: Round-trip BER analysis for all three code types
    with GPR fixed equalizer."""
    print("\n[EXP13] All Codes Round-Trip BER")
    t = time.time()

    snr_range = np.arange(26, 40, 2)  # 26-38 dB
    codes = [
        ("RLL(4/5)", enc_4by5rll_code, dec_4by5rll_code, SECTOR_RLL),
        ("MTR(6/7)", enc_6by7mtr_code, dec_6by7mtr_code, SECTOR_MTR),
        ("TMTR(8/9)", enc_8by9tmtr_code, dec_8by9tmtr_code, SECTOR_TMTR),
    ]
    num_eq_sectors = 20

    # Pre-compute GPR target/coeffs for each code
    gpr_cache: dict = {}
    for name, enc_fn, dec_fn, sector_len in codes:
        print(f"  Computing GPR target for {name} ...")
        gpr_target, eq_coeff = compute_gpr_target_for_sweep(
            "Perpendicular", enc_fn, sector_len, num_eq_sectors=num_eq_sectors)
        gpr_cache[(enc_fn, sector_len)] = (gpr_target, eq_coeff)

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["blue", "green", "red"]
    for (name, enc_fn, dec_fn, sector_len), color in zip(codes, colors):
        gpr_target, eq_coeff = gpr_cache[(enc_fn, sector_len)]
        decoder_fn, _ = _get_decoder_for_encoder(dec_fn) if dec_fn else (None, 0)
        ber_points = []

        for snr in snr_range:
            ber, _, _ = run_single_sector_sweep_with_gpr(
                "Perpendicular", enc_fn, sector_len, snr, 50,
                eq_coeff_init=eq_coeff, gpr_target_init=gpr_target,
                decoder_fn=decoder_fn)
            ber_points.append(ber)
            print(f"    {name}: SNR={snr}dB BER={ber:.2e}")

        ax.semilogy(snr_range, ber_points, "o-", label=name,
                    linewidth=2, markersize=8, color=color)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title("All Encoders - Viterbi + Decoder Round-Trip (GPR Equalizer)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    save_fig("exp13_all_codes_ber.png")

    print(f"  Done in {time.time()-t:.2f}s")
    return {}


# ===========================================================================
# Main: Run All Experiments
# ===========================================================================

def main():
    """Run all experiments and collect results."""
    print("=" * 60)
    print("HAMR Receiver Simulator -- Comprehensive Experiment Suite")
    print("=" * 60)

    all_results = {}

    exp_funcs = [
        ("Exp1: Channel Impulse Response", exp1_channel_impulse_response),
        ("Exp2: LPF Frequency Response", exp2_lpf_frequency_response),
        ("Exp3: FIR Filter Verification", exp3_fir_filters),
        ("Exp4: RLL(4/5) Code Analysis", exp4_rll_code),
        ("Exp5: BER vs SNR (Viterbi)", exp5_ber_snr_viterbi),
        ("Exp6: SOVA Soft Output", exp6_sova_soft_output),
        ("Exp7: Equalizer Convergence", exp7_equalizer_convergence),
        ("Exp8: GPR Target Adaptation", exp8_gpr_target),
        ("Exp9: Encoding Overhead", exp9_encoding_overhead),
        ("Exp10: LCG RNG Analysis", exp10_rng_analysis),
        ("Exp11: Viterbi vs SOVA", exp11_detector_comparison),
        ("Exp12: Full Pipeline", exp12_full_pipeline),
        ("Exp13: All Codes Comparison", exp13_all_codes_roundtrip),
    ]

    total_start = time.time()

    for name, func in exp_funcs:
        try:
            result = func()
            all_results[name] = result
            print(f"  PASSED: {name}")
        except Exception as e:
            print(f"  FAILED: {name}: {e}")
            import traceback
            traceback.print_exc()
            all_results[name] = {"error": str(e)}

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)
    print(f"All experiments completed in {total_elapsed:.1f}s")
    print("=" * 60)

    summary = {"total_elapsed": total_elapsed, "results": all_results}
    with open(RESULTS / "experiment_results.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Results saved to {RESULTS / 'experiment_results.json'}")


if __name__ == "__main__":
    main()

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

PRI_IMP = np.array([1, 1, -1, -1], dtype=np.float64)
NUM_EQ_TAPS = 21


def save_fig(name: str, dpi: int = 200) -> None:
    plt.tight_layout()
    plt.savefig(ASSETS / name, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"  -> saved {ASSETS / name}")


def make_bipolar(bits: np.ndarray) -> np.ndarray:
    """0/1 -> bipolar (+1/-1)."""
    return 2.0 * bits.astype(np.float64) - 1.0


def run_single_sector_sweep(
    ch_type: str,
    encoder_fn,
    sector_len: int,
    snr_db: float,
    num_sectors: int = 100,
) -> tuple[float, float]:
    """Run one SNR point, return (ber, ser).

    Matches the C code pipeline:
    1. Encode user bits
    2. Pad with PREAMBLE_LENGTH pre + POSTAMBLE_LENGTH post (matching C)
    3. Channel -> LPF -> Downsample from PRE*OSR
    4. Viterbi detection
    5. Decode and compare to original user bits
    """
    PRE = 20  # preamble length (matches C PREAMBLE_LENGTH)
    lcg_bits = uniform_random(-500)
    lcg_noise = uniform_random(-600)
    noise_sigma = math.sqrt(10) * 10 ** (-snr_db / 20) * 2 * math.sqrt(2.5 / 2)
    total_errors = 0
    total_bits = 0
    error_sectors = 0

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

        # Pad with pre/post padding (matches C code: PadSector with PREAMBLE/POSTAMBLE)
        padded = np.zeros(PRE + encoded_len + PRE, dtype=np.int64)
        padded[PRE: PRE + encoded_len] = encoded
        bipolar = make_bipolar(padded)

        # Channel + noise
        # Channel output length = len(bipolar) = (PRE + encoded_len + PRE)
        # (C code truncates FIR output to input length)
        ch_out = channel(bipolar, ch_type, 2.5, 201, 10, 0.0, 0.0, {}, s)

        noise = np.array([gaussian_random(lcg_noise) for _ in range(len(ch_out))],
                         dtype=np.float64)
        ch_out += noise * noise_sigma

        # LPF + downsample
        # LPF extends by filter_order//2 = 10 at the front
        lpf_out = lpf(ch_out, 20, 0.1)

        # Downsample: start from PRE * OSR + LPF_front_extension
        # C code: Downsample[PRE * OSR .. PRE * OSR + (SectorLength+PADDING) * OSR : OSR]
        ds_start = PRE * 10 + 10  # 210 (200 + 10 LPF front extension)
        ds = lpf_out[ds_start: ds_start + encoded_len * 10: 10]

        # Extract data region (skip pre-padding in downsampled domain)
        # Pre-padding in bits = PRE, in samples = PRE * OSR, in downsampled = PRE
        # But since we already started from PRE*OSR, ds[0] = sample at PRE
        # So the data starts at ds[0] directly, no further skip needed
        eq_out = ds[:encoded_len]

        # Viterbi detection
        detected, _ = classical_viterbi(10, eq_out, encoded_len, PRI_IMP)

        # Decode if needed
        if encoder_fn is not None:
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
    return ber, ser


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

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Codeword display
    ax = axes[0]
    cw_bits = rll_cw.astype(int)  # shape (16, 5)
    n_cw = len(rll_cw)
    for i in range(n_cw):
        ax.bar(i, cw_bits[i], width=0.8, color="steelblue", edgecolor="black")
    ax.set_xlabel("Codeword Decimal (input value)")
    ax.set_ylabel("Bit Value")
    ax.set_xticks(range(n_cw))
    ax.set_title(f"RLL(4/5) Codeword Table\n({n_cw} codewords, {cw_bits.shape[1]} bits each)")
    ax.set_ylim(-0.1, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

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
        nt = len(encoded)
        user_t = np.sum(np.abs(np.diff(bits))) / max(len(bits) - 1, 1)
        enc_t = np.sum(np.abs(np.diff(encoded))) / max(nt - 1, 1)
        user_trans.append(user_t)
        enc_trans.append(enc_t)

    ax.hist(user_trans, bins=30, alpha=0.5, label="User bits", density=True)
    ax.hist(enc_trans, bins=30, alpha=0.5, label="Encoded bits", density=True)
    ax.set_xlabel("Transition Density")
    ax.set_ylabel("Density")
    ax.set_title("Transition Density: User vs RLL(4/5) Encoded")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Sample encoded sequence
    ax = axes[2]
    bits = np.random.randint(0, 2, 21, dtype=np.int64)
    bits[0] = 0
    encoded = enc_4by5rll_code(bits, 21)
    # Expand bits to match encoded length for display
    scale = len(encoded) / len(bits)
    x_pos = np.linspace(0, len(encoded) - 1, len(bits)).astype(int)
    display_bits = np.zeros(len(encoded), dtype=int)
    for i, pos in enumerate(x_pos):
        display_bits[pos] = bits[i]

    ax.imshow([display_bits[:len(encoded)], encoded], aspect="auto",
              cmap="Blues", interpolation="nearest")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["User bits (expanded)", "Encoded"])
    ax.set_xlabel("Bit Position")
    ax.set_title("Sample RLL(4/5) Encoding\n(0/1 -> NRZI -> NRZ)")
    ax.grid(False)

    save_fig("exp4_rll_code.png")
    print(f"  Done in {time.time()-t:.2f}s")
    return {}


# ===========================================================================
# Experiment 5: BER vs SNR - Viterbi Detector
# ===========================================================================

def exp5_ber_snr_viterbi():
    """Experiment 5: BER vs SNR curves for different channel/encoding
    combinations using Viterbi detector."""
    print("\n[EXP5] BER vs SNR (Viterbi)")
    t = time.time()

    snr_range = np.arange(24, 41, 2)  # 24, 26, ..., 40 dB
    num_sectors = 50

    configs = [
        ("Perpendicular", "No Encoding", False, None),
        ("Perpendicular", "RLL(4/5)", True, enc_4by5rll_code),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    all_ber = {}
    all_ser = {}

    for ax_idx, (ch_type, enc_name, use_enc, enc_fn) in enumerate(configs):
        if ax_idx >= 2:
            continue
        ax = axes[ax_idx]
        label = f"{ch_type} {'(' + enc_name + ')' if enc_name != 'None' else ''}"
        ber_points = []
        for snr in snr_range:
            ber, _ = run_single_sector_sweep(
                ch_type, enc_fn, SECTOR_FAST, snr, num_sectors)
            ber_points.append(ber)
            print(f"    {ch_type}/{enc_name}: SNR={snr}dB BER={ber:.2e}")
        ax.semilogy(snr_range, ber_points, "o-", label=label,
                    linewidth=2, markersize=6)
        all_ber[label] = ber_points

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title("BER vs SNR - Viterbi Detector")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # SER curve
    ax2 = axes[1]
    for idx, (ch_type, enc_name, use_enc, enc_fn) in enumerate(configs):
        label = f"{ch_type} {'(' + enc_name + ')' if enc_name != 'None' else ''}"
        ser_points = []
        for snr in snr_range:
            _, ser = run_single_sector_sweep(
                ch_type, enc_fn, SECTOR_FAST, snr, num_sectors)
            ser_points.append(ser)
        ax2.semilogy(snr_range, ser_points, "s-", label=label,
                     linewidth=2, markersize=6)
        all_ser[label] = ser_points

    ax2.set_xlabel("SNR (dB)")
    ax2.set_ylabel("Block Error Rate (SER)")
    ax2.set_title("SER vs SNR - Viterbi Detector")
    ax2.legend()
    ax2.grid(True, which="both", alpha=0.3)

    save_fig("exp5_ber_snr_viterbi.png")
    print(f"  Done in {time.time()-t:.2f}s")
    return {"ber_points": all_ber, "ser_points": all_ser}


# ===========================================================================
# Experiment 6: SOVA Soft Output Analysis
# ===========================================================================

def exp6_sova_soft_output():
    """Experiment 6: Analyze SOVA soft output quality at different SNR levels."""
    print("\n[EXP6] SOVA Soft Output Analysis")
    t = time.time()

    snr_values = [26, 29, 32, 35]
    sector_len = SECTOR_RLL
    PRE = 20
    lcg_base = 0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, snr in enumerate(snr_values):
        ax = axes[idx // 2, idx % 2]
        lcg_bits = uniform_random(-500 - lcg_base)
        lcg_noise = uniform_random(-600 - lcg_base)
        lcg_base += 100
        noise_sigma = math.sqrt(10) * 10 ** (-snr / 20) * 2 * math.sqrt(2.5 / 2)

        bits = np.array(
            [int(lcg_bits.random() > 0.5) for _ in range(sector_len)],
            dtype=np.int64,
        )
        bits[0] = 0

        # Pre/post padding before channel
        padded = np.zeros(PRE + sector_len + PRE, dtype=np.int64)
        padded[PRE: PRE + sector_len] = bits
        bipolar = make_bipolar(padded)

        ch_out = channel(bipolar, "Perpendicular", 2.5, 201, 10, 0.0, 0.0, {}, 0)
        noise = np.array([gaussian_random(lcg_noise) for _ in range(len(ch_out))],
                         dtype=np.float64)
        ch_out += noise * noise_sigma
        lpf_out = lpf(ch_out, 20, 0.1)

        # Downsample from PRE*OSR + LPF_front_extension
        ds_start = PRE * 10 + 10  # 210
        ds = lpf_out[ds_start: ds_start + sector_len * 10: 10]
        eq_out = ds[:sector_len]

        hard, soft, _ = classical_sova(10, eq_out, sector_len, PRI_IMP, noise_sigma)

        errors = int(np.sum(hard[:sector_len] != bits[:sector_len]))
        mean_conf = float(np.mean(soft))
        std_conf = float(np.std(soft))

        # Plot soft outputs color-coded by correctness
        correct = hard[:sector_len] == bits[:sector_len]
        for i in range(sector_len):
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
    iterations using real channel signals with proper timing alignment."""
    print("\n[EXP7] LMS Equalizer Convergence")
    t = time.time()

    np.random.seed(42)
    pri_imp_res = np.array([1, 1, -1, -1], dtype=np.float64)
    num_eq_taps = 21
    sector_len = SECTOR_RLL
    PRE = 20
    num_iterations = 30

    eq_coeff = np.zeros(num_eq_taps)
    mse_history = []
    lmse_history = []

    for iteration in range(num_iterations):
        start = 1 if iteration == 0 else 0

        # Generate a real channel signal with proper preamble/postamble
        bits = np.random.randint(0, 2, sector_len).astype(np.int64)
        bits[0] = 0
        padded = np.zeros(PRE + sector_len + PRE, dtype=np.int64)
        padded[PRE: PRE + sector_len] = bits
        bipolar = make_bipolar(padded)

        ch_out = channel(bipolar, "Perpendicular", 2.5, 201, 10, 0.0, 0.0, {}, iteration)
        lpf_out = lpf(ch_out, 20, 0.1)

        # Downsample from PRE*OSR + LPF_front_extension
        ds_start = PRE * 10 + 10
        ds_output = lpf_out[ds_start: ds_start + sector_len * 10: 10][:sector_len]

        # Desired output = PR impulse response repeated/padded to sector_len
        desired = np.zeros(sector_len)
        pr_len = len(pri_imp_res)
        # Place PR response centered in the sector (matches timing alignment)
        center = sector_len // 2
        for i in range(pr_len):
            idx = center - pr_len // 2 + i
            if 0 <= idx < sector_len:
                desired[idx] = pri_imp_res[i]

        mse, avg_lmse = adapt_equalizer(
            ds_output, desired, eq_coeff, num_eq_taps,
            sector_len, pri_imp_res, pri_imp_res_length=4,
            start_flag=start,
        )
        mse_history.append(mse)
        lmse_history.append(avg_lmse)

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
    # Apply final equalizer coefficients to show alignment
    applied = np.convolve(ds_output, eq_coeff, mode="same")
    # Compensate for equalizer group delay: shift by (num_eq_taps-1)//2
    delay = (num_eq_taps - 1) // 2
    if delay > 0:
        aligned_eq = np.zeros_like(applied)
        aligned_eq[delay:] = applied[:-delay]
        applied = aligned_eq
    x = np.arange(len(applied))
    ax.plot(x, applied, "b-", label="Equalizer Output", linewidth=1.5, alpha=0.7)
    ax.plot(x, desired, "r--", label="Desired PR Response", linewidth=2)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    ax.set_title("Equalizer Output vs Desired PR (delay-compensated)")
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
    """Experiment 11: Compare Viterbi and SOVA performance."""
    print("\n[EXP11] Viterbi vs SOVA Comparison")
    t = time.time()

    snr_range = np.arange(24, 41, 2)  # 24-40 dB
    sector_len = SECTOR_RLL  # 101 bits (4Z+1)
    viterbi_ber = []
    sova_ber = []
    sova_conf = []

    for snr in snr_range:
        v_errors = 0
        s_errors = 0
        total_bits = 0
        conf_sum = 0.0
        num_trials = 50  # Increased for reliable BER statistics

        lcg_v = uniform_random(-500)
        lcg_s = uniform_random(-500)
        lcg_n = uniform_random(-600)
        noise_sigma = math.sqrt(10) * 10 ** (-snr / 20) * 2 * math.sqrt(2.5 / 2)

        for _ in range(num_trials):
            bits = np.array(
                [int(lcg_v.random() > 0.5) for _ in range(sector_len)],
                dtype=np.int64,
            )
            bits[0] = 0

            # Pre/post padding before channel (matches C code pipeline)
            PRE = 20
            padded = np.zeros(PRE + sector_len + PRE, dtype=np.int64)
            padded[PRE: PRE + sector_len] = bits
            bipolar = make_bipolar(padded)

            ch_out = channel(bipolar, "Perpendicular", 2.5, 201, 10, 0.0, 0.0, {}, 0)
            noise = np.array([gaussian_random(lcg_n) for _ in range(len(ch_out))],
                             dtype=np.float64)
            ch_out += noise * noise_sigma
            lpf_out = lpf(ch_out, 20, 0.1)

            # Downsample from PRE*OSR + LPF_front_extension
            ds_start = PRE * 10 + 10  # 210
            ds = lpf_out[ds_start: ds_start + sector_len * 10: 10]
            eq_out = ds[:sector_len]

            # Viterbi
            v_hard, _ = classical_viterbi(10, eq_out, sector_len, PRI_IMP)
            v_errors += int(np.sum(v_hard[:sector_len] != bits[:sector_len]))

            # SOVA
            s_hard, s_soft, _ = classical_sova(
                10, eq_out, sector_len, PRI_IMP, noise_sigma)
            s_errors += int(np.sum(s_hard[:sector_len] != bits[:sector_len]))
            conf_sum += float(np.mean(s_soft))
            total_bits += sector_len

        viterbi_ber.append(v_errors / total_bits)
        sova_ber.append(s_errors / total_bits)
        sova_conf.append(conf_sum / (num_trials * sector_len))

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
    at fixed SNR."""
    print("\n[EXP12] Full Pipeline Comparison")
    t = time.time()

    snr = 28
    pri_imp_res = np.array([1, 1, -1, -1], dtype=np.float64)
    configs = [
        ("No Encoding", None, None, SECTOR_RLL),
        ("RLL(4/5)", enc_4by5rll_code, dec_4by5rll_code, SECTOR_RLL),
        ("MTR(6/7)", enc_6by7mtr_code, dec_6by7mtr_code, SECTOR_MTR),
        ("TMTR(8/9)", enc_8by9tmtr_code, dec_8by9tmtr_code, SECTOR_TMTR),
    ]
    ch_type = "Perpendicular"
    num_sectors = 100
    PRE = 20

    lcg_base = 0
    labels = []
    bers = []
    sers = []

    fig, ax = plt.subplots(figsize=(10, 6))

    for enc_name, enc_fn, dec_fn, sector_len in configs:
        lcg_bits = uniform_random(-500 - lcg_base)
        lcg_noise = uniform_random(-600 - lcg_base)
        lcg_base += 100
        noise_sigma = math.sqrt(10) * 10 ** (-snr / 20) * 2 * math.sqrt(2.5 / 2)

        total_errors = 0
        total_bits = 0
        error_sectors = 0

        for s in range(num_sectors):
            bits = np.array(
                [int(lcg_bits.random() > 0.5) for _ in range(sector_len)],
                dtype=np.int64,
            )
            bits[0] = 0

            if enc_fn is not None:
                bits[0] = 0
                encoded = enc_fn(bits, sector_len)
            else:
                encoded = bits[:sector_len].copy()

            encoded_len = len(encoded)

            # Pre/post padding before channel (matches C code)
            padded = np.zeros(PRE + encoded_len + PRE, dtype=np.int64)
            padded[PRE: PRE + encoded_len] = encoded
            bipolar = make_bipolar(padded)

            ch_out = channel(bipolar, ch_type, 2.5, 201, 10, 0.0, 0.0, {}, s)
            noise = np.array([gaussian_random(lcg_noise) for _ in range(len(ch_out))],
                             dtype=np.float64)
            ch_out += noise * noise_sigma
            lpf_out = lpf(ch_out, 20, 0.1)

            # Downsample from PRE*OSR + LPF_front_extension
            ds_start = PRE * 10 + 10  # 210
            ds = lpf_out[ds_start: ds_start + encoded_len * 10: 10]
            eq_out = ds[:encoded_len]

            v_hard, _ = classical_viterbi(10, eq_out, encoded_len, pri_imp_res)

            if dec_fn is not None:
                # MTR decoder returns 1 value, RLL/TMTR return 2
                result = dec_fn(v_hard, 0, encoded_len)
                decoded = result if isinstance(result, np.ndarray) else result[0]
                compare = min(len(decoded), sector_len)
                errors = int(np.sum(decoded[:compare] != bits[:compare]))
            else:
                compare = sector_len
                errors = int(np.sum(v_hard[:compare] != bits[:compare]))

            total_errors += errors
            total_bits += compare
            if errors > 0:
                error_sectors += 1

        ber = total_errors / total_bits if total_bits > 0 else 1.0
        ser = error_sectors / num_sectors
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
    """Experiment 13: Round-trip BER analysis for all three code types."""
    print("\n[EXP13] All Codes Round-Trip BER")
    t = time.time()

    snr_range = np.arange(24, 38, 2)  # 24-36 dB
    PRE = 20
    codes = [
        ("RLL(4/5)", enc_4by5rll_code, dec_4by5rll_code, SECTOR_RLL),
        ("MTR(6/7)", enc_6by7mtr_code, dec_6by7mtr_code, SECTOR_MTR),
        ("TMTR(8/9)", enc_8by9tmtr_code, dec_8by9tmtr_code, SECTOR_TMTR),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    colors = ["blue", "green", "red"]
    for (name, enc_fn, dec_fn, sector_len), color in zip(codes, colors):
        ber_points = []
        for snr in snr_range:
            lcg_bits = uniform_random(-500)
            lcg_noise = uniform_random(-600)
            noise_sigma = math.sqrt(10) * 10 ** (-snr / 20) * 2 * math.sqrt(2.5 / 2)

            total_errors = 0
            total_bits = 0
            for s in range(50):  # 50 sectors for reliable statistics
                bits = np.array(
                    [int(lcg_bits.random() > 0.5) for _ in range(sector_len)],
                    dtype=np.int64,
                )
                bits[0] = 0
                encoded = enc_fn(bits, sector_len)
                encoded_len = len(encoded)

                # Pre/post padding before channel
                padded = np.zeros(PRE + encoded_len + PRE, dtype=np.int64)
                padded[PRE: PRE + encoded_len] = encoded
                bipolar = make_bipolar(padded)

                ch_out = channel(bipolar, "Perpendicular", 2.5, 201, 10, 0.0, 0.0, {}, s)
                noise = np.array([gaussian_random(lcg_noise) for _ in range(len(ch_out))],
                                 dtype=np.float64)
                ch_out += noise * noise_sigma
                lpf_out = lpf(ch_out, 20, 0.1)

                # Downsample from PRE*OSR + LPF_front_extension
                ds_start = PRE * 10 + 10  # 210
                ds = lpf_out[ds_start: ds_start + encoded_len * 10: 10]
                eq_out = ds[:encoded_len]

                v_hard, _ = classical_viterbi(10, eq_out, encoded_len, PRI_IMP)
                # MTR decoder returns 1 value, RLL/TMTR return 2
                result = dec_fn(v_hard, 0, encoded_len)
                decoded = result if isinstance(result, np.ndarray) else result[0]
                compare = min(len(decoded), sector_len)
                errors = int(np.sum(decoded[:compare] != bits[:compare]))
                total_errors += errors
                total_bits += compare

            ber = total_errors / total_bits if total_bits > 0 else 1.0
            ber_points.append(ber)
        ax.semilogy(snr_range, ber_points, "o-", label=name,
                    linewidth=2, markersize=8, color=color)

    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Bit Error Rate (BER)")
    ax.set_title("All Encoders - Viterbi + Decoder Round-Trip")
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

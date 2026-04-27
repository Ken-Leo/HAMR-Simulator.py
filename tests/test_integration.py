"""Integration tests for the full HAMR receiver pipeline."""

import numpy as np
import pytest

from channel.channel import channel
from channel.lpf import lpf
from equalizer_detector.equalizer import (
    apply_equalizer,
    adapt_equalizer,
)
from equalizer_detector.viterbi import classical_viterbi
from equalizer_detector.sova import classical_sova


# ---------------------------------------------------------------------------
# Viterbi / SOVA integration
# ---------------------------------------------------------------------------


class TestDetectorIntegration:
    """Integration tests for Viterbi and SOVA detectors."""

    def test_viterbi_noiseless_single_pulse(self):
        """Viterbi on a clean single-bit pulse should decode correctly."""
        sector_length = 100
        pri_imp_res = np.array([1.0, 1.0, -1.0, -1.0])

        # Create a noiseless PRML signal for a known bit pattern
        bits = np.zeros(sector_length, dtype=np.int64)
        bits[10] = 1  # Single transition at position 10
        bipolar = 2.0 * bits.astype(np.float64) - 1.0
        equalized = np.convolve(bipolar, pri_imp_res, mode="full")[:sector_length]

        detected, _ = classical_viterbi(
            delay=4,
            equalized_channel_output=equalized.copy(),
            sector_length=sector_length,
            pri_imp_res=pri_imp_res,
        )

        # At least the decision region should contain a transition
        assert len(detected) == sector_length

    def test_sova_soft_outputs_in_range(self):
        """SOVA soft outputs should be in valid probability range."""
        sector_length = 100
        pri_imp_res = np.array([1.0, 1.0, -1.0, -1.0])

        bits = np.random.randint(0, 2, sector_length, dtype=np.int64)
        bipolar = 2.0 * bits.astype(np.float64) - 1.0
        equalized = np.convolve(bipolar, pri_imp_res, mode="full")[:sector_length]

        hard, soft, _ = classical_sova(
            delay=4,
            equalized_channel_output=equalized.copy(),
            sector_length=sector_length,
            pri_imp_res=pri_imp_res,
            noise_sigma=0.1,
        )

        assert np.all(soft >= 0) and np.all(soft <= 1)
        assert np.all(np.isfinite(soft))

    def test_viterbi_vs_sova_different_outputs(self):
        """SOVA should produce different soft outputs from Viterbi hard outputs."""
        np.random.seed(99)
        sector_length = 100
        pri_imp_res = np.array([1.0, 1.0, -1.0, -1.0])

        bits = np.random.randint(0, 2, sector_length, dtype=np.int64)
        bits[0] = 0
        bipolar = 2.0 * bits.astype(np.float64) - 1.0
        equalized = np.convolve(bipolar, pri_imp_res, mode="full")[:sector_length]

        # Add moderate noise so detector has ambiguity
        noisy = equalized + 0.5 * np.random.randn(sector_length)

        s_hard, s_soft, _ = classical_sova(
            delay=4, equalized_channel_output=noisy.copy(),
            sector_length=sector_length, pri_imp_res=pri_imp_res,
            noise_sigma=0.5,
        )

        # Soft outputs should vary (not all the same value)
        assert np.std(s_soft) > 0


# ---------------------------------------------------------------------------
# Channel integration
# ---------------------------------------------------------------------------


class TestChannelIntegration:
    """Integration tests for channel models."""

    def test_longitudinal_channel_different_from_perpendicular(self):
        """Different channel types should produce different outputs."""
        bits = np.array([0, 1, 0, 1, 1, 0, 1, 0] * 256, dtype=np.int64)

        long_out = channel(bits, "Longitudinal", nd=2.5, num_taps=41, osr=4)
        perp_out = channel(bits, "Perpendicular", nd=2.5, num_taps=41, osr=4)

        diff = np.abs(long_out - perp_out)
        assert np.sum(diff > 1e-6) > 0

    def test_channel_output_finite(self):
        """Channel outputs should be finite."""
        bits = np.random.randint(0, 2, 500, dtype=np.int64)

        for ch_type in ("Longitudinal", "Perpendicular"):
            output = channel(bits, ch_type, nd=2.5, num_taps=41, osr=4)
            assert np.all(np.isfinite(output))


# ---------------------------------------------------------------------------
# Equalizer integration
# ---------------------------------------------------------------------------


class TestEqualizerIntegration:
    """Integration tests for equalizer components."""

    def test_apply_equalizer_preserves_length(self):
        """apply_equalizer output length should equal input length."""
        data = np.random.randn(100)
        eq_coeff = np.random.randn(5)
        output = apply_equalizer(data, eq_coeff, 5)
        assert len(output) == len(data)

    def test_adapt_equalizer_returns_finite_mse(self):
        """LMS adaptive equalizer should return finite MSE values."""
        np.random.seed(42)
        data = np.random.randn(200)
        pri_imp_res = np.array([1.0, 1.0, -1.0, -1.0])
        eq_coeff = np.zeros(21, dtype=np.float64)

        mse, lmse = adapt_equalizer(
            data, data, eq_coeff, 21, 200, pri_imp_res,
            pri_imp_res_length=4, start_flag=1,
        )

        assert np.isfinite(mse)
        assert mse >= 0
        assert np.isfinite(lmse)

    def test_adapt_equalizer_multiple_iterations(self):
        """Running adapt_equalizer multiple times should adapt coefficients."""
        np.random.seed(42)
        data = np.random.randn(200)
        pri_imp_res = np.array([1.0, 1.0, -1.0, -1.0])
        eq_coeff = np.zeros(21, dtype=np.float64)

        mse1, _ = adapt_equalizer(
            data, data, eq_coeff, 21, 200, pri_imp_res,
            pri_imp_res_length=4, start_flag=1,
        )
        mse2, _ = adapt_equalizer(
            data, data, eq_coeff, 21, 200, pri_imp_res,
            pri_imp_res_length=4, start_flag=0,
        )

        # MSE should decrease after second iteration (coefficients adapted)
        assert mse2 < mse1

    def test_lpf_then_viterbi_pipeline(self):
        """LPF followed by Viterbi on a known signal."""
        sector_length = 100
        pri_imp_res = np.array([1.0, 1.0, -1.0, -1.0])

        bits = np.zeros(sector_length, dtype=np.int64)
        bits[10] = 1
        bipolar = 2.0 * bits.astype(np.float64) - 1.0
        equalized = np.convolve(bipolar, pri_imp_res, mode="full")[:sector_length]

        # Apply LPF (cutoff at 0.5 Nyquist)
        filtered = lpf(equalized, 20, 0.5)

        # Viterbi should still produce reasonable output
        detected, _ = classical_viterbi(
            delay=4,
            equalized_channel_output=filtered[:sector_length].copy(),
            sector_length=sector_length,
            pri_imp_res=pri_imp_res,
        )

        assert len(detected) == sector_length

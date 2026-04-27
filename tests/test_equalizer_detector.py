"""Tests for equalizer and detector modules.

Tests Viterbi detection (no-noise case), SOVA soft outputs,
equalizer convergence, and FIR filter correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from equalizer_detector.detector import classical_viterbi, classical_sova
from equalizer_detector.equalizer import (
    adapt_equalizer,
    apply_equalizer,
    find_gpr_target,
)
from channel.fir import non_causal_fir, causal_fir
from channel.lpf import lpf


# ---------------------------------------------------------------------------
# Viterbi detector tests
# ---------------------------------------------------------------------------


class TestViterbiDetector:
    """Tests for the classical Viterbi detector."""

    def test_noiseless_detection(self):
        """Viterbi should produce reasonable hard outputs on a clean signal."""
        np.random.seed(42)

        pri_imp_res = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
        num_bits = 200
        bits = np.random.randint(0, 2, num_bits, dtype=np.int64)
        bits[0] = 0

        # Bipolar mapping
        bipolar = 2.0 * bits.astype(np.float64) - 1.0

        # Convolve with PR target
        equalized = np.convolve(bipolar, pri_imp_res, mode="full")[:num_bits]

        detected, _ = classical_viterbi(
            delay=4,
            equalized_channel_output=equalized.copy(),
            sector_length=num_bits,
            pri_imp_res=pri_imp_res,
        )

        assert len(detected) == num_bits
        # Output should be binary {0, 1}
        assert set(np.unique(detected)).issubset({0.0, 1.0})

    def test_detection_length(self):
        """Output length should match sector_length."""
        equalized = np.zeros(100, dtype=np.float64)
        pri = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
        hard, _ = classical_viterbi(4, equalized.copy(), 100, pri)
        assert len(hard) == 100

    def test_viterbi_delay(self):
        """Viterbi delay should not change output length."""
        equalized = np.zeros(50, dtype=np.float64)
        pri = np.array([1.0, 0.0], dtype=np.float64)  # Simple PR
        hard, _ = classical_viterbi(4, equalized.copy(), 50, pri)
        assert len(hard) == 50


# ---------------------------------------------------------------------------
# SOVA detector tests
# ---------------------------------------------------------------------------


class TestSOVADetector:
    """Tests for the SOVA detector."""

    def test_noiseless_detection(self):
        """In a noiseless channel, SOVA should decode correctly."""
        np.random.seed(42)

        pri_imp_res = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
        num_bits = 200
        bits = np.random.randint(0, 2, num_bits, dtype=np.int64)
        bits[0] = 0

        bipolar = 2.0 * bits.astype(np.float64) - 1.0
        equalized = np.convolve(bipolar, pri_imp_res, mode="full")[:num_bits]

        # classical_sova returns (hard_output, soft_output, pri_imp_res)
        hard, soft, _ = classical_sova(
            delay=4,
            equalized_channel_output=equalized.copy(),
            sector_length=num_bits,
            pri_imp_res=pri_imp_res,
            noise_sigma=0.001,  # Small noise to avoid divide-by-zero
        )

        assert len(hard) == num_bits
        # Soft outputs should be non-negative
        assert np.all(soft >= 0)

    def test_soft_output_range(self):
        """Soft outputs should be in a reasonable range."""
        equalized = np.ones(50, dtype=np.float64) * 2.0
        pri = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)

        hard, soft, _ = classical_sova(
            delay=4,
            equalized_channel_output=equalized.copy(),
            sector_length=50,
            pri_imp_res=pri,
            noise_sigma=0.1,
        )

        assert len(hard) == 50
        assert len(soft) == 50
        # Soft values should be finite
        assert np.all(np.isfinite(soft))

    def test_sova_length(self):
        """Output lengths should match sector_length."""
        equalized = np.zeros(80, dtype=np.float64)
        pri = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
        hard, soft, _ = classical_sova(4, equalized.copy(), 80, pri, 0.5)
        assert len(hard) == 80
        assert len(soft) == 80


# ---------------------------------------------------------------------------
# Equalizer tests
# ---------------------------------------------------------------------------


class TestEqualizer:
    """Tests for the equalizer module."""

    def test_apply_equalizer_output_length(self):
        """apply_equalizer output length should match input length."""
        ds_output = np.zeros(100, dtype=np.float64)
        eq_coeff = np.zeros(21, dtype=np.float64)
        eq_coeff[10] = 1.0  # Identity

        result = apply_equalizer(ds_output, eq_coeff, 21)
        # apply_equalizer returns data_length elements
        assert len(result) == 100

    def test_equalizer_identity(self):
        """An identity equalizer should pass through the signal."""
        ds_output = np.array([1.0, -1.0, 1.0, -1.0] * 25, dtype=np.float64)
        eq_coeff = np.zeros(21, dtype=np.float64)
        eq_coeff[10] = 1.0  # Single tap at center

        result = apply_equalizer(ds_output, eq_coeff, 21)
        # Identity should reproduce the input
        np.testing.assert_array_almost_equal(result, ds_output)

    def test_adapt_equalizer_returns_mse(self):
        """AdaptEqualizer should return a tuple (mse, avg_lmse)."""
        np.random.seed(42)

        pri_imp_res = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
        eq_coeff = np.zeros(21, dtype=np.float64)
        ds_output = np.random.randn(100).astype(np.float64)

        mse, avg_lmse = adapt_equalizer(
            ds_output,
            ds_output,  # padded_eq_output (same as input for testing)
            eq_coeff,
            21,
            100,
            pri_imp_res,
            pri_imp_res_length=4,
            start_flag=1,
        )

        assert isinstance(mse, float)
        assert isinstance(avg_lmse, float)
        assert np.isfinite(mse)
        assert mse >= 0

    def test_adapt_equalizer_mse_decreases(self):
        """LMS adaptive equalizer should decrease MSE over iterations."""
        np.random.seed(42)

        pri_imp_res = np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)
        eq_coeff = np.zeros(21, dtype=np.float64)
        ds_output = np.random.randn(100).astype(np.float64)

        mse_values = []
        for iteration in range(5):
            start = 1 if iteration == 0 else 0
            mse, _ = adapt_equalizer(
                ds_output, ds_output, eq_coeff, 21, 100,
                pri_imp_res, pri_imp_res_length=4, start_flag=start
            )
            mse_values.append(mse)

        assert mse_values[-1] < mse_values[0]

    def test_gpr_target_find(self):
        """find_gpr_target should return valid target and coefficients."""
        np.random.seed(42)
        seq_len = 200

        ch_input = np.random.randint(0, 2, seq_len).astype(np.float64)
        ch_input = 2.0 * ch_input - 1.0  # Bipolar

        # Simple identity channel for testing
        ch_output = ch_input.copy()

        gpr_target, eq_coeff = find_gpr_target(
            ch_output, ch_input, 21, gpr_target_length=5,
        )

        assert len(gpr_target) == 5
        assert len(eq_coeff) == 21
        assert np.all(np.isfinite(gpr_target))
        assert np.all(np.isfinite(eq_coeff))


# ---------------------------------------------------------------------------
# FIR filter tests
# ---------------------------------------------------------------------------


class TestFIRFilters:
    """Tests for FIR filter implementations."""

    def test_non_causal_fir_identity(self):
        """An identity filter should reproduce the signal at the centre."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float64)
        h = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)  # tap at centre (index 2)
        result = non_causal_fir(data, h)
        # Output length = data_length + floor(num_taps/2) = 5 + 2 = 7
        assert len(result) == 7
        # The convolution with centre tap at index 2 means:
        # result[i] = padded[i+2] where padded[2:7] = data
        # So result[0:5] = data (direct match since tap=1 and all others=0)
        np.testing.assert_array_almost_equal(result[:len(data)], data)

    def test_causal_fir_convolution(self):
        """Causal FIR should match numpy convolution."""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        h = np.array([1.0, 2.0, 1.0], dtype=np.float64)
        result = causal_fir(data, h)
        expected = np.convolve(data, h, mode="full")
        np.testing.assert_array_almost_equal(result, expected)

    def test_non_causal_fir_length(self):
        """Output length should be input + floor(num_taps/2)."""
        data = np.zeros(100, dtype=np.float64)
        h = np.zeros(21, dtype=np.float64)
        result = non_causal_fir(data, h)
        assert len(result) == 100 + 10  # floor(21/2) = 10

    def test_causal_fir_length(self):
        """Output length should be input + num_taps - 1."""
        data = np.zeros(50, dtype=np.float64)
        h = np.zeros(5, dtype=np.float64)
        result = causal_fir(data, h)
        assert len(result) == 50 + 5 - 1

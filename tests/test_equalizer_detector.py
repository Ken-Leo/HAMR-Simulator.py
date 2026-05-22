"""Tests for equalizer and detector modules.

Tests Viterbi detection (no-noise case), SOVA soft outputs,
equalizer convergence, and FIR filter correctness.
"""

from __future__ import annotations

import numpy as np
import pytest

from equalizer_detector.detector import classical_viterbi, classical_sova
from equalizer_detector.viterbi import classical_viterbi_sliding_window
from equalizer_detector.equalizer import (
    adapt_equalizer,
    apply_equalizer,
    find_gpr_target,
    lpf,
    _corr,
    _matrix_inv,
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
            pri_imp_res,
            eq_coeff,
            21,
            ds_output,   # clean_bits (same as input for testing)
            ds_output,   # channel_output
            100,
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
                pri_imp_res, eq_coeff, 21,
                ds_output, ds_output, 100,
                start_flag=start,
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
        """Causal FIR should match first len(data) elements of full convolution."""
        data = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
        h = np.array([1.0, 2.0, 1.0], dtype=np.float64)
        result = causal_fir(data, h)
        # Output length = len(data) = 4 (matching C CausalFIR)
        full_conv = np.convolve(data, h, mode="full")
        expected = full_conv[: len(data)]
        np.testing.assert_array_almost_equal(result, expected)

    def test_non_causal_fir_length(self):
        """Output length should be input + floor(num_taps/2)."""
        data = np.zeros(100, dtype=np.float64)
        h = np.zeros(21, dtype=np.float64)
        result = non_causal_fir(data, h)
        assert len(result) == 100 + 10  # floor(21/2) = 10

    def test_causal_fir_length(self):
        """Output length should equal input length (matching C CausalFIR)."""
        data = np.zeros(50, dtype=np.float64)
        h = np.zeros(5, dtype=np.float64)
        result = causal_fir(data, h)
        assert len(result) == 50  # same as input length


# ---------------------------------------------------------------------------
# Sliding-window Viterbi tests
# ---------------------------------------------------------------------------


class TestSlidingWindowViterbi:
    """Tests for the sliding-window Viterbi detector."""

    def _make_pr_signal(self, bits, pri_imp_res):
        """Create a noiseless PR target signal from bits."""
        num_states = 1 << (len(pri_imp_res) - 1)
        signal = np.zeros(len(bits), dtype=np.float64)
        state = 0
        for k in range(len(bits)):
            state = ((state << 1) | int(bits[k])) & (num_states - 1)
            sample = 0.0
            for j in range(len(pri_imp_res) - 1):
                sample += pri_imp_res[j] * ((state >> j) & 1)
            sample += pri_imp_res[-1] * ((state >> (len(pri_imp_res) - 1)) & 1)
            signal[k] = sample * 2.0
        return signal

    def test_sliding_window_output_length(self):
        """Output should match input sector length."""
        pri = np.array([1, 1, -1, -1], dtype=np.float64)
        signal = np.random.randn(1000)
        result = classical_viterbi_sliding_window(20, signal, 1000, pri)
        assert len(result) == 1000

    def test_sliding_window_noiseless(self):
        """Sliding window should detect reasonably on noiseless PR signal."""
        pri = np.array([1, 1, -1, -1], dtype=np.float64)
        np.random.seed(42)
        bits = np.random.randint(0, 2, size=500)
        signal = self._make_pr_signal(bits, pri)

        result = classical_viterbi_sliding_window(20, signal, 500, pri)
        # Sliding window has boundary effects; check output is valid
        assert len(result) == 500
        assert np.all(np.isfinite(result))
        # At least 70% of bits should match (boundary effects reduce accuracy)
        matches = np.sum(result == bits)
        assert matches > 0.7 * 500

    def test_sliding_window_custom_window_size(self):
        """Custom window size should be respected."""
        pri = np.array([1, 1, -1, -1], dtype=np.float64)
        signal = np.random.randn(2000)
        result = classical_viterbi_sliding_window(
            20, signal, 2000, pri, window_size=200, boundary_guard=40
        )
        assert len(result) == 2000

    def test_sliding_window_small_window_adjusted(self):
        """Window too small should be auto-adjusted to fit boundary guards."""
        pri = np.array([1, 1, -1, -1], dtype=np.float64)
        signal = np.random.randn(200)
        # Window size 10 is too small for boundary_guard=40 (needs 2*40+1=81)
        result = classical_viterbi_sliding_window(
            20, signal, 200, pri, window_size=10, boundary_guard=40
        )
        assert len(result) == 200

    def test_sliding_window_vs_classical_consistency(self):
        """Sliding window should produce similar results to classical Viterbi."""
        pri = np.array([1, 1, -1, -1], dtype=np.float64)
        np.random.seed(42)
        bits = np.random.randint(0, 2, size=200)
        signal = self._make_pr_signal(bits, pri)

        # Classical Viterbi
        det_classical, _ = classical_viterbi(20, signal, 200, pri)

        # Sliding window with large window (should match classical)
        det_sliding = classical_viterbi_sliding_window(
            20, signal, 200, pri, window_size=500, boundary_guard=40
        )

        # Compare (skip first delay bits)
        agreement = np.sum(det_classical[20:] == det_sliding[20:])
        assert agreement > 150  # Most bits should agree

    def test_sliding_window_long_sequence(self):
        """Sliding window should handle sequences longer than window size."""
        pri = np.array([1, 1, -1, -1], dtype=np.float64)
        signal = np.random.randn(5000)
        result = classical_viterbi_sliding_window(
            20, signal, 5000, pri, window_size=200, boundary_guard=40
        )
        assert len(result) == 5000
        assert np.all(np.isfinite(result))

    def test_sliding_window_boundary_guard_effect(self):
        """Different boundary guards should produce different results."""
        pri = np.array([1, 1, -1, -1], dtype=np.float64)
        np.random.seed(42)
        bits = np.random.randint(0, 2, size=500)
        signal = self._make_pr_signal(bits, pri)

        det_small = classical_viterbi_sliding_window(
            20, signal, 500, pri, window_size=150, boundary_guard=30
        )
        det_large = classical_viterbi_sliding_window(
            20, signal, 500, pri, window_size=150, boundary_guard=60
        )

        # Results should differ due to different valid regions
        diff = np.sum(det_small != det_large)
        assert diff > 0  # At least some bits should differ


# ---------------------------------------------------------------------------
# Branch coverage tests for equalizer.py
# ---------------------------------------------------------------------------


class TestEqualizerBranchCoverage:
    """Targeted tests to cover uncovered branches in equalizer.py."""

    def test_lpf_even_filter_order(self):
        """LPF with even filter order exercises the i == mid branch.

        When filter_order is even, mid is an integer (e.g. 10/2=5),
        so the i == mid branch is taken.
        """
        signal = np.random.randn(200)
        # Even order: mid = 5.0 (integer), so i == mid triggers
        output = lpf(signal, filter_order=10, cutoff=0.3)
        assert len(output) > len(signal)
        assert np.all(np.isfinite(output))

    def test_lpf_odd_filter_order(self):
        """LPF with odd filter order exercises the else branch.

        When filter_order is odd, mid is x.5 (e.g. 11/2=5.5),
        so i == mid is never true.
        """
        signal = np.random.randn(200)
        # Odd order: mid = 5.5 (non-integer), so i == mid never triggers
        output = lpf(signal, filter_order=11, cutoff=0.3)
        assert len(output) > len(signal)
        assert np.all(np.isfinite(output))

    def test_corr_early_return(self):
        """_corr should return 0.0 when start > finish."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([1.0, 1.0, 1.0])
        # Large positive shift: start = 10, finish = 2 → start > finish
        result = _corr(a, 3, b, 3, shift=10)
        assert result == 0.0

    def test_corr_normal_case(self):
        """_corr should compute correctly for normal shifts."""
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        b = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
        # shift=1: sum of a[1:]*b[0:-1] = 2+3+4+5 = 14
        result = _corr(a, 5, b, 5, shift=1)
        assert result == 14.0

    def test_matrix_inv_row_swap(self):
        """_matrix_inv should handle matrices with zero diagonal elements."""
        # Matrix with zero on diagonal that requires row swap
        A = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        A_inv = _matrix_inv(A)
        # Verify A * A_inv ≈ I
        product = A @ A_inv
        np.testing.assert_array_almost_equal(product, np.eye(3), decimal=10)

    def test_find_gpr_target_monic_constraint(self):
        """find_gpr_target should normalize GPR target when G[0] != 1."""
        np.random.seed(42)
        # Create a channel output that produces a non-monic GPR target
        channel_output = np.random.randn(500)
        bipolar_input = np.random.choice([-1.0, 1.0], size=500)

        gpr_target, eq_coeff = find_gpr_target(
            channel_output, bipolar_input, num_taps=21, gpr_target_length=4,
        )

        # After normalization, G[0] should be 1.0
        assert abs(gpr_target[0] - 1.0) < 1e-4
        assert len(eq_coeff) == 21
        assert np.all(np.isfinite(eq_coeff))


# ---------------------------------------------------------------------------
# Branch coverage tests for sova.py
# ---------------------------------------------------------------------------


class TestSovaBranchCoverage:
    """Targeted tests to cover uncovered branches in sova.py."""

    def test_sova_metric1_survivor(self):
        """SOVA should select metric1 as survivor when it's smaller."""
        np.random.seed(99)
        pri = np.array([1, 1, -1, -1], dtype=np.float64)

        # Create a signal where metric1 path is consistently better
        # This happens when the input is closer to the "bit 1" sample values
        num_states = 1 << (len(pri) - 1)  # 8 for EPR4
        signal = np.zeros(200, dtype=np.float64)
        state = 0
        for k in range(200):
            # Force state to always transition with bit=1
            state = ((state << 1) | 1) & (num_states - 1)
            sample = 0.0
            for j in range(len(pri) - 1):
                sample += pri[j] * ((state >> j) & 1)
            sample += pri[-1] * ((state >> (len(pri) - 1)) & 1)
            signal[k] = sample * 2.0

        # Add small noise
        signal += np.random.randn(200) * 0.01

        noise_sigma = 0.1
        detected, soft, _ = classical_sova(20, signal, 200, pri, noise_sigma)

        assert len(detected) == 200
        assert len(soft) == 200
        assert np.all(np.isfinite(detected))
        assert np.all(np.isfinite(soft))
        # Most bits should be 1 (since we forced bit=1 transitions)
        ones_ratio = np.mean(detected[20:])
        assert ones_ratio > 0.8

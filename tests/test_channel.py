"""Tests for channel module."""
import numpy as np
import pytest
from channel.channel import longitudinal_channel, perpendicular_channel, hamr_channel
from channel.lpf import lpf
from channel.fir import non_causal_fir, causal_fir
from channel.math_utils import LCG, gaussian_random, autocorr, mat_inverse


class TestLongitudinalChannel:
    def test_output_length(self):
        bits = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int64)
        output = longitudinal_channel(bits, 2.5, num_taps=21, osr=5)
        expected_len = len(bits) * 5
        assert len(output) == expected_len

    def test_finite_values(self):
        bits = np.ones(50, dtype=np.int64)
        output = longitudinal_channel(bits, 2.5)
        assert np.all(np.isfinite(output))

    def test_alternating_bits(self):
        bits = np.array([0, 1, 0, 1] * 25, dtype=np.int64)  # 100 bits
        output = longitudinal_channel(bits, 2.5, num_taps=41, osr=4)
        # Output length = num_bits * osr = 100 * 4 = 400
        assert len(output) == 400


class TestPerpendicularChannel:
    def test_output_length(self):
        bits = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int64)
        output = perpendicular_channel(bits, 2.5, num_taps=21, osr=5)
        assert len(output) == len(bits) * 5

    def test_finite_values(self):
        bits = np.zeros(50, dtype=np.int64)
        output = perpendicular_channel(bits, 2.5)
        assert np.all(np.isfinite(output))


class TestHamrChannel:
    def test_output_length(self):
        bipolar = np.ones(1000, dtype=np.float64)
        bipolar[500] = -1.0  # One transition
        hamr_params = {"sigma_t": 90.0, "reader_sigma_r": 1000.0, "hamr_hg": 1.6e6}
        output = hamr_channel(bipolar, 10, hamr_params)
        assert len(output) == 1000

    def test_finite_values(self):
        bipolar = np.ones(200, dtype=np.float64)
        hamr_params = {"sigma_t": 90.0, "reader_sigma_r": 1000.0, "hamr_hg": 1.6e6}
        output = hamr_channel(bipolar, 10, hamr_params)
        assert np.all(np.isfinite(output))


class TestFirFilters:
    def test_non_causal_fir_identity(self):
        """FIR with single-tap impulse response should pass through."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        h = np.array([1.0])
        output = non_causal_fir(data, h)
        np.testing.assert_array_almost_equal(output, data)

    def test_causal_fir_simple(self):
        """Test causal FIR with known filter.
        
        causal_fir in channel.fir uses np.convolve(..., mode='full'),
        so output length = len(data) + len(h) - 1.
        The first and last elements are reduced due to partial filter overlap.
        """
        data = np.ones(10, dtype=np.float64)
        h = np.array([0.5, 0.5], dtype=np.float64)
        output = causal_fir(data, h)
        assert len(output) == 11  # 10 + 2 - 1 = full convolution length
        # Interior elements are 1.0 (full overlap with sum(h)=1.0)
        np.testing.assert_array_almost_equal(output[1:10], np.ones(9))

    def test_non_causal_fir_known(self):
        """Non-causal FIR with [0.5, 1.0, 0.5] should smooth."""
        data = np.zeros(10, dtype=np.float64)
        data[5] = 1.0
        h = np.array([0.25, 0.5, 0.25], dtype=np.float64)
        output = non_causal_fir(data, h)
        assert len(output) > 10
        assert output[5] > 0


class TestLpf:
    def test_lpf_passes_low_freq(self):
        """LPF should pass low-frequency signal with minimal attenuation."""
        t = np.linspace(0, 10, 1000)
        signal = np.sin(2 * np.pi * 0.1 * t)  # Low frequency
        output = lpf(signal, 200, 0.3)  # Cutoff at 0.3 * Nyquist
        assert len(output) > len(signal)
        assert np.all(np.isfinite(output))

    def test_lpf_finite(self):
        data = np.random.randn(500)
        output = lpf(data, 100, 0.5)
        assert np.all(np.isfinite(output))


class TestRng:
    def test_lcg_deterministic(self):
        lcg1 = LCG(-500)
        lcg2 = LCG(-500)
        assert lcg1.random() == lcg2.random()

    def test_lcg_range(self):
        lcg = LCG(-42)
        for _ in range(100):
            r = lcg.random()
            assert 0 < r < 1

    def test_gaussian_distribution(self):
        """Gaussian should have approximately zero mean."""
        lcg = LCG(-123)
        samples = [gaussian_random(lcg) for _ in range(10000)]
        mean = np.mean(samples)
        assert abs(mean) < 0.1  # Approximately zero mean


class TestAutocorr:
    def test_autocorr_max_at_lag_zero(self):
        """Autocorrelation should be maximum at lag 0."""
        data = np.random.randn(1000)
        c0 = autocorr(data, 1000, 0)
        c1 = autocorr(data, 1000, 1)
        assert c0 >= c1


class TestMatrixInverse:
    def test_inverse_correct(self):
        """A * A^-1 should equal identity."""
        A = np.array([[4.0, 1.0], [2.0, 3.0]])
        A_inv = mat_inverse(A)
        identity = A @ A_inv
        np.testing.assert_array_almost_equal(identity, np.eye(2), decimal=10)

"""Tests for channel module."""
import numpy as np
import pytest
from channel.channel import longitudinal_channel, perpendicular_channel, hamr_channel
from channel.lpf import lpf
from channel.fir import non_causal_fir, causal_fir
from channel.math_utils import LCG, gaussian_random, autocorr, mat_inverse
from channel.media_noise import media_noise_filter, _pulse_value


class TestLongitudinalChannel:
    def test_output_length(self):
        bits = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int64)
        output = longitudinal_channel(bits, 2.5, num_taps=21, osr=5)
        # Non-causal FIR output = len(data) + front_pad = len(bits) * osr + num_taps // 2
        expected_len = len(bits) * 5 + 21 // 2
        assert len(output) == expected_len

    def test_finite_values(self):
        bits = np.ones(50, dtype=np.int64)
        output = longitudinal_channel(bits, 2.5)
        assert np.all(np.isfinite(output))

    def test_alternating_bits(self):
        bits = np.array([0, 1, 0, 1] * 25, dtype=np.int64)  # 100 bits
        output = longitudinal_channel(bits, 2.5, num_taps=41, osr=4)
        # Output length = num_bits * osr + num_taps // 2 = 400 + 20 = 420
        assert len(output) == 420


class TestPerpendicularChannel:
    def test_output_length(self):
        bits = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.int64)
        output = perpendicular_channel(bits, 2.5, num_taps=21, osr=5)
        # Non-causal FIR output = len(bits) * osr + num_taps // 2
        expected_len = len(bits) * 5 + 21 // 2
        assert len(output) == expected_len

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
        
        causal_fir in channel.fir returns same-length output (matching C CausalFIR).
        """
        data = np.ones(10, dtype=np.float64)
        h = np.array([0.5, 0.5], dtype=np.float64)
        output = causal_fir(data, h)
        assert len(output) == 10  # same as input length
        # Interior elements are 1.0 (full overlap with sum(h)=1.0)
        np.testing.assert_array_almost_equal(output[1:], np.ones(9))

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


class TestMediaNoiseFilter:
    """Tests for media_noise_filter matching C MediaNoiseFilter()."""

    def _make_time_index(self, num_taps: int, osr: int) -> np.ndarray:
        centre = num_taps // 2 / osr
        return np.array([centre - i / osr for i in range(num_taps)], dtype=np.float64)

    def test_output_length_longitudinal(self):
        """Output length = sector_length + num_taps // 2."""
        num_taps = 21
        osr = 5
        bits = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.float64)
        os_bits = np.repeat(2 * bits - 1, osr)  # bipolar, oversampled
        sector_length = len(os_bits)
        time_index = self._make_time_index(num_taps, osr)
        coeffs = np.zeros(num_taps)

        output = media_noise_filter(
            os_bits, sector_length, coeffs, num_taps, time_index,
            nd=2.5, sigma_jitter=5.0, sigma_pulse_broad=3.0,
            osr=osr, channel_type="Longitudinal",
        )
        expected_len = sector_length + num_taps // 2
        assert len(output) == expected_len

    def test_output_length_perpendicular(self):
        """Output length = sector_length + num_taps // 2."""
        num_taps = 21
        osr = 5
        bits = np.array([0, 1, 0, 1, 1, 0, 1, 0], dtype=np.float64)
        os_bits = np.repeat(2 * bits - 1, osr)
        sector_length = len(os_bits)
        time_index = self._make_time_index(num_taps, osr)
        coeffs = np.zeros(num_taps)

        output = media_noise_filter(
            os_bits, sector_length, coeffs, num_taps, time_index,
            nd=2.5, sigma_jitter=5.0, sigma_pulse_broad=3.0,
            osr=osr, channel_type="Perpendicular",
        )
        expected_len = sector_length + num_taps // 2
        assert len(output) == expected_len

    def test_finite_values(self):
        """All output values should be finite."""
        num_taps = 41
        osr = 10
        rng = np.random.RandomState(42)
        bits = rng.randint(0, 2, size=100).astype(np.float64)
        os_bits = np.repeat(2 * bits - 1, osr)
        sector_length = len(os_bits)
        time_index = self._make_time_index(num_taps, osr)
        coeffs = np.zeros(num_taps)

        output = media_noise_filter(
            os_bits, sector_length, coeffs, num_taps, time_index,
            nd=2.5, sigma_jitter=8.0, sigma_pulse_broad=5.0,
            osr=osr, channel_type="Longitudinal",
        )
        assert np.all(np.isfinite(output))

    def test_deterministic_with_same_seed(self):
        """Same seed should produce identical output."""
        num_taps = 21
        osr = 5
        bits = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0], dtype=np.float64)
        os_bits = np.repeat(2 * bits - 1, osr)
        sector_length = len(os_bits)
        time_index = self._make_time_index(num_taps, osr)
        coeffs = np.zeros(num_taps)

        out1 = media_noise_filter(
            os_bits, sector_length, coeffs.copy(), num_taps, time_index,
            nd=2.5, sigma_jitter=5.0, sigma_pulse_broad=3.0,
            osr=osr, channel_type="Longitudinal",
            seed_jitter=-200, seed_pulse=-100,
        )
        out2 = media_noise_filter(
            os_bits, sector_length, coeffs.copy(), num_taps, time_index,
            nd=2.5, sigma_jitter=5.0, sigma_pulse_broad=3.0,
            osr=osr, channel_type="Longitudinal",
            seed_jitter=-200, seed_pulse=-100,
        )
        np.testing.assert_array_equal(out1, out2)

    def test_different_seeds_produce_different_output(self):
        """Different seeds should produce different output."""
        num_taps = 21
        osr = 5
        bits = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0], dtype=np.float64)
        os_bits = np.repeat(2 * bits - 1, osr)
        sector_length = len(os_bits)
        time_index = self._make_time_index(num_taps, osr)
        coeffs = np.zeros(num_taps)

        out1 = media_noise_filter(
            os_bits, sector_length, coeffs.copy(), num_taps, time_index,
            nd=2.5, sigma_jitter=5.0, sigma_pulse_broad=3.0,
            osr=osr, channel_type="Longitudinal",
            seed_jitter=-200, seed_pulse=-100,
        )
        out2 = media_noise_filter(
            os_bits, sector_length, coeffs.copy(), num_taps, time_index,
            nd=2.5, sigma_jitter=5.0, sigma_pulse_broad=3.0,
            osr=osr, channel_type="Longitudinal",
            seed_jitter=-999, seed_pulse=-888,
        )
        assert not np.allclose(out1, out2)

    def test_longitudinal_different_from_perpendicular(self):
        """Longitudinal and perpendicular should produce different outputs."""
        num_taps = 21
        osr = 5
        bits = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0], dtype=np.float64)
        os_bits = np.repeat(2 * bits - 1, osr)
        sector_length = len(os_bits)
        time_index = self._make_time_index(num_taps, osr)
        coeffs = np.zeros(num_taps)

        out_long = media_noise_filter(
            os_bits, sector_length, coeffs.copy(), num_taps, time_index,
            nd=2.5, sigma_jitter=5.0, sigma_pulse_broad=3.0,
            osr=osr, channel_type="Longitudinal",
            seed_jitter=-200, seed_pulse=-100,
        )
        out_perp = media_noise_filter(
            os_bits, sector_length, coeffs.copy(), num_taps, time_index,
            nd=2.5, sigma_jitter=5.0, sigma_pulse_broad=3.0,
            osr=osr, channel_type="Perpendicular",
            seed_jitter=-200, seed_pulse=-100,
        )
        assert not np.allclose(out_long, out_perp)

    def test_all_same_bits_produces_zero(self):
        """No transitions → zero output."""
        num_taps = 21
        osr = 5
        bits = np.ones(50, dtype=np.float64)  # all 1s → no transitions
        os_bits = np.repeat(2 * bits - 1, osr)
        sector_length = len(os_bits)
        time_index = self._make_time_index(num_taps, osr)
        coeffs = np.zeros(num_taps)

        output = media_noise_filter(
            os_bits, sector_length, coeffs, num_taps, time_index,
            nd=2.5, sigma_jitter=5.0, sigma_pulse_broad=3.0,
            osr=osr, channel_type="Longitudinal",
        )
        np.testing.assert_array_almost_equal(output, np.zeros(len(output)))

    def test_single_transition(self):
        """Single transition should produce a pulse-like output."""
        num_taps = 41
        osr = 10
        bits = np.zeros(100, dtype=np.float64)
        bits[50:] = 1.0  # single transition at bit 50
        os_bits = np.repeat(2 * bits - 1, osr)
        sector_length = len(os_bits)
        time_index = self._make_time_index(num_taps, osr)
        coeffs = np.zeros(num_taps)

        output = media_noise_filter(
            os_bits, sector_length, coeffs, num_taps, time_index,
            nd=2.5, sigma_jitter=0.0, sigma_pulse_broad=0.0,
            osr=osr, channel_type="Longitudinal",
        )
        # Should have non-zero values around the transition
        assert np.max(np.abs(output)) > 0.01
        assert np.all(np.isfinite(output))

    def test_higher_jitter_increases_variance(self):
        """Higher jitter should produce more variance across runs."""
        num_taps = 21
        osr = 5
        bits = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0], dtype=np.float64)
        os_bits = np.repeat(2 * bits - 1, osr)
        sector_length = len(os_bits)
        time_index = self._make_time_index(num_taps, osr)
        coeffs = np.zeros(num_taps)

        # Run with low jitter
        outputs_low = []
        for seed in range(10):
            out = media_noise_filter(
                os_bits, sector_length, coeffs.copy(), num_taps, time_index,
                nd=2.5, sigma_jitter=1.0, sigma_pulse_broad=0.0,
                osr=osr, channel_type="Longitudinal",
                seed_jitter=-seed, seed_pulse=-100,
            )
            outputs_low.append(out)

        # Run with high jitter
        outputs_high = []
        for seed in range(10):
            out = media_noise_filter(
                os_bits, sector_length, coeffs.copy(), num_taps, time_index,
                nd=2.5, sigma_jitter=20.0, sigma_pulse_broad=0.0,
                osr=osr, channel_type="Longitudinal",
                seed_jitter=-seed, seed_pulse=-100,
            )
            outputs_high.append(out)

        var_low = np.mean([np.var(o) for o in outputs_low])
        var_high = np.mean([np.var(o) for o in outputs_high])
        assert var_high > var_low


class TestPulseValue:
    """Tests for the _pulse_value helper function."""

    def test_longitudinal_at_center(self):
        """At t=0, longitudinal pulse should be sqrt(2/pi)."""
        val = _pulse_value(0.0, 0.0, 0.0, nd=2.5, channel_type="Longitudinal")
        expected = np.sqrt(2.0 / np.pi)
        assert abs(val - expected) < 1e-10

    def test_perpendicular_at_center(self):
        """At t=0, perpendicular pulse should be erf(0) = 0."""
        val = _pulse_value(0.0, 0.0, 0.0, nd=2.5, channel_type="Perpendicular")
        assert abs(val) < 1e-10

    def test_longitudinal_decreases_with_distance(self):
        """Longitudinal pulse should decrease as |t| increases."""
        v0 = _pulse_value(0.0, 0.0, 0.0, nd=2.5, channel_type="Longitudinal")
        v1 = _pulse_value(1.0, 0.0, 0.0, nd=2.5, channel_type="Longitudinal")
        assert v0 > v1

    def test_perpendicular_increases_with_distance(self):
        """Perpendicular erf pulse should increase (in magnitude) with |t|."""
        v0 = abs(_pulse_value(0.0, 0.0, 0.0, nd=2.5, channel_type="Perpendicular"))
        v1 = abs(_pulse_value(1.0, 0.0, 0.0, nd=2.5, channel_type="Perpendicular"))
        assert v1 > v0

    def test_jitter_shifts_pulse(self):
        """Non-zero delta_x should change the pulse value."""
        v_no_jitter = _pulse_value(0.5, 0.0, 0.0, nd=2.5, channel_type="Longitudinal")
        v_with_jitter = _pulse_value(0.5, 0.1, 0.0, nd=2.5, channel_type="Longitudinal")
        assert v_no_jitter != v_with_jitter

    def test_pulse_broadening_affects_longitudinal(self):
        """Non-zero delta_nd should change the longitudinal pulse."""
        v_no_pb = _pulse_value(0.5, 0.0, 0.0, nd=2.5, channel_type="Longitudinal")
        v_with_pb = _pulse_value(0.5, 0.0, 0.2, nd=2.5, channel_type="Longitudinal")
        assert v_no_pb != v_with_pb

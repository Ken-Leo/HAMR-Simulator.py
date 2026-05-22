"""Tests for permutation and math_utils modules."""
import math
import numpy as np
import pytest
from channel.math_utils import (
    LCG, gaussian_random, gaussian_raw, autocorr, cross_corr,
    make_symmetric, solve_linear_system, mat_inverse, matrix_inv,
    mat_mult, matrix_mult, transpose, gammp, erf, uniform_random,
    PI,
)
from encoders.permutation import count_possible_error_events, permute


# ===========================================================================
# Permutation module tests
# ===========================================================================

class TestCountPossibleErrorEvents:
    """Tests for count_possible_error_events (tribit counting)."""

    def test_no_transitions(self):
        """All same bits → 0 tribits."""
        bits = np.array([0, 0, 0, 0, 0], dtype=np.int64)
        assert count_possible_error_events(bits, len(bits)) == 0

    def test_single_transition(self):
        """Single 0→1 transition → 0 tribits."""
        bits = np.array([0, 0, 1, 1, 1], dtype=np.int64)
        assert count_possible_error_events(bits, len(bits)) == 0

    def test_two_transitions(self):
        """Two transitions (0→1→0) → 0 tribits (need 3+)."""
        bits = np.array([0, 1, 0, 0, 0], dtype=np.int64)
        assert count_possible_error_events(bits, len(bits)) == 0

    def test_three_consecutive_transitions(self):
        """Three transitions (0→1→0→1) → 3 tribits (4 transitions - 1 = 3)."""
        bits = np.array([0, 1, 0, 1, 0], dtype=np.int64)
        # 4 consecutive transitions: positions 1,2,3,4 → 4-1 = 3
        assert count_possible_error_events(bits, len(bits)) == 3

    def test_four_consecutive_transitions(self):
        """Four transitions → 4 tribits (5 transitions - 1 = 4)."""
        bits = np.array([0, 1, 0, 1, 0, 1], dtype=np.int64)
        # 5 consecutive transitions → 5-1 = 4
        assert count_possible_error_events(bits, len(bits)) == 4

    def test_five_consecutive_transitions(self):
        """Five transitions → 5 tribits."""
        bits = np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
        # 6 consecutive transitions → 6-1 = 5
        assert count_possible_error_events(bits, len(bits)) == 5

    def test_multiple_tribit_groups(self):
        """Two separate groups of tribits."""
        # Group 1: 0→1→0→1→0 (4 trans → 3 tribits)
        # Gap: 0→0→0 (no transition)
        # Group 2: 0→1→0→1→0 (4 trans → 3 tribits)
        bits = np.array([0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0], dtype=np.int64)
        assert count_possible_error_events(bits, len(bits)) == 6

    def test_length_parameter_limits_scan(self):
        """Only first `length` bits should be examined."""
        bits = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        # First 5 bits: 0,1,0,1,0 → 4 transitions → 3 tribits
        assert count_possible_error_events(bits, 5) == 3
        # First 3 bits: 0,1,0 → 2 transitions → 0 tribits
        assert count_possible_error_events(bits, 3) == 0

    def test_empty_sequence(self):
        """Length 0 → 0 tribits."""
        bits = np.array([], dtype=np.int64)
        assert count_possible_error_events(bits, 0) == 0

    def test_single_bit(self):
        """Single bit → 0 tribits."""
        bits = np.array([1], dtype=np.int64)
        assert count_possible_error_events(bits, 1) == 0

    def test_alternating_pattern_long(self):
        """Long alternating pattern."""
        bits = np.array([0, 1] * 25, dtype=np.int64)  # 50 bits, 49 transitions
        assert count_possible_error_events(bits, 50) == 48


class TestPermute:
    """Tests for the permute function (optimal cyclic shift)."""

    def test_identity_permutation(self):
        """Identity permutation should return shift 0."""
        word = np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int64)
        p = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        n_shifts, result = permute(word, 8, p, 8)
        assert n_shifts == 0
        np.testing.assert_array_equal(result, word)

    def test_returns_correct_length(self):
        """Output should have sector_length elements."""
        word = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        p = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        n_shifts, result = permute(word, 8, p, 8)
        assert len(result) == 8

    def test_returns_numpy_array(self):
        """Output should be a numpy array."""
        word = np.array([0, 1, 0, 1], dtype=np.int64)
        p = np.array([0, 1, 2, 3])
        n_shifts, result = permute(word, 4, p, 4)
        assert isinstance(result, np.ndarray)

    def test_shift_returns_tuple(self):
        """Should return (num_permutations, result) tuple."""
        word = np.array([0, 1, 0, 1], dtype=np.int64)
        p = np.array([0, 1, 2, 3])
        result = permute(word, 4, p, 4)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_permutation_preserves_values(self):
        """Permutation should preserve the set of values."""
        word = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)
        p = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        n_shifts, result = permute(word, 8, p, 8)
        assert sorted(result.tolist()) == sorted(word.tolist())

    def test_cyclic_shift_reduces_tribits(self):
        """A good shift should reduce tribit count."""
        # Create a word where a specific shift reduces tribits
        # 0,1,0,1,0 has 4 transitions → 3 tribits
        # After shift, if we can group transitions differently, tribits may reduce
        word = np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype=np.int64)
        p = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        n_shifts, result = permute(word, 8, p, 8)
        # The function should find the best shift
        assert isinstance(n_shifts, int)
        assert 0 <= n_shifts < 8

    def test_period_less_than_length(self):
        """Period can be less than sector_length."""
        word = np.array([0, 1, 0, 1, 0, 0, 0, 0], dtype=np.int64)
        p = np.array([0, 1, 2, 3])  # Only permute first 4 positions
        n_shifts, result = permute(word, 8, p, 4)
        assert 0 <= n_shifts < 4
        assert len(result) == 8

    def test_single_element_permutation(self):
        """Single element permutation should return shift 0."""
        word = np.array([1], dtype=np.int64)
        p = np.array([0])
        n_shifts, result = permute(word, 1, p, 1)
        assert n_shifts == 0
        assert result[0] == 1


# ===========================================================================
# Math utils tests
# ===========================================================================

class TestConstants:
    """Tests for mathematical constants."""

    def test_pi_value(self):
        """PI should match math.pi."""
        assert PI == math.pi


class TestLCGReset:
    """Tests for LCG reset functionality."""

    def test_reset_changes_sequence(self):
        """Resetting with different seed should change the sequence."""
        lcg = LCG(-100)
        first_val = lcg.random()
        lcg.reset(-200)
        assert lcg.random() != first_val

    def test_reset_same_seed_same_sequence(self):
        """Resetting with same seed should reproduce the sequence."""
        lcg = LCG(-100)
        vals1 = [lcg.random() for _ in range(10)]
        lcg.reset(-100)
        vals2 = [lcg.random() for _ in range(10)]
        assert vals1 == vals2

    def test_reset_with_positive_seed(self):
        """Reset should handle positive seeds (converted to negative)."""
        lcg = LCG(100)
        lcg.reset(100)
        r = lcg.random()
        assert 0 < r < 1


class TestGaussianRaw:
    """Tests for gaussian_raw function."""

    def test_returns_float(self):
        """Should return a float."""
        lcg = LCG(-42)
        result = gaussian_raw(lcg)
        assert isinstance(result, float)

    def test_different_from_gaussian_random(self):
        """gaussian_raw and gaussian_random should produce same distribution."""
        lcg1 = LCG(-42)
        lcg2 = LCG(-42)
        # Both use Box-Muller, should produce same first value
        r1 = gaussian_raw(lcg1)
        r2 = gaussian_random(lcg2)
        assert abs(r1 - r2) < 1e-10

    def test_distribution_mean_near_zero(self):
        """Mean of many samples should be near zero."""
        lcg = LCG(-999)
        samples = [gaussian_raw(lcg) for _ in range(5000)]
        mean = sum(samples) / len(samples)
        assert abs(mean) < 0.05


class TestUniformRandom:
    """Tests for uniform_random factory function."""

    def test_returns_lcg_instance(self):
        """Should return an LCG instance."""
        lcg = uniform_random(-42)
        assert isinstance(lcg, LCG)

    def test_different_seeds_different_instances(self):
        """Different seeds should produce different sequences."""
        lcg1 = uniform_random(-42)
        lcg2 = uniform_random(-43)
        assert lcg1.random() != lcg2.random()


class TestCrossCorr:
    """Tests for cross_corr function."""

    def test_cross_corr_same_signal_equals_autocorr(self):
        """Cross-corr of signal with itself at lag 0 equals autocorr."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        cc = cross_corr(data, 5, data, 5, 0)
        ac = autocorr(data, 5, 0)
        assert abs(cc - ac) < 1e-10

    def test_cross_corr_positive_lag(self):
        """Test cross-correlation with positive lag."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        cc = cross_corr(x, 5, y, 5, 1)
        # x[0]*y[1] + x[1]*y[2] + x[2]*y[3] + x[3]*y[4]
        expected = 1*1 + 2*2 + 3*3 + 4*4
        assert abs(cc - expected) < 1e-10

    def test_cross_corr_negative_lag(self):
        """Test cross-correlation with negative lag."""
        x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y = np.array([2.0, 3.0, 4.0, 5.0, 0.0])
        cc = cross_corr(x, 5, y, 5, -1)
        # For lag=-1: sum(x[i+1] * y[i]) for i in range(min(5, 4))
        # = x[1]*y[0] + x[2]*y[1] + x[3]*y[2] + x[4]*y[3]
        # = 2*2 + 3*3 + 4*4 + 5*5 = 4 + 9 + 16 + 25 = 54
        expected = 54
        assert abs(cc - expected) < 1e-10

    def test_cross_corr_orthogonal_signals(self):
        """Orthogonal signals should have near-zero cross-corr at some lags."""
        x = np.array([1.0, 0.0, -1.0, 0.0])
        y = np.array([0.0, 1.0, 0.0, -1.0])
        cc = cross_corr(x, 4, y, 4, 0)
        assert abs(cc) < 1e-10


class TestMakeSymmetric:
    """Tests for make_symmetric function."""

    def test_symmetric_from_first_row(self):
        """Matrix should become symmetric based on first row."""
        data = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        result = make_symmetric(data.copy())
        # Row 1 should be [2, 1, 2], Row 2 should be [3, 2, 1]
        np.testing.assert_array_almost_equal(result[1], [2.0, 1.0, 2.0])
        np.testing.assert_array_almost_equal(result[2], [3.0, 2.0, 1.0])

    def test_symmetric_matrix_unchanged(self):
        """Already symmetric matrix should remain the same."""
        data = np.array([[1.0, 2.0], [2.0, 1.0]])
        result = make_symmetric(data.copy())
        np.testing.assert_array_almost_equal(result, data)

    def test_1x1_matrix(self):
        """1x1 matrix should be unchanged."""
        data = np.array([[5.0]])
        result = make_symmetric(data.copy())
        np.testing.assert_array_almost_equal(result, data)


class TestSolveLinearSystem:
    """Tests for solve_linear_system function."""

    def test_simple_2x2(self):
        """Solve a simple 2x2 system."""
        A = np.array([[2.0, 1.0], [1.0, 3.0]])
        b = np.array([5.0, 7.0])
        x = solve_linear_system(A, b)
        np.testing.assert_array_almost_equal(A @ x, b)

    def test_3x3_system(self):
        """Solve a 3x3 system."""
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
        b = np.array([6.0, 15.0, 25.0])
        x = solve_linear_system(A, b)
        np.testing.assert_array_almost_equal(A @ x, b)


class TestMatrixAliases:
    """Tests for matrix function aliases."""

    def test_matrix_inv_same_as_mat_inverse(self):
        """matrix_inv should return same result as mat_inverse."""
        A = np.array([[4.0, 1.0], [2.0, 3.0]])
        inv1 = matrix_inv(A)
        inv2 = mat_inverse(A)
        np.testing.assert_array_almost_equal(inv1, inv2)

    def test_matrix_mult_same_as_mat_mult(self):
        """matrix_mult should return same result as mat_mult."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        mult1 = matrix_mult(A, B)
        mult2 = mat_mult(A, B)
        np.testing.assert_array_almost_equal(mult1, mult2)

    def test_mat_mult_correct(self):
        """Matrix multiplication should be correct."""
        A = np.array([[1.0, 2.0], [3.0, 4.0]])
        B = np.array([[5.0, 6.0], [7.0, 8.0]])
        result = mat_mult(A, B)
        expected = np.array([[19.0, 22.0], [43.0, 50.0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transpose_correct(self):
        """Transpose should swap rows and columns."""
        A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        result = transpose(A)
        expected = np.array([[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]])
        np.testing.assert_array_almost_equal(result, expected)


class TestGammp:
    """Tests for gammp (regularized incomplete gamma function)."""

    def test_gammp_returns_float(self):
        """Should return a float."""
        result = gammp(1.0, 1.0)
        assert isinstance(result, float)

    def test_gammp_in_range(self):
        """Result should be in [0, 1]."""
        for a in [0.5, 1.0, 2.0, 5.0]:
            for x in [0.1, 1.0, 5.0, 10.0]:
                result = gammp(a, x)
                assert 0 <= result <= 1

    def test_gammp_at_zero(self):
        """gammp(a, 0) should be 0."""
        result = gammp(1.0, 0.0)
        assert abs(result) < 1e-10

    def test_gammp_large_x_approaches_one(self):
        """gammp(a, x) should approach 1 as x → ∞."""
        result = gammp(1.0, 100.0)
        assert result > 0.99

    def test_gammp_a1_is_1_minus_exp(self):
        """gammp(1, x) = 1 - exp(-x)."""
        x = 2.0
        result = gammp(1.0, x)
        expected = 1 - math.exp(-x)
        assert abs(result - expected) < 1e-10


class TestErf:
    """Tests for erf function."""

    def test_erf_zero(self):
        """erf(0) should be 0."""
        assert abs(erf(0.0)) < 1e-15

    def test_erf_positive(self):
        """erf(positive) should be positive."""
        assert erf(1.0) > 0

    def test_erf_negative(self):
        """erf(negative) should be negative."""
        assert erf(-1.0) < 0

    def test_erf_odd_function(self):
        """erf(-x) = -erf(x)."""
        x = 1.5
        assert abs(erf(-x) + erf(x)) < 1e-15

    def test_erf_approaches_one(self):
        """erf(x) should approach 1 as x → ∞."""
        assert erf(10.0) > 0.9999

    def test_erf_matches_math_erf(self):
        """Should match math.erf."""
        for x in [0.0, 0.5, 1.0, 2.0, -1.0, -3.0]:
            assert abs(erf(x) - math.erf(x)) < 1e-15


class TestAutocorrNegativeLag:
    """Tests for autocorr with negative lag."""

    def test_negative_lag(self):
        """Test autocorr with negative lag."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = autocorr(data, 5, -1)
        # For lag=-1: sum(data[i-(-1)] * data[i]) = sum(data[i+1] * data[i])
        # = data[1]*data[0] + data[2]*data[1] + data[3]*data[2] + data[4]*data[3]
        # = 2*1 + 3*2 + 4*3 + 5*4 = 2 + 6 + 12 + 20 = 40
        expected = 40
        assert abs(result - expected) < 1e-10

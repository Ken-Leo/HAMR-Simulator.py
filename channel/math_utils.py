"""Math utilities for the HAMR simulator.

Provides random number generators (based on Numerical Recipes LCG),
mathematical constants, and helper functions used throughout the
channel, equalizer, and detector modules.

Based on MagneticDisk.c RNG implementations.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PI: float = math.pi

# LCG parameters (Numerical Recipes - glibc)
_IA: int = 16807
_IM: int = 2147483647
_AM: float = 1.0 / _IM
_IQ: int = 127773
_IR: int = 2836
_NTAB: int = 32
_NDIV: int = 1 + (_IM - 1) // _NTAB
_EPS: float = 1.2e-7
_RNMX: float = 1.0 - _EPS

# ---------------------------------------------------------------------------
# Deterministic LCG (drop-in replacement for the C LCG)
# ---------------------------------------------------------------------------


class LCG:
    """Linear Congruential Generator matching the C implementation.

    Implements the Park-Miller LCG: next = (IA * current) % IM
    with the Shuffling extension from Numerical Recipes.
    """

    __slots__ = ("_idum", "_iy", "_iv")

    def __init__(self, idum: int = -1) -> None:
        self._idum = idum
        self._iy = 0
        self._iv: np.ndarray = np.zeros(_NTAB, dtype=np.int64)
        self._init()

    def _init(self) -> None:
        if self._idum <= 0 or self._iy == 0:
            if -self._idum < 1:
                self._idum = 1
            else:
                self._idum = -self._idum
            for j in range(_NTAB + 7, -1, -1):
                k = self._idum // _IQ
                self._idum = _IA * (self._idum - k * _IQ) - _IR * k
                if self._idum < 0:
                    self._idum += _IM
                if j < _NTAB:
                    self._iv[j] = self._idum
            self._iy = self._iv[0]

    def random(self) -> float:
        """Return a uniform random number in (0, 1)."""
        k = self._idum // _IQ
        self._idum = _IA * (self._idum - k * _IQ) - _IR * k
        if self._idum < 0:
            self._idum += _IM
        j = self._iy // _NDIV
        self._iy = self._iv[j]
        self._iv[j] = self._idum
        temp = _AM * self._iy
        return float(_RNMX) if temp > _RNMX else temp

    def reset(self, idum: int) -> None:
        """Reset the generator with a new seed."""
        self._idum = idum
        self._iy = 0
        self._iv = np.zeros(_NTAB, dtype=np.int64)
        self._init()


def uniform_random(idum: int = -1) -> LCG:
    """Return a fresh LCG instance seeded with *idum*."""
    return LCG(idum)


# ---------------------------------------------------------------------------
# Gaussian (Box-Muller)
# ---------------------------------------------------------------------------


def gaussian_random(lcg: LCG) -> float:
    """Return a Gaussian random number (mean=0, std=1).

    Uses the Box-Muller transform.  One call consumes 2 uniform samples.
    This is the non‑cached version.
    """
    while True:
        v1 = 2.0 * lcg.random() - 1.0
        v2 = 2.0 * lcg.random() - 1.0
        s = v1 * v1 + v2 * v2
        if s < 1.0:
            break
    factor = math.sqrt(-2.0 * math.log(s) / s)
    return v1 * factor


def gaussian_raw(lcg: LCG) -> float:
    """Return a single Gaussian random number (mean=0, std=1).

    Uses Box-Muller – same as ``gaussian_random``.  This function exists
    for some internal callers that use the name ``gaussian_raw``.
    """
    return gaussian_random(lcg)


class CachedGaussian:
    """Box-Muller Gaussian generator with caching, matching C's gasdev.

    Internal state:
      iset == 0 → generate 2 uniform samples, compute v1*fac/v2*fac,
                  store v1*fac, return v2*fac, set iset=1.
      iset == 1 → return stored v1*fac, set iset=0.

    This halves the average number of uniform calls per Gaussian compared
    to ``gaussian_random``.
    """

    def __init__(self, lcg: LCG) -> None:
        """Wrap an LCG instance."""
        self._lcg = lcg
        self._is = 0
        self._gs = 0.0

    def __call__(self) -> float:
        """Return the next Gaussian sample."""
        if self._is == 0:
            while True:
                v1 = 2.0 * self._lcg.random() - 1.0
                v2 = 2.0 * self._lcg.random() - 1.0
                s = v1 * v1 + v2 * v2
                if s < 1.0:
                    break
            fac = math.sqrt(-2.0 * math.log(s) / s)
            self._gs = v1 * fac
            self._is = 1
            return v2 * fac
        self._is = 0
        return self._gs


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------


def autocorr(data: np.ndarray, seq_length: int, lag: int) -> float:
    """Compute autocorrelation of *data* at a given *lag*.

    Mirrors the C ``Corr()`` function.

    Args:
        data: Input array (length >= seq_length).
        seq_length: Number of samples to use.
        lag: Correlation lag (can be negative).

    Returns:
        Autocorrelation value.
    """
    result = 0.0
    n = int(abs(lag))
    for i in range(seq_length - n):
        result += data[i] * data[i + n]
    return result


# ---------------------------------------------------------------------------
# Matrix helpers (for GPR equalizer)
# ---------------------------------------------------------------------------


def make_symmetric(data: np.ndarray) -> np.ndarray:
    """Make a matrix symmetric from its first row/column (as C code does)."""
    n = data.shape[0]
    for i in range(1, n):
        for j in range(n):
            data[i, j] = data[0, abs(j - i)]
    return data


def solve_linear_system(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve A x = b using numpy (equivalent to C Gaussian elimination)."""
    return np.linalg.solve(A, b)


def mat_inverse(A: np.ndarray) -> np.ndarray:
    """Return the inverse of matrix A."""
    return np.linalg.inv(A)


def matrix_inv(A: np.ndarray) -> np.ndarray:
    """Alias for mat_inverse, matching C code naming."""
    return mat_inverse(A)


def mat_mult(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiplication."""
    return A @ B


def matrix_mult(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Alias for mat_mult, matching C code naming."""
    return mat_mult(A, B)


def transpose(A: np.ndarray) -> np.ndarray:
    """Return the transpose of matrix A."""
    return A.T


# ---------------------------------------------------------------------------
# Cross-correlation (vectorized version of C Corr)
# ---------------------------------------------------------------------------


def gammp(a: float, x: float) -> float:
    """Regularized incomplete gamma function P(a, x).

    Numerical Recipes ``gammp`` equivalent.  Wraps scipy.special.gammainc.

    Args:
        a: Shape parameter (positive).
        x: Evaluation point.

    Returns:
        P(a, x) = gamma(a, x) / gamma(a) in [0, 1].
    """
    from scipy.special import gammainc
    return gammainc(a, x)


def cross_corr(x: np.ndarray, x_len: int,
               y: np.ndarray, y_len: int,
               lag: int) -> float:
    """Compute cross-correlation between x and y at *lag*.

    Equivalent to the C ``Corr(x, x_len, y, y_len, lag, &result)``.
    """
    result = 0.0
    n = int(abs(lag))
    if lag >= 0:
        limit = min(x_len - n, y_len)
        for i in range(limit):
            result += x[i] * y[i + n]
    else:
        limit = min(x_len, y_len - n)
        for i in range(limit):
            result += x[i + n] * y[i]
    return result


def erf(x: float) -> float:
    """Error function, matching C ``erf()`` in MagneticDisk.c.

    The C implementation uses ``gammp(0.5, x*x)`` with sign handling.
    We delegate to ``math.erf`` for accuracy and performance.
    """
    return math.erf(x)

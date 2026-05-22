"""
HAMR Channel Implementation.

Translated from MagneticDisk.c (lines 2084-3310).
Contains the full HAMR magnetic recording channel simulation including
transition writing, microtrack readback, NLTS effects, and media noise.

NOTE: This is a near-complete standalone translation that depends on
several math utilities not yet ported.  The main simulator uses the
simplified hamr_channel() in channel/channel.py instead.

This module is imported lazily and will gracefully fail if dependencies
are missing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple

if TYPE_CHECKING:
    pass

# Lazy import of math utilities -- may fail if not all functions are ported
try:
    from channel.math_utils import (
        LCG,
        gammp,
        gaussian_raw,
        gaussian_random,
        matrix_inv,
        matrix_mult,
        uniform_random,
        erf,
    )
except ImportError:
    LCG = None  # type: ignore[misc,assignment]
    gammp = None  # type: ignore[misc,assignment]
    gaussian_raw = None  # type: ignore[misc,assignment]
    gaussian_random = None  # type: ignore[misc,assignment]
    matrix_inv = None  # type: ignore[misc,assignment]
    matrix_mult = None  # type: ignore[misc,assignment]
    uniform_random = None  # type: ignore[misc,assignment]
    erf = None  # type: ignore[misc,assignment]

PI = 3.1415926
LARGE_SPOT = 1
MEDIUM_SPOT = 0
BISECTION_ACC = 0.01
ITERATION_ACC = 0.1
JMAX = 2000
YSTART = -500.0
YEND = 500.0
YSTEP = 0.1


# =============================================================================
# Parameter Dataclasses (C struct translations)
# =============================================================================

@dataclass
class Mag_Param:
    """Magnetic recording parameter struct.

    Attributes
    ----------
    sigma_t : float
        Sigma of T-Profile in nm.
    T_Peak : float
        Peak temperature in centigrade.
    Orig_T_Peak : float
        Original peak temperature (for modulation effects).
    c : float
        Downtrack location of peak temp w.r.t. gap centreline (nm).
    d : float
        Crosstrack location of peak temp w.r.t. track centre (nm).
    Orig_d : float
        Original d (for write head cross-track movement).
    z0 : float
        Crosstrack location for T calculation.
    Hc : list
        Hc = a*T + b, [a, b] in A/m.
    Mr : list
        Mr = a*T + b, [a, b] in A/m.
    S : list
        S = a*T + b, [a, b].
    Hg : float
        Deep gap field in A/m.
    """

    sigma_t: float = 90.0
    T_Peak: float = 350.0
    Orig_T_Peak: float = 350.0
    c: float = 0.0
    d: float = 0.0
    Orig_d: float = 0.0
    z0: float = 0.0
    Hc: Optional[List[float]] = None
    Mr: Optional[List[float]] = None
    S: Optional[List[float]] = None
    Hg: float = 1.6e6  # 16e5 A/m

    def __post_init__(self) -> None:
        if self.Hc is None:
            self.Hc = [0.0, 0.0]
        if self.Mr is None:
            self.Mr = [0.0, 0.0]
        if self.S is None:
            self.S = [0.0, 0.0]
        if self.Orig_T_Peak == 0.0:
            self.Orig_T_Peak = self.T_Peak
        if self.Orig_d == 0.0:
            self.Orig_d = self.d


@dataclass
class Physical_Param:
    """Physical head/medium parameter struct.

    Attributes
    ----------
    g : float
        Write/head gap in nm.
    d : float
        Medium spacing (nm).
    t : float
        Medium thickness (nm).
    y : float
        Distance from bottom of pole to center of medium (nm).
    wt : float
        Track width (nm).
    """

    g: float = 100.0
    d: float = 19.0
    t: float = 2.0
    y: float = 28.0
    wt: float = 160.0


@dataclass
class Reader_Param:
    """GMR reader parameter struct.

    Attributes
    ----------
    C : float
        GMR reader constant.
    gr : float
        Spacing between shield and GMR element (nm).
    tr : float
        Width of GMR element (nm).
    wr : float
        Width of reader crosstrack (nm).
    sigma_r : float
        Sigma of reader sensitivity Gaussian (nm).
    """

    C: float = 1.0
    gr: float = 5.0
    tr: float = 1.0
    wr: float = 1000.0
    sigma_r: float = 1000.0


# =============================================================================
# Core Channel Functions
# =============================================================================

def _gaussian_truncated_truncated(
    sigma: float, rng: LCG, truncation: float = 0.5
) -> float:
    """Truncated Gaussian: reject samples outside [-truncation, truncation]."""
    while True:
        val = gaussian_raw() * sigma
        if abs(val) < truncation:
            return val


def _gaussian_truncated_positive(
    sigma: float, rng: LCG
) -> float:
    """Single-sided Gaussian: reject negative samples."""
    while True:
        val = gaussian_raw() * sigma
        if val >= 0:
            return val


def _temperature_at(
    T_Peak: float,
    z0: float,
    d: float,
    c: float,
    sigma_t: float,
    x: float,
) -> float:
    """Calculate temperature at a given downtrack position x.

    T(x) = T_Peak * exp(-(z0-d)^2 / (2*sigma_t^2)) * exp(-(x-c)^2 / (2*sigma_t^2)) + 300
    """
    exp_z = math.exp(-((z0 - d) ** 2) / (2.0 * sigma_t ** 2))
    exp_x = math.exp(-((x - c) ** 2) / (2.0 * sigma_t ** 2))
    return T_Peak * exp_z * exp_x + 300.0


def bisection(
    x1: float,
    x2: float,
    sigma_t: float,
    T_Peak: float,
    c_val: float,
    d_val: float,
    z0: float,
    Hc: List[float],
    Hg: float,
    g: float,
    y_val: float,
) -> float:
    """Bisection root-finding for Ha(Xo) = Hh(Xo).

    Solves Hc(T(x)) = Hh(x) for the transition centre Xo.
    """

    def _eval(x_pt: float) -> float:
        T_at = T_Peak * math.exp(-((z0 - d_val) ** 2) / (2.0 * sigma_t ** 2))
        T_at *= math.exp(-((x_pt - c_val) ** 2) / (2.0 * sigma_t ** 2))
        T_at += 300.0
        Hc_x = Hc[0] * T_at + Hc[1]
        Hh_x = (Hg / PI) * (
            math.atan((x_pt + g / 2.0) / y_val)
            - math.atan((x_pt - g / 2.0) / y_val)
        )
        return Hc_x - Hh_x

    f = _eval(x1)
    fmid = _eval(x2)

    if f * fmid >= 0.0:
        if x2 == 0.0:
            x2 = c_val
        else:
            x2 = 0.0
        fmid = _eval(x2)
        if f * fmid >= 0.0:
            raise ValueError("No solution for Hc(T(x)) = Hh(x)")

    if f < 0.0:
        rtb, dx = x1, x2 - x1
    else:
        rtb, dx = x2, x1 - x2

    for _ in range(JMAX):
        dx *= 0.5
        xmid = rtb + dx
        fmid = _eval(xmid)
        if fmid <= 0.0:
            rtb = xmid
        if abs(dx) < BISECTION_ACC or fmid == 0.0:
            return rtb

    raise RuntimeError("Bisection did not converge")


def Hd(
    mp: Mag_Param,
    pp: Physical_Param,
    _rp: Reader_Param,
    x0: float,
    a: float,
    x: float,
) -> float:
    """Calculate the demagnetizing field p(x0) at a given location.

    Numerical integration over y of -dM(y) * Hstep(x - y).
    """
    sigma_t = mp.sigma_t
    T_Peak = mp.T_Peak
    c = mp.c
    d = mp.d
    z0 = mp.z0
    Mr0 = mp.Mr[0]
    Mr1 = mp.Mr[1]
    g = pp.g
    t = pp.t
    y_val = pp.y

    P = 0.0
    y = YSTART
    while y <= YEND:
        T_y = T_Peak * math.exp(-((z0 - d) ** 2) / (2.0 * sigma_t ** 2))
        T_y *= math.exp(-((y - c) ** 2) / (2.0 * sigma_t ** 2))
        T_y += 300.0

        dMy = (2.0 / PI) * -abs(Mr0 * T_y + Mr1) * (a / (a * a + (y - x0) ** 2))
        dMy += (
            (2.0 / PI)
            * math.atan((y - x0) / a)
            * abs(Mr0)
            * (T_y - 300.0)
            * -(y - c)
            / (sigma_t ** 2)
        )
        dMy *= YSTEP

        if x - y != 0.0:
            Hstep = (1.0 / PI) * math.atan(t / (2.0 * (x - y)))
        else:
            Hstep = 0.5

        P -= dMy * Hstep
        y += YSTEP

    return P


def dHd(
    mp: Mag_Param,
    pp: Physical_Param,
    rp: Reader_Param,
    x0: float,
    a: float,
    x: float,
) -> float:
    """Gradient of Hd w.r.t. x, via central difference."""
    Hd1 = Hd(mp, pp, rp, x0, a, x - YSTEP)
    Hd2 = Hd(mp, pp, rp, x0, a, x + YSTEP)
    return (Hd2 - Hd1) / (2.0 * YSTEP)


def bisection_tcentre(
    x1: float,
    x2: float,
    xacc: float,
    mp: Mag_Param,
    pp: Physical_Param,
    rp: Reader_Param,
    a: float,
    TCentre: List[float],
    TParam: List[float],
    num_transition: int,
    num_nlts: int,
    trans_bit_loc: List[int],
    oversampled_bit_length: float,
) -> float:
    """Find transition centre with NLTS effects included."""
    sigma_t = mp.sigma_t
    T_Peak = mp.T_Peak
    c_val = mp.c
    d_val = mp.d
    z0 = mp.z0
    Hg = -mp.Hg
    Hc = mp.Hc[:]
    g = pp.g
    y_val = pp.y

    def _eval(x_pt: float) -> float:
        T_at = T_Peak * math.exp(-((z0 - d_val) ** 2) / (2.0 * sigma_t ** 2))
        T_at *= math.exp(-((x_pt - c_val) ** 2) / (2.0 * sigma_t ** 2))
        T_at += 300.0
        Hc_x = -abs(Hc[0] * T_at + Hc[1])
        Hh_x = (Hg / PI) * (
            math.atan((x_pt + g / 2.0) / y_val)
            - math.atan((x_pt - g / 2.0) / y_val)
        )
        Hd_x = Hd(mp, pp, rp, x_pt, a, x_pt)

        Hd_prev = 0.0
        if num_transition > 1:
            start_i = num_transition - 1
            end_i = (
                max(num_transition - num_nlts, 1)
                if num_transition - 1 >= num_nlts
                else 1
            )
            temp = 0
            for i in range(start_i, end_i - 1, -1):
                B = -((trans_bit_loc[num_transition] - trans_bit_loc[i])
                      * oversampled_bit_length + abs(TCentre[i - 1]))
                Hd_prev += ((-1) ** temp) * -Hd(mp, pp, rp, B, TParam[i - 1], x_pt)
                temp += 1

        return -(Hc_x - Hh_x - Hd_x - Hd_prev)

    f = _eval(x1)
    fmid = _eval(x2)

    if f * fmid >= 0.0:
        if x2 == 0.0:
            x2 = c_val
        else:
            x2 = 0.0
        fmid = _eval(x2)
        if f * fmid >= 0.0:
            raise ValueError("No solution for Hc(T(x)) = Hh(x) with NLTS")

    if f < 0.0:
        rtb, dx = x1, x2 - x1
    else:
        rtb, dx = x2, x1 - x2

    for _ in range(JMAX):
        dx *= 0.5
        xmid = rtb + dx
        fmid = _eval(xmid)
        if fmid <= 0.0:
            rtb = xmid
        if abs(dx) < xacc or fmid == 0.0:
            return rtb

    raise RuntimeError("BisectionTCentre did not converge")


def bisection_tparam(
    x1: float,
    x2: float,
    xacc: float,
    mp: Mag_Param,
    pp: Physical_Param,
    rp: Reader_Param,
    x0: float,
) -> float:
    """Find transition parameter a via thermal-Williams slope equation."""
    sigma_t = mp.sigma_t
    T_Peak = mp.T_Peak
    c_val = mp.c
    z0 = mp.z0
    Hg = -mp.Hg
    Hc = mp.Hc[:]
    Mr = mp.Mr[:]
    S = mp.S[:]
    g = pp.g
    y_val = pp.y

    T0 = T_Peak * math.exp(-((z0 - mp.d) ** 2) / (2.0 * sigma_t ** 2))
    T0 *= math.exp(-((x0 - c_val) ** 2) / (2.0 * sigma_t ** 2))
    T0 += 300.0

    Mr0 = -abs(Mr[0] * T0 + Mr[1])
    Hc0 = -abs(Hc[0] * T0 + Hc[1])

    dM_dH = abs(Mr0 / (Hc0 * (1.0 - S[0] * T0 - S[1])))
    dHh_dx = (
        Hg / (PI * y_val)
        * (1.0 / (1.0 + ((x0 + g / 2.0) / y_val) ** 2)
           - 1.0 / (1.0 + ((x0 - g / 2.0) / y_val) ** 2))
    )
    dHc_dT = abs(Hc[0])
    dT_dx = (T0 - 300.0) * -(x0 - c_val) / (sigma_t ** 2)

    def _eval(x_pt: float) -> float:
        if x_pt != 0.0:
            dM_dx = 2.0 * Mr0 / (PI * x_pt)
        else:
            dM_dx = -1e50
        dHd_dx = dHd(mp, pp, rp, x0, x_pt, x0)
        return dM_dx - dM_dH * (dHh_dx + dHd_dx - dHc_dT * dT_dx)

    f = _eval(x1)
    fmid = _eval(x2)

    if f * fmid >= 0.0:
        if x2 == 0.0:
            x2 = c_val
        else:
            x2 = 0.0
        fmid = _eval(x2)
        if f * fmid >= 0.0:
            raise ValueError("No solution for thermal-Williams slope equation")

    if f < 0.0:
        rtb, dx = x1, x2 - x1
    else:
        rtb, dx = x2, x1 - x2

    for _ in range(JMAX):
        dx *= 0.5
        xmid = rtb + dx
        fmid = _eval(xmid)
        if fmid <= 0.0:
            rtb = xmid
        if abs(dx) < xacc or fmid == 0.0:
            return rtb

    raise RuntimeError("BisectionTParam did not converge")


def transition_large_spot(
    mp: Mag_Param,
    pp: Physical_Param,
    rp: Reader_Param,
) -> Tuple[float, float]:
    """Compute transition centre and parameter via large-spot approximation.

    Returns
    -------
    (x0, a) : tuple
        Transition centre and transition parameter.
    """
    sigma_t = mp.sigma_t
    T_Peak = mp.T_Peak
    c_val = mp.c
    d_val = mp.d
    z0 = mp.z0
    Hg_val = mp.Hg
    Hc = mp.Hc[:]
    Mr = mp.Mr[:]
    S = mp.S[:]
    g = pp.g
    t = pp.t
    y_val = pp.y

    if c_val > 0.0:
        x2 = 0.0
    else:
        x2 = c_val
    x1 = x2 - 5.0 * sigma_t

    x0 = bisection(x1, x2, sigma_t, T_Peak, c_val, d_val, z0, Hc, Hg_val, g, y_val)

    T0 = T_Peak * math.exp(-((z0 - d_val) ** 2) / (2.0 * sigma_t ** 2))
    T0 *= math.exp(-((x0 - c_val) ** 2) / (2.0 * sigma_t ** 2))
    T0 += 300.0

    dT = (
        T_Peak
        * math.exp(-((z0 - d_val) ** 2) / (2.0 * sigma_t ** 2))
        * math.exp(-((x0 - c_val) ** 2) / (2.0 * sigma_t ** 2))
        * (-(x0 - c_val))
        / (sigma_t ** 2)
    )
    Hc_x0 = -abs(Hc[0] * T0 + Hc[1])
    dHc = abs(Hc[0])
    Mr_x0 = -abs(Mr[0] * T0 + Mr[1])
    Hg_neg = -Hg_val

    Q = (
        math.sin(PI * Hc_x0 / Hg_neg) ** 2
        * 2.0 * x0 * Hg_neg
        / (PI * g * Hc_x0)
    )
    beta = 1.0 - dHc * dT * y_val / (abs(Hc_x0) * Q)
    temp1 = -y_val * (1.0 - (S[0] * T0 + S[1])) / (PI * Q * beta)
    temp2 = Mr_x0 * t * y_val / (PI * Q * abs(Hc_x0) * beta)
    a = temp1 - t / 4.0 + math.sqrt(
        (temp1 - t / 4.0) ** 2 + temp2 + t * temp1
    )

    return x0, a


def validate_answer(
    mp: Mag_Param,
    pp: Physical_Param,
    rp: Reader_Param,
    x0: float,
    a: float,
) -> float:
    """Validate transition parameters against thermal-Williams slope equation.

    Returns the residual (should be near zero).
    """
    sigma_t = mp.sigma_t
    T_Peak = mp.T_Peak
    c_val = mp.c
    z0 = mp.z0
    Hg = -mp.Hg
    Hc = mp.Hc[:]
    Mr = mp.Mr[:]
    S = mp.S[:]
    g = pp.g
    y_val = pp.y

    T0 = T_Peak * math.exp(-((z0 - mp.d) ** 2) / (2.0 * sigma_t ** 2))
    T0 *= math.exp(-((x0 - c_val) ** 2) / (2.0 * sigma_t ** 2))
    T0 += 300.0

    Mr0 = -abs(Mr[0] * T0 + Mr[1])
    Hc0 = -abs(Hc[0] * T0 + Hc[1])

    dM_dH = abs(Mr0 / (Hc0 * (1.0 - S[0] * T0 - S[1])))
    dHh_dx = (
        Hg / (PI * y_val)
        * (1.0 / (1.0 + ((x0 + g / 2.0) / y_val) ** 2)
           - 1.0 / (1.0 + ((x0 - g / 2.0) / y_val) ** 2))
    )
    dHc_dT = abs(Hc[0])
    dT_dx = (T0 - 300.0) * -(x0 - c_val) / (sigma_t ** 2)
    dM_dx = 2.0 * Mr0 / (PI * a)
    dHd_dx = dHd(mp, pp, rp, x0, a, x0)

    return dM_dx - dM_dH * (dHh_dx + dHd_dx - dHc_dT * dT_dx)


def transition(
    mp: Mag_Param,
    pp: Physical_Param,
    rp: Reader_Param,
    num_transition: int,
    num_nlts: int,
    trans_bit_loc: List[int],
    oversampled_bit_length: float,
) -> Tuple[float, float]:
    """Compute transition centre and parameter iteratively.

    Uses large-spot initial guess, then refines with NLTS-aware bisection.

    Returns
    -------
    (TCentre, TParam) : tuple
    """
    max_iter = 20
    num_iter = 0

    x0_prev = 0.0
    x0_curr = 0.0
    a_prev = 0.0
    a_curr = 0.0

    # Initial large-spot estimate
    x0_curr, a_curr = transition_large_spot(mp, pp, rp)
    x0_prev = x0_curr
    a_prev = a_curr

    while (abs(x0_curr - x0_prev) > ITERATION_ACC
           or abs(a_curr - a_prev) > ITERATION_ACC):

        # Bisection for centre
        if mp.c > 0.0:
            bx2 = 0.0
        else:
            bx2 = mp.c
        bx1 = x0_prev - 30.0

        x0_curr = bisection_tcentre(
            bx1, bx2, BISECTION_ACC, mp, pp, rp, a_prev,
            [0.0], [0.0], num_transition, num_nlts,
            trans_bit_loc, oversampled_bit_length,
        )

        # Bisection for parameter
        ax1 = a_prev - 20.0 if a_prev - 20.0 > 2.0 else 2.0
        ax2 = a_prev + 20.0
        a_curr = bisection_tparam(ax1, ax2, BISECTION_ACC, mp, pp, rp, x0_curr)

        diff_a = abs(a_curr - a_prev)
        diff_x = abs(x0_curr - x0_prev)

        x0_prev = x0_curr
        a_prev = a_curr
        num_iter += 1

        if num_iter >= max_iter:
            break

        temp = validate_answer(mp, pp, rp, x0_prev, a_prev)
        if abs(temp) < 10:
            break

    temp = validate_answer(mp, pp, rp, x0_prev, a_prev)
    if abs(temp) > 200:
        pass  # Warning suppressed in non-main code

    return x0_prev, a_prev


def microtrack(
    mp: Mag_Param,
    pp: Physical_Param,
    rp: Reader_Param,
    N: int,
    x: List[float],
    v: List[float],
    length: int,
    num_transition: int,
    TCentre: List[List[float]],
    TParam: List[List[float]],
    k: List[float],
    rho: List[float],
    trans_bit_loc: List[int],
    num_nlts: int,
    oversampled_bit_length: float,
    OSR: int,
    num_sectors: int,
    temperature_variation: int,
    sigma_temp_variation: float,
    peak_temp_trunc_value: float,
    write_hd_cross_track_mov: int,
    max_write_hd_cr_tr_mov: float,
    mean_write_hd_cr_tr_mov: float,
    sigma_jitter: float,
    HMD: int,
    sigma_hmd_variation: float,
    num_ar_coeff: int,
    ar_model_coeff: List[float],
    delta_d: List[float],
    normalization_factor: List[float],
    trans_to_calc_trans_param: int,
    temperature_modulated_flag: int,
    rng_temp: LCG,
    rng_write_hd: LCG,
    rng_hmd: LCG,
    rng_jitter: LCG,
) -> None:
    """Simulate single microtrack readback for all N tracks.

    Modifies v[] in-place, updates TCentre, TParam, k, rho, delta_d,
    and normalization_factor.
    """
    delz = pp.wt / N
    sigma_t = mp.sigma_t
    sigma_r = rp.sigma_r
    T_Peak = mp.T_Peak
    c_val = mp.c
    d_val = mp.d
    Hg_val = mp.Hg
    Hc = mp.Hc[:]
    Mr = mp.Mr[:]
    S = mp.S[:]
    g = pp.g
    d = pp.d
    t = pp.t
    y_val = pp.y
    gr = rp.gr

    # Temperature variation for first transition
    if trans_to_calc_trans_param == 0 and temperature_variation == 1:
        while abs(mp.T_Peak - mp.Orig_T_Peak) > peak_temp_trunc_value:
            mp.T_Peak = (
                gaussian_raw() * (sigma_temp_variation / 100.0 * mp.Orig_T_Peak)
                + mp.Orig_T_Peak
            )
        T_Peak = mp.T_Peak

    # Write head cross-track movement
    if trans_to_calc_trans_param == 0 and write_hd_cross_track_mov == 1:
        mp.d = 10000.0
        while abs(mp.d - mp.Orig_d) > max_write_hd_cr_tr_mov / 100.0 * pp.wt:
            u = rng_write_hd.uniform()
            if u <= 0:
                u = 1e-10
            mp.d = (
                mp.Orig_d
                - (mean_write_hd_cr_tr_mov / 100.0 * pp.wt)
                * math.log(u)
            )

    # HMD variation
    if trans_to_calc_trans_param == 0 and HMD == 1:
        if num_ar_coeff >= 1:
            b = math.sqrt(1 + ar_model_coeff[0] ** 2 - 2 * ar_model_coeff[0])
        else:
            b = 1.0
        delta_d[num_transition - 1] = (
            b * sigma_hmd_variation * gaussian_raw()
        )
        limit = num_ar_coeff if num_ar_coeff < num_transition else num_transition - 1
        for i in range(1, limit):
            delta_d[num_transition - 1] += (
                ar_model_coeff[i - 1] * delta_d[num_transition - 1 - i]
            )
        d = pp.d + delta_d[num_transition - 1]
        sigma_t = mp.sigma_t * math.sqrt(
            1 + (delta_d[num_transition - 1] / pp.d) ** 2
        )

    # Jitter
    DeltaX = 0.0
    if (trans_to_calc_trans_param == 0
            and sigma_jitter != 0.0
            and num_sectors >= 0):
        DeltaX = 0.5
        while abs(DeltaX) >= 0.5:
            DeltaX = (sigma_jitter / 100.0) * gaussian_raw()
        DeltaX = DeltaX * OSR * oversampled_bit_length

    # Loop over microtracks (1-indexed in C, 0-indexed in Python)
    for i in range(1, N + 1):
        mp.z0 = pp.wt / 2.0 - (delz * (i - 1) + delz / 2.0)
        z0 = mp.z0

        # Decide x1, x2 for bisection
        if c_val > 0.0:
            x2_bisect = 0.0
        else:
            x2_bisect = c_val
        x1_bisect = x2_bisect - 5.0 * sigma_t

        x0 = 0.0
        a = 0.0

        if LARGE_SPOT:
            if num_transition == 1:
                if temperature_modulated_flag == -1:
                    orig_T = mp.T_Peak
                    mp.T_Peak = mp.Orig_T_Peak
                    TCentre[i - 1][0] = bisection(
                        x1_bisect, x2_bisect, sigma_t, mp.Orig_T_Peak,
                        c_val, d_val, z0, Hc, Hg_val, g, y_val,
                    )
                    mp.T_Peak = orig_T
                else:
                    TCentre[i - 1][0] = bisection(
                        x1_bisect, x2_bisect, sigma_t, T_Peak,
                        c_val, d_val, z0, Hc, Hg_val, g, y_val,
                    )
                x0 = TCentre[i - 1][0]
            else:
                # Reallocate for new transition (Python lists just append)
                TCentre[i - 1].append(TCentre[i - 1][0])
                TParam[i - 1].append(0.0)

                if temperature_variation == 1 or write_hd_cross_track_mov == 1:
                    TCentre[i - 1][num_transition - 1] = bisection(
                        x1_bisect, x2_bisect, sigma_t, T_Peak,
                        c_val, d_val, z0, Hc, Hg_val, g, y_val,
                    )
                    x0 = TCentre[i - 1][num_transition - 1]
                else:
                    # NLTS: compute delta from previous transitions
                    delta = 0.0
                    temp_flag = 1
                    TCentre[i - 1][num_transition - 1] = TCentre[i - 1][0]
                    start_j = num_transition - 1
                    end_j = (
                        max(num_transition - num_nlts, 1)
                        if num_transition - 1 >= num_nlts
                        else 1
                    )
                    for j in range(start_j, end_j - 1, -1):
                        B = (
                            (trans_bit_loc[num_transition] - trans_bit_loc[j])
                            * oversampled_bit_length
                            + TCentre[i - 1][0]
                            - TCentre[i - 1][j - 1]
                        )
                        delta = ((-1) ** temp_flag) * k[i - 1] / (B ** rho[i - 1])
                        T_prev = T_Peak * math.exp(-((z0 - d_val) ** 2)
                                                   / (2.0 * sigma_t ** 2))
                        T_prev *= math.exp(
                            -((-(B + abs(TCentre[i - 1][0])) - c_val) ** 2)
                            / (2.0 * sigma_t ** 2)
                        ) + 300.0
                        delta *= (
                            (Mr[0] * T_prev + Mr[1])
                            / (Mr[0] * 300.0 + Mr[1])
                        )
                        TCentre[i - 1][num_transition - 1] += delta
                        temp_flag += 1
                    x0 = TCentre[i - 1][num_transition - 1]

            # Temperature modulation adjustment
            if temperature_modulated_flag == -1:
                x0 = bisection(
                    x1_bisect, x2_bisect, sigma_t, T_Peak,
                    c_val, d_val, z0, Hc, Hg_val, g, y_val,
                )

            # Transition parameter evaluation
            T0 = T_Peak * math.exp(-((z0 - d_val) ** 2) / (2.0 * sigma_t ** 2))
            T0 *= math.exp(-((x0 - c_val) ** 2) / (2.0 * sigma_t ** 2))
            T0 += 300.0
            dT = (
                T_Peak
                * math.exp(-((z0 - d_val) ** 2) / (2.0 * sigma_t ** 2))
                * math.exp(-((x0 - c_val) ** 2) / (2.0 * sigma_t ** 2))
                * (-(x0 - c_val))
                / (sigma_t ** 2)
            )
            Hc_x0 = -abs(Hc[0] * T0 + Hc[1])
            dHc = abs(Hc[0])
            Mr_x0 = -abs(Mr[0] * T0 + Mr[1])
            Q = (
                math.sin(PI * Hc_x0 / (-Hg_val)) ** 2
                * 2.0 * x0 * (-Hg_val)
                / (PI * g * Hc_x0)
            )
            beta = 1.0 - dHc * dT * y_val / (abs(Hc_x0) * Q)
            temp1 = -y_val * (1.0 - (S[0] * T0 + S[1])) / (PI * Q * beta)
            temp2 = Mr_x0 * t * y_val / (PI * Q * abs(Hc_x0) * beta)
            a = (temp1 - t / 4.0
                 + math.sqrt((temp1 - t / 4.0) ** 2 + temp2 + t * temp1))

            TParam[i - 1][num_transition - 1] = a

        elif MEDIUM_SPOT:
            TCentre[i - 1].append(TCentre[i - 1][0])
            TParam[i - 1].append(0.0)
            xc, xp = transition(
                mp, pp, rp, num_transition, num_nlts,
                trans_bit_loc, oversampled_bit_length,
            )
            TCentre[i - 1][num_transition - 1] = xc
            TParam[i - 1][num_transition - 1] = xp
            x0 = TCentre[i - 1][num_transition - 1]
            a = TParam[i - 1][num_transition - 1]
        else:
            raise ValueError("Select either LARGE_SPOT or MEDIUM_SPOT")

        # Temperature modulation revert
        if temperature_modulated_flag == -1:
            x0 = TCentre[i - 1][num_transition - 1]

        # Reader sensitivity
        rs = math.exp(
            -((-0.5 * (N + 1) * delz + i * delz) ** 2) / (2.0 * sigma_r ** 2)
        )

        # Add contribution to readback signal
        for j in range(length):
            if j == 0:
                x0_adj = x0 + DeltaX
            else:
                x0_adj = x0
            diffx = abs(x[j] - x0_adj)
            v[j] += (
                normalization_factor[0]
                * (1.0 / N)
                * rs
                * (Mr[0] * 300.0 + Mr[1])
                * (
                    math.atan((diffx + gr / 2.0) / (a + d))
                    - math.atan((diffx - gr / 2.0) / (a + d))
                )
            )

    # Calculate PW50 and normalization factor (for first transition only)
    if trans_to_calc_trans_param == 1:
        max_val = v[0]
        index_max = 0
        for i in range(1, length):
            if v[i] > max_val:
                max_val = v[i]
                index_max = i

        min_dist = 1e5
        for i in range(index_max, -1, -1):
            d_val = abs(v[i] - max_val / 2.0)
            if d_val < min_dist:
                min_dist = d_val
                left_index = i

        min_dist = 1e5
        for i in range(index_max, length):
            d_val = abs(v[i] - max_val / 2.0)
            if d_val < min_dist:
                min_dist = d_val
                right_index = i

        normalization_factor[0] = (2.0 * math.sqrt(2.0 / PI)) / max_val

        # Would set PW and PeakAmpPositionIndex if returned


def hamr_channel(
    mp: Mag_Param,
    pp: Physical_Param,
    rp: Reader_Param,
    N: int,
    oversampled_input_bits: List[int],
    length_padded: int,
    oversampled_bit_length: float,
    OSR: int,
    num_sectors: int,
    k: List[float],
    rho: List[float],
    x: List[float],
    length: int,
    NLTS_compensation: int = 0,
    num_nlts_influencing: int = 5,
    temperature_variation: int = 0,
    sigma_temp_variation: float = 0.0,
    peak_temp_trunc_value: float = 0.1,
    write_hd_cross_track_mov: int = 0,
    max_write_hd_cr_tr_mov: float = 0.0,
    mean_write_hd_cr_tr_mov: float = 0.0,
    sigma_jitter: float = 0.0,
    HMD: int = 0,
    sigma_hmd_variation: float = 0.0,
    num_ar_coeff: int = 0,
    ar_model_coeff: List[float] = None,
    temperature_modulation: int = 0,
    modulated_peak_temp: float = 0.0,
    trans_to_calc_trans_param: int = 0,
    normalization_factor: float = 1.0,
    seed: int = -500,
) -> tuple:
    """Full HAMR channel simulation.

    Translates the C Hamr() function (MagneticDisk.c lines 3067-3310).

    Parameters
    ----------
    mp : Mag_Param
        Magnetic parameters.
    pp : Physical_Param
        Physical head/medium parameters.
    rp : Reader_Param
        Reader parameters.
    N : int
        Number of microtracks.
    oversampled_input_bits : list of int
        Bipolar input data bits (+1/-1 or 1/0 format with transitions).
    length_padded : int
        Length of oversampled_input_bits.
    oversampled_bit_length : float
        Downtrack length of one oversampled bit.
    OSR : int
        Oversampling rate.
    num_sectors : int
        Sector index (-1 for trans param calculation).
    k : list of float
        NLTS amplitude parameters per track.
    rho : list of float
        NLTS decay exponent per track.
    x : list of float
        Downtrack positions.
    length : int
        Length of x.
    NLTS_compensation : int
        NLTS compensation flag.
    num_nlts_influencing : int
        Number of previous transitions influencing NLTS.
    temperature_variation : int
        Temperature variation flag.
    sigma_temp_variation : float
        Temperature variation sigma (%).
    peak_temp_trunc_value : float
        Truncation value for peak temp variation.
    write_hd_cross_track_mov : int
        Write head cross-track movement flag.
    max_write_hd_cr_tr_mov : float
        Max cross-track movement (% of track width).
    mean_write_hd_cr_tr_mov : float
        Mean cross-track movement (% of track width).
    sigma_jitter : float
        Jitter noise sigma (% of bit period).
    HMD : int
        HMD variation flag.
    sigma_hmd_variation : float
        HMD variation sigma.
    num_ar_coeff : int
        Number of AR model coefficients.
    ar_model_coeff : list of float
        AR model coefficients.
    temperature_modulation : int
        Temperature modulation flag.
    modulated_peak_temp : float
        Modulated peak temperature.
    trans_to_calc_trans_param : int
        Flag: 0 = normal simulation, 1 = calculate trans params.
    seed : int
        RNG seed for reproducibility.

    Returns
    -------
    readback_signal : list of float
        Combined readback signal.
    pw50 : float
        PW50 width.
    peak_amp_index : int
        Index of peak amplitude.
    normalization_factor : float
        Normalization factor.
    TCentre : list of list of float
        Transition centres per track.
    TParam : list of list of float
        Transition parameters per track.
    """
    if ar_model_coeff is None:
        ar_model_coeff = []

    pi = PI

    # Initialize readback signal (accumulated output).
    # Must be large enough to hold the full oversampled domain signal.
    readback_signal = [0.0] * length_padded

    # Temporary microtrack buffer (size of X array)
    v = [0.0] * length

    # NLTS params (per track, index 0..N-1)
    num_transition = 0

    # TCentre and TParam: indexed 1..N in C => 0..N-1 in Python
    TCentre: List[List[float]] = [[0.0]] * N
    TParam: List[List[float]] = [[0.0]] * N
    for i in range(N):
        TCentre[i] = [0.0]
        TParam[i] = [0.0]

    # Transition bit location
    trans_bit_loc: List[int] = [0]  # 1-indexed in C; trans_bit_loc[0] = 0

    # HMD DeltaD
    if HMD == 1:
        delta_d: List[float] = []
    else:
        delta_d = []

    # Normalization factor (passed by reference via list)
    normalization_factor = [normalization_factor]

    # RNG states (separate streams for different noise sources)
    rng_temp = LCG(seed - 100) if seed < 0 else LCG(-(seed + 100))
    rng_write_hd = LCG(seed - 900) if seed < 0 else LCG(-(seed + 900))
    rng_hmd = LCG(seed - 280) if seed < 0 else LCG(-(seed + 280))
    rng_jitter = LCG(seed - 160) if seed < 0 else LCG(-(seed + 160))

    count_consecutive_trans = 0
    temperature_modulated_flag = 1
    peak_amp_position_index = 0
    pw50 = 0.0

    # Loop over all input bits (1-indexed in C)
    for i in range(1, length_padded):
        # Transition detection
        if oversampled_input_bits[i] - oversampled_input_bits[i - 1] != 0:
            num_transition += 1
            trans_bit_loc.append(i)

            if HMD == 1:
                delta_d.append(0.0)

            # Temperature modulation: detect tribits
            if (temperature_modulation == 1
                    and num_sectors >= 0
                    and trans_to_calc_trans_param == 0):
                if (i >= 2 * OSR
                        and i < length_padded - 3 * OSR):
                    prev_trans = (
                        oversampled_input_bits[i - OSR]
                        - oversampled_input_bits[i - OSR - 1]
                    )
                    next_trans = (
                        oversampled_input_bits[i + OSR]
                        - oversampled_input_bits[i + OSR - 1]
                    )
                    if prev_trans != 0 and next_trans != 0:
                        count_consecutive_trans += 1
                        if (num_sectors >= 0
                                and (count_consecutive_trans % 3 == 2)):
                            pass  # Modulated bit (not implemented in v1)
                    elif (prev_trans != 0
                          and i >= 2 * OSR
                          and oversampled_input_bits[i - 2 * OSR]
                          - oversampled_input_bits[i - 2 * OSR - 1] != 0):
                        count_consecutive_trans += 1
                        if count_consecutive_trans % 3 == 0:
                            count_consecutive_trans = 0
                    elif (next_trans != 0
                          and oversampled_input_bits[i + 2 * OSR]
                          - oversampled_input_bits[i + 2 * OSR - 1] != 0):
                        count_consecutive_trans += 1
                        if count_consecutive_trans % 3 == 1:
                            mp.T_Peak = modulated_peak_temp
                            temperature_modulated_flag = -1
                    else:
                        count_consecutive_trans = 0

            # Call Microtrack
            microtrack(
                mp, pp, rp, N, x, v, length, num_transition,
                TCentre, TParam, k, rho, trans_bit_loc,
                num_nlts_influencing, oversampled_bit_length, OSR,
                num_sectors, temperature_variation,
                sigma_temp_variation, peak_temp_trunc_value,
                write_hd_cross_track_mov, max_write_hd_cr_tr_mov,
                mean_write_hd_cr_tr_mov, sigma_jitter, HMD,
                sigma_hmd_variation, num_ar_coeff, ar_model_coeff,
                delta_d, normalization_factor,
                trans_to_calc_trans_param, temperature_modulated_flag,
                rng_temp, rng_write_hd, rng_hmd, rng_jitter,
            )

            temperature_modulated_flag = 1
            if temperature_modulation == 1:
                mp.T_Peak = mp.Orig_T_Peak

            peak_amp_position_index = _get_peak_index(v, length)

            # Accumulate into readback signal
            if oversampled_input_bits[i] - oversampled_input_bits[i - 1] > 0:
                _accumulate_signal(
                    v, readback_signal,
                    i=i, peak_amp_position_index=peak_amp_position_index,
                    length=length, length_padded=length_padded,
                )
            elif oversampled_input_bits[i] - oversampled_input_bits[i - 1] < 0:
                _accumulate_signal_neg(
                    v, readback_signal,
                    i=i, peak_amp_position_index=peak_amp_position_index,
                    length=length, length_padded=length_padded,
                )

            # Reset v
            v = [0.0] * length

    # Calculate PW50 (pulse width at 50% of peak amplitude)
    if readback_signal and max(abs(x) for x in readback_signal) > 0:
        max_val = max(readback_signal)
        min_val = min(readback_signal)
        peak_val = max_val if abs(max_val) > abs(min_val) else min_val
        half_val = peak_val / 2.0

        # Find peak index
        peak_idx = readback_signal.index(peak_val)

        # Find left half-maximum point
        left_idx = peak_idx
        for i in range(peak_idx, -1, -1):
            if readback_signal[i] <= half_val if peak_val > 0 else readback_signal[i] >= half_val:
                left_idx = i
                break

        # Find right half-maximum point
        right_idx = peak_idx
        for i in range(peak_idx, len(readback_signal)):
            if readback_signal[i] <= half_val if peak_val > 0 else readback_signal[i] >= half_val:
                right_idx = i
                break

        # PW50 in nm (assuming x is sampled at 1nm intervals for initial calc)
        if trans_to_calc_trans_param == 1:
            # x is sampled at 1nm
            pw50 = float(right_idx - left_idx)
        else:
            # x is sampled at over_sampled_bit_length intervals
            pw50 = float(right_idx - left_idx) * oversampled_bit_length

    return readback_signal, pw50, peak_amp_position_index, normalization_factor[0]


def _get_peak_index(v: List[float], length: int) -> int:
    """Find index of maximum value in v."""
    max_val = v[0]
    idx = 0
    for i in range(1, length):
        if v[i] > max_val:
            max_val = v[i]
            idx = i
    return idx


def _accumulate_signal(
    v: List[float],
    readback_signal: List[float],
    i: int,
    peak_amp_position_index: int,
    length: int,
    length_padded: int,
) -> None:
    """Accumulate positive readback signal."""
    if peak_amp_position_index > i:
        start = peak_amp_position_index - i
    else:
        start = 0

    rem = length - peak_amp_position_index
    avail = length_padded - i
    if rem > avail:
        end = peak_amp_position_index + avail
    else:
        end = length

    temp = i - (peak_amp_position_index - start)
    for j in range(start, end):
        if 0 <= temp < len(readback_signal):
            readback_signal[temp] += v[j]
        temp += 1


def _accumulate_signal_neg(
    v: List[float],
    readback_signal: List[float],
    i: int,
    peak_amp_position_index: int,
    length: int,
    length_padded: int,
) -> None:
    """Accumulate negative readback signal."""
    if peak_amp_position_index > i:
        start = peak_amp_position_index - i
    else:
        start = 0

    rem = length - peak_amp_position_index
    avail = length_padded - i
    if rem > avail:
        end = peak_amp_position_index + avail
    else:
        end = length

    temp = i - (peak_amp_position_index - start)
    for j in range(start, end):
        if 0 <= temp < len(readback_signal):
            readback_signal[temp] -= v[j]
        temp += 1

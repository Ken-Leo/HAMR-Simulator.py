"""Soft-Output Viterbi Algorithm (SOVA).

Translates ClassicalSOVA() from MagneticDisk.c (lines 1053-1400).

SOVA extends the classical Viterbi detector by producing soft
outputs (reliability information / log-likelihood ratios) in
addition to hard decisions.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

_LARGE_METRIC: float = 1e50


def classical_sova(
    delay: int,
    equalized_channel_output: np.ndarray,
    sector_length: int,
    pri_imp_res: np.ndarray,
    noise_sigma: float,
    constraint_callback: Callable[[int, int], bool] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft-Output Viterbi Algorithm (SOVA).

    Parameters
    ----------
    delay : int
        Detection delay (decision latency in bit periods).
    equalized_channel_output : np.ndarray
        Received samples after equalisation (modified in-place).
    sector_length : int
        Number of samples to decode.
    pri_imp_res : np.ndarray
        PR impulse response (e.g. ``[1, 1, -1, -1]`` for EPR4).
    noise_sigma : float
        Estimated noise standard deviation.
    constraint_callback : Callable[[int, int], bool] | None
        Optional callback to enforce code constraints.
        Signature: `callback(k, state) -> bool` (True if valid).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(detected_hard_output, detected_soft_output, pri_imp_res)``.
        *detected_soft_output* contains the probability of correct
        decision in range ``(0, 1]``.
    """
    pri_imp_res_length = len(pri_imp_res)
    num_states = 1 << (pri_imp_res_length - 1)  # 2^(K-1)

    # DC bias correction for non-DC-free PR targets
    dc_bias = float(np.sum(pri_imp_res))

    noise_variance = noise_sigma * noise_sigma

    # Path metric: ping-pong buffers  [2 x num_states]
    path_metric = np.zeros((2, num_states), dtype=np.float64)

    # Path history: ping-pong buffers  [2 x num_states x delay]
    path = np.zeros((2, num_states, delay), dtype=np.int64)

    # Probability of wrong detection: [2 x num_states x delay]
    prob_wrong_det = np.zeros((2, num_states, delay), dtype=np.float64)

    # Valid-state tracking during startup transient
    vs_prev: list[int] = [0]
    vs_curr: list[int] = []

    min_path_metric_index = 0

    # Combined buffer for hard decisions and soft outputs.
    # hard values stored at indices 0..sector_length-1
    # soft values stored at indices sector_length..2*sector_length-1
    combined = np.zeros(2 * sector_length, dtype=np.float64)
    combined[:sector_length] = equalized_channel_output[:sector_length]

    for k in range(sector_length):
        vs_curr = []

        for i in range(num_states):
            # ---- Branch metrics ----
            sample0 = 0.0
            for j in range(pri_imp_res_length - 1):
                sample0 += pri_imp_res[j] * ((i >> j) & 1)
            sample0 = sample0 * 2.0 - dc_bias

            cur = k & 1
            prev = 1 - cur

            prev_state0 = i >> 1
            metric0 = path_metric[prev, prev_state0] + (
                combined[k] - sample0
            ) ** 2

            sample1 = 0.0
            for j in range(pri_imp_res_length - 1):
                sample1 += pri_imp_res[j] * ((i >> j) & 1)
            sample1 += pri_imp_res[pri_imp_res_length - 1]
            sample1 = sample1 * 2.0 - dc_bias

            prev_state1 = (i >> 1) | (1 << (pri_imp_res_length - 2))
            metric1 = path_metric[prev, prev_state1] + (
                combined[k] - sample1
            ) ** 2

            if k < (pri_imp_res_length - 1):
                # Startup transient
                for vs_val in vs_prev:
                    if prev_state0 == vs_val:
                        path[cur, i, : delay - 1] = path[prev, prev_state0, 1:]
                        path[cur, i, delay - 1] = i & 1
                        path_metric[cur, i] = metric0
                        vs_curr.append(i)
                        break
                    if prev_state1 == vs_val:
                        path[cur, i, : delay - 1] = path[prev, prev_state1, 1:]
                        path[cur, i, delay - 1] = i & 1
                        path_metric[cur, i] = metric1
                        vs_curr.append(i)
                        break
                else:
                    path_metric[cur, i] = _LARGE_METRIC
                continue

            # ---- Survivor selection ----
            if metric0 <= metric1:
                path[cur, i, : delay - 1] = path[prev, prev_state0, 1:]
                path[cur, i, delay - 1] = i & 1
                path_metric[cur, i] = metric0
                surviving_state = prev_state0
            else:
                path[cur, i, : delay - 1] = path[prev, prev_state1, 1:]
                path[cur, i, delay - 1] = i & 1
                path_metric[cur, i] = metric1
                surviving_state = prev_state1

            # ---- Update probability of wrong detection ----
            delta = abs(metric0 - metric1) / (2.0 * noise_variance)
            # Clamp delta to prevent math.exp() overflow
            delta_safe = min(delta, 500.0)
            prob_wrong_surv = 1.0 / (1.0 + math.exp(delta_safe))

            # Bits where the two incoming paths differ (XOR)
            differing_bits = path[prev, prev_state0] ^ path[prev, prev_state1]

            # Current-time probability of wrong decision
            prob_wrong_det[cur, i, delay - 1] = prob_wrong_surv

            # Propagate to earlier bits.
            # DifferingBits[j] is 1 where the two incoming paths differ at
            # position j in the delay buffer.  Position 0 is the oldest
            # bit (already decided); we skip it and start from position 1.
            # This matches the C code:
            #   for(j=0;j<Delay-1;j++) { if(DifferingBits[j+1]==1) ... }
            for j in range(delay - 1):
                bit_idx = delay - 2 - j
                if differing_bits[j + 1]:
                    # Bits differ between the two paths
                    prob_wrong_det[cur, i, bit_idx] = (
                        prob_wrong_det[prev, surviving_state, delay - 1 - j]
                        * (1.0 - prob_wrong_surv)
                        + (
                            1.0
                            - prob_wrong_det[prev, surviving_state, delay - 1 - j]
                        )
                        * prob_wrong_surv
                    )
                else:
                    # Bits do not differ
                    prob_wrong_det[cur, i, bit_idx] = prob_wrong_det[
                        prev, surviving_state, delay - 1 - j
                    ]

            # ---- Apply Code Constraints ----
            if constraint_callback is not None:
                if not constraint_callback(k, i):
                    path_metric[cur, i] = _LARGE_METRIC

        # ---- Make hard decision ----
        if k >= delay - 1:
            min_path_metric_index = int(np.argmin(path_metric[cur]))
            detected_bit = path[cur, min_path_metric_index, 0]
            pos = k - (delay - 1)
            combined[pos] = float(detected_bit)

            # Soft output: probability of correct decision
            if detected_bit == 1:
                soft_out = prob_wrong_det[cur, min_path_metric_index, 0]
            else:
                soft_out = 1.0 - prob_wrong_det[cur, min_path_metric_index, 0]
            combined[pos + sector_length] = soft_out

        # ---- Ping-pong swap: copy computed buffer (cur) -> buffer 0 ----
        path_metric[0] = path_metric[cur].copy()
        path[0] = path[cur].copy()
        prob_wrong_det[0] = prob_wrong_det[cur].copy()

        # ---- Swap valid-state lists ----
        if k < (pri_imp_res_length - 1):
            vs_prev = vs_curr
            vs_curr = []

    # ---- Traceback: last ``delay - 1`` bits ----
    detected_hard_output = np.zeros(sector_length, dtype=np.float64)
    detected_soft_output = np.zeros(sector_length, dtype=np.float64)

    # Fill in the decisions made during the main loop
    for k in range(delay - 1, sector_length):
        pos = k - (delay - 1)
        detected_hard_output[pos] = combined[pos]
        detected_soft_output[pos] = combined[pos + sector_length]

    # Final traceback
    for i in range(2, delay + 1):
        pos = sector_length - 1 - (delay - i)
        detected_bit = path[0, min_path_metric_index, i - 1]
        detected_hard_output[pos] = float(detected_bit)

        if detected_bit == 1:
            detected_soft_output[pos] = prob_wrong_det[
                0, min_path_metric_index, i - 1
            ]
        else:
            detected_soft_output[pos] = 1.0 - prob_wrong_det[
                0, min_path_metric_index, i - 1
            ]

    detected_soft_output = np.clip(detected_soft_output, 1e-10, 1.0)
    return detected_hard_output, detected_soft_output, pri_imp_res

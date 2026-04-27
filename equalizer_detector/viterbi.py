"""Classical Viterbi detector for PRML channels.

Translates Viterbi() from MagneticDisk.c (lines 1382-1598).

Uses the non-classical initialisation pattern:
- Path metric starts at 0 for all states (state 0 assumed at time 0).
- Valid states are tracked during the startup transient
  (k < PRImpResLength - 1) so that only transitions from reachable
  states are accepted.
- After the transient all 2^(K-1) states are valid.
"""

import numpy as np

_LARGE_METRIC: float = 1e50


def classical_viterbi(
    delay: int,
    equalized_channel_output: np.ndarray,
    sector_length: int,
    pri_imp_res: np.ndarray,
) -> tuple:
    """Classical Viterbi detector for a PRML channel.

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

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(detected_hard_output, pri_imp_res)``.
        *detected_hard_output* has length ``sector_length`` and contains
        bits in ``{0, 1}``.
    """
    pri_imp_res_length = len(pri_imp_res)
    num_states = 1 << (pri_imp_res_length - 1)  # 2^(K-1)

    # Path metric: ping-pong buffers  [2 x num_states]
    path_metric = np.zeros((2, num_states), dtype=np.float64)

    # Path history: ping-pong buffers  [2 x num_states x delay]
    # path[cur, state, bit_index] -- each entry is 0 or 1
    path = np.zeros((2, num_states, delay), dtype=np.int64)

    # Valid-state tracking during startup transient
    # VS stores the actual state integers that are reachable.
    vs_prev: list[int] = [0]  # start from state 0 at time -1
    vs_curr: list[int] = []

    min_path_metric_index = 0

    for k in range(sector_length):
        vs_curr = []

        for i in range(num_states):
            # ---- Branch metrics for the two incoming transitions ----
            # sample for transition from previous state with bit 0 appended
            sample0 = 0.0
            for j in range(pri_imp_res_length - 1):
                sample0 += pri_imp_res[j] * ((i >> j) & 1)
            sample0 = sample0 * 2.0

            prev_state0 = i >> 1
            metric0 = path_metric[0, prev_state0] + (
                equalized_channel_output[k] - sample0
            ) ** 2

            # sample for transition from previous state with bit 1 appended
            sample1 = 0.0
            for j in range(pri_imp_res_length - 1):
                sample1 += pri_imp_res[j] * ((i >> j) & 1)
            sample1 += pri_imp_res[pri_imp_res_length - 1]
            sample1 = sample1 * 2.0

            prev_state1 = (i >> 1) | (1 << (pri_imp_res_length - 2))
            metric1 = path_metric[0, prev_state1] + (
                equalized_channel_output[k] - sample1
            ) ** 2

            cur = k & 1
            prev = 1 - cur

            if k < (pri_imp_res_length - 1):
                # Startup transient: only accept transitions from valid
                # previous states.
                found = False
                for vs_val in vs_prev:
                    if prev_state0 == vs_val:
                        path[cur, i, : delay - 1] = path[prev, prev_state0, 1:]
                        path[cur, i, delay - 1] = i & 1
                        path_metric[cur, i] = metric0
                        vs_curr.append(i)
                        found = True
                        break
                    if prev_state1 == vs_val:
                        path[cur, i, : delay - 1] = path[prev, prev_state1, 1:]
                        path[cur, i, delay - 1] = i & 1
                        path_metric[cur, i] = metric1
                        vs_curr.append(i)
                        found = True
                        break
                if not found:
                    path_metric[cur, i] = _LARGE_METRIC
                    continue
            else:
                # All states and all incoming paths are valid.
                if metric0 <= metric1:
                    path[cur, i, : delay - 1] = path[prev, prev_state0, 1:]
                    path[cur, i, delay - 1] = i & 1
                    path_metric[cur, i] = metric0
                else:
                    path[cur, i, : delay - 1] = path[prev, prev_state1, 1:]
                    path[cur, i, delay - 1] = i & 1
                    path_metric[cur, i] = metric1

        # ---- Make decision on the bit transmitted ``delay`` bit
        #      periods ago ----
        if k >= delay - 1:
            # Find the survivor state (lowest path metric).
            min_path_metric_index = int(np.argmin(path_metric[cur]))
            detected_bit = path[cur, min_path_metric_index, 0]
            equalized_channel_output[k - (delay - 1)] = float(detected_bit)

        # ---- Ping-pong swap: copy cur -> prev, zero cur ----
        path_metric[prev] = path_metric[cur].copy()
        path[prev] = path[cur].copy()
        path_metric[cur] = 0.0
        path[cur] = 0

        # ---- Swap valid-state lists ----
        if k < (pri_imp_res_length - 1):
            vs_prev = vs_curr
            vs_curr = []

    # ---- Traceback: last ``delay - 1`` bits ----
    # After the loop, cur has been swapped to prev, so the final
    # path data lives in index ``prev``.
    for i in range(2, delay + 1):
        pos = sector_length - 1 - (delay - i)
        detected_bit = path[prev, min_path_metric_index, i - 1]
        equalized_channel_output[pos] = float(detected_bit)

    return equalized_channel_output, pri_imp_res

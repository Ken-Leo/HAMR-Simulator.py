"""Classical Viterbi detector for PRML channels.

Direct translation of ClassicalViterbi() from CustomDetectors.c.

Key pattern (matches C exactly):
- Read accumulated metrics and path history from index 0.
- Write new metrics and path history to index 1.
- Swap: copy index 1 -> index 0 at end of each iteration.
- Initialisation: all metrics = 1e50 except state 0 = 0.
"""

from __future__ import annotations

from typing import Callable
import numpy as np

_LARGE_METRIC: float = 1e50


def classical_viterbi(
    delay: int,
    equalized_channel_output: np.ndarray,
    sector_length: int,
    pri_imp_res: np.ndarray,
    constraint_callback: Callable[[int, int], bool] | None = None,
) -> tuple:
    """Classical Viterbi detector for a PRML channel.

    Parameters
    ----------
    delay : int
        Detection delay (decision latency in bit periods).
    equalized_channel_output : np.ndarray
        Received samples after equalisation (NOT modified).
    sector_length : int
        Number of samples to decode.
    pri_imp_res : np.ndarray
        PR impulse response (e.g. ``[1, 1, -1, -1]`` for EPR4).
    constraint_callback : Callable[[int, int], bool] | None
        Optional callback to enforce code constraints.
        Signature: `callback(k, state) -> bool` (True if valid).

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(detected_hard_output, pri_imp_res)``.
        *detected_hard_output* has length ``sector_length`` and contains
        bits in ``{0, 1}``.
    """
    pri_imp_res_length = len(pri_imp_res)
    num_states = 1 << (pri_imp_res_length - 1)  # 2^(K-1)

    # DC bias correction: for non-DC-free PR targets, subtract sum(pri)
    # from the expected sample so that the branch metric for a correct
    # hypothesis is zero.  (DC-free targets have sum(pri) = 0 → no effect.)
    dc_bias = float(np.sum(pri_imp_res))

    # Path metric: ping-pong buffers  [2 x num_states]
    path_metric = np.full((2, num_states), _LARGE_METRIC, dtype=np.float64)
    path_metric[0, 0] = 0.0
    path_metric[1, 0] = 0.0

    # Path history: ping-pong buffers  [2 x num_states x delay]
    # Always read from path[0], always write to path[1], swap at end.
    path = np.zeros((2, num_states, delay), dtype=np.int64)

    min_path_metric_index = 0

    # Output array — matches C: separate DetectedOutput array
    detected_output = np.zeros(sector_length, dtype=np.float64)

    for k in range(sector_length):
        # Compute new metrics into buffer 1, read from buffer 0
        for i in range(num_states):
            # ---- Branch metrics for the two incoming transitions ----

            # sample for transition with bit 0 appended
            sample0 = 0.0
            for j in range(pri_imp_res_length - 1):
                sample0 += pri_imp_res[j] * ((i >> j) & 1)
            sample0 = sample0 * 2.0 - dc_bias
            prev_state0 = i >> 1
            metric0 = path_metric[0, prev_state0] + (
                equalized_channel_output[k] - sample0
            ) ** 2

            # sample for transition with bit 1 appended
            sample1 = 0.0
            for j in range(pri_imp_res_length - 1):
                sample1 += pri_imp_res[j] * ((i >> j) & 1)
            sample1 += pri_imp_res[pri_imp_res_length - 1]
            sample1 = sample1 * 2.0 - dc_bias

            prev_state1 = (i >> 1) | (1 << (pri_imp_res_length - 2))
            metric1 = path_metric[0, prev_state1] + (
                equalized_channel_output[k] - sample1
            ) ** 2

            # ---- Select survivor path ----
            if metric0 <= metric1:
                path[1, i, : delay - 1] = path[0, prev_state0, 1:]
                path[1, i, delay - 1] = i & 1
                path_metric[1, i] = metric0
            else:
                path[1, i, : delay - 1] = path[0, prev_state1, 1:]
                path[1, i, delay - 1] = i & 1
                path_metric[1, i] = metric1

            # ---- Apply Code Constraints ----
            if constraint_callback is not None:
                if not constraint_callback(k, i):
                    path_metric[1, i] = _LARGE_METRIC

        # ---- Make decision on the bit transmitted ``delay`` bit
        #      periods ago ----
        if k >= delay - 1:
            min_path_metric_index = int(np.argmin(path_metric[1]))
            detected_bit = int(path[1, min_path_metric_index, 0])
            detected_output[k - (delay - 1)] = float(detected_bit)

        # ---- Ping-pong swap: copy buffer 1 -> buffer 0 ----
        path_metric[0] = path_metric[1].copy()
        path[0] = path[1].copy()

    # ---- Traceback: last ``delay - 1`` bits ----
    # After the loop, the final swap put path data into buffer 0.
    for i in range(2, delay + 1):
        pos = sector_length - 1 - (delay - i)
        detected_bit = int(path[0, min_path_metric_index, i - 1])
        detected_output[pos] = float(detected_bit)

    return detected_output, pri_imp_res


def classical_viterbi_sliding_window(
    delay: int,
    equalized_channel_output: np.ndarray,
    sector_length: int,
    pri_imp_res: np.ndarray,
    window_size: int | None = None,
    boundary_guard: int = 40,
) -> np.ndarray:
    """Sliding-window Viterbi detector for long sequences.

    Processes the input in overlapping windows. Each window runs a full
    Viterbi trellis, and only the middle portion (well past the initial
    transient and before the abrupt truncation) is kept.

    Parameters
    ----------
    delay : int
        Detection delay.
    equalized_channel_output : np.ndarray
        Received samples (not modified).
    sector_length : int
        Total number of samples to decode.
    pri_imp_res : np.ndarray
        PR impulse response.
    window_size : int | None
        Window size. Defaults to ``max(delay * 5, 100)``.
    boundary_guard : int
        Number of samples at each edge of the window that are discarded
        as unreliable (must be >= delay).

    Returns
    -------
    np.ndarray
        Detected bits of length ``sector_length``.
    """
    if window_size is None:
        window_size = max(delay * 5, 100)

    result = np.zeros(sector_length, dtype=np.float64)

    # Ensure window is large enough to have a valid middle region
    min_window = 2 * boundary_guard + 1
    if window_size < min_window:
        window_size = min_window

    step = window_size - 2 * boundary_guard  # valid samples per window
    if step <= 0:
        step = 1

    pos = 0
    while pos < sector_length:
        # Window start/end in the original signal
        win_start = max(0, pos - boundary_guard)
        win_end = min(sector_length, pos + window_size - boundary_guard)

        # Extract window data
        win_data = equalized_channel_output[win_start:win_end].copy()
        win_len = len(win_data)

        # Run Viterbi
        det, _ = classical_viterbi(delay, win_data, win_len, pri_imp_res)

        # Valid region indices within the window
        # The first 'boundary_guard' samples are boundary effects
        # The last 'boundary_guard' samples are truncation effects
        v_start = boundary_guard
        v_end = win_len - boundary_guard

        # Map window indices back to original signal indices
        sig_start = win_start + v_start
        sig_end = win_start + v_end

        # Clamp to output array
        out_start = max(0, sig_start)
        out_end = min(sector_length, sig_end)

        # Map to detection buffer indices
        det_start = out_start - win_start
        det_end = out_end - win_start

        if det_end > det_start:
            result[out_start:out_end] = det[det_start:det_end]

        pos += step

    return result

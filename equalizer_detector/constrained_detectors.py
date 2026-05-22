"""Code-constrained detectors for MTR and TMTR codes.

Translates constrained Viterbi and SOVA detectors from
CustomDetectors.c. These are more complex and build on the
classical detectors by enforcing code constraints on the trellis.
"""
from __future__ import annotations

import numpy as np
from .sova import classical_sova
from .viterbi import classical_viterbi


def viterbi_6by7mtr_code(
    delay: int,
    equalized_channel_output: np.ndarray,
    pre_padding_length: int,
    sector_length: int,
    pri_imp_res: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """6/7 MTR code-constrained Viterbi detector."""
    def mtr_constraint(k: int, state: int) -> bool:
        if k < pre_padding_length + 1:
            return True
        if state in {5, 10}:
            return False
        return True

    return classical_viterbi(
        delay, equalized_channel_output, sector_length, pri_imp_res,
        constraint_callback=mtr_constraint,
    )


def viterbi_8by9tmtr_code(
    delay: int,
    equalized_channel_output: np.ndarray,
    pre_padding_length: int,
    sector_length: int,
    pri_imp_res: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """8/9 TMTR code-constrained Viterbi detector."""
    def tmtr_constraint(k: int, state: int) -> bool:
        if k < pre_padding_length + 1:
            return True

        rel_k = k - pre_padding_length
        num_codeword = (rel_k - 1) // 9 + 1

        if (rel_k - (num_codeword - 1) * 9) % 2 == 1:
            if state in {5, 10}:
                return False
        return True

    return classical_viterbi(
        delay, equalized_channel_output, sector_length, pri_imp_res,
        constraint_callback=tmtr_constraint,
    )


def sova_6by7mtr_code(
    delay: int,
    equalized_channel_output: np.ndarray,
    pre_padding_length: int,
    sector_length: int,
    pri_imp_res: np.ndarray,
    noise_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """6/7 MTR code-constrained SOVA detector."""
    def mtr_constraint(k: int, state: int) -> bool:
        if k < pre_padding_length + 1:
            return True
        if state in {5, 10}:
            return False
        return True

    return classical_sova(
        delay, equalized_channel_output, sector_length, pri_imp_res,
        noise_sigma, constraint_callback=mtr_constraint,
    )


def sova_8by9mtr_code(
    delay: int,
    equalized_channel_output: np.ndarray,
    pre_padding_length: int,
    sector_length: int,
    pri_imp_res: np.ndarray,
    noise_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """8/9 TMTR code-constrained SOVA detector."""
    def tmtr_constraint(k: int, state: int) -> bool:
        if k < pre_padding_length + 1:
            return True

        rel_k = k - pre_padding_length
        num_codeword = (rel_k - 1) // 9 + 1

        if (rel_k - (num_codeword - 1) * 9) % 2 == 1:
            if state in {5, 10}:
                return False
        return True

    return classical_sova(
        delay, equalized_channel_output, sector_length, pri_imp_res,
        noise_sigma, constraint_callback=tmtr_constraint,
    )


def classical_viterbi_for_temp_mod_with_perm(
    delay: int,
    equalized_channel_output: np.ndarray,
    sector_length: int,
    pri_imp_res: np.ndarray,
    mse: np.ndarray,
    num_bits_rep_perm_period: int,
    sub_sector_length: int,
    num_interleaved_bits: int,
    pre_padding_length: int,
) -> np.ndarray:
    """Viterbi with temperature modulation and permutation tracking.

    Translates ``ClassicalViterbiForTempModWithPerm`` from CustomDetectors.c
    (lines 2713-2951).

    This detector processes data in sub-sectors, making a decision at the
    end of each sub-sector and storing the minimum path metric (MSE).
    Interleaved bits between sub-sectors are skipped.

    Parameters
    ----------
    delay : int
        Detection delay (overridden to ``sub_sector_length`` internally).
    equalized_channel_output : np.ndarray
        Received samples after equalisation.
    sector_length : int
        Total number of samples to decode.
    pri_imp_res : np.ndarray
        PR impulse response.
    mse : np.ndarray
        Output array of length ``num_bits_rep_perm_period`` to store
        the minimum path metric for each sub-sector.
    num_bits_rep_perm_period : int
        Number of sub-sectors (repetition period).
    sub_sector_length : int
        Length of each sub-sector.
    num_interleaved_bits : int
        Number of interleaved (skipped) bits between sub-sectors.
    pre_padding_length : int
        Pre-padding length for sub-sector boundary calculation.

    Returns
    -------
    np.ndarray
        Detected bits of length ``sector_length``.
    """
    pri_imp_res_length = len(pri_imp_res)
    num_states = 1 << (pri_imp_res_length - 1)
    delay = sub_sector_length  # C code overrides delay

    # Path metric: ping-pong buffers
    path_metric = np.full((2, num_states), 1e50, dtype=np.float64)
    path_metric[0, 0] = 0.0
    path_metric[1, 0] = 0.0

    # Path history
    path = np.zeros((2, num_states, delay), dtype=np.int64)

    detected_output = np.zeros(sector_length, dtype=np.float64)
    current_sub_sector = 1

    k = 0
    while k < sector_length:
        # Compute new metrics into buffer 1, read from buffer 0
        for i in range(num_states):
            # Branch metric for bit 0 transition
            sample0 = 0.0
            for j in range(pri_imp_res_length - 1):
                sample0 += pri_imp_res[j] * ((i >> j) & 1)
            sample0 *= 2.0
            metric0 = path_metric[0, i >> 1] + (
                equalized_channel_output[k] - sample0
            ) ** 2

            # Branch metric for bit 1 transition
            sample1 = 0.0
            for j in range(pri_imp_res_length - 1):
                sample1 += pri_imp_res[j] * ((i >> j) & 1)
            sample1 += pri_imp_res[pri_imp_res_length - 1]
            sample1 *= 2.0
            prev_state1 = (i >> 1) | (1 << (pri_imp_res_length - 2))
            metric1 = path_metric[0, prev_state1] + (
                equalized_channel_output[k] - sample1
            ) ** 2

            # Select survivor
            if metric0 <= metric1:
                path[1, i, : delay - 1] = path[0, i >> 1, 1:]
                path[1, i, delay - 1] = i & 1
                path_metric[1, i] = metric0
            else:
                path[1, i, : delay - 1] = path[0, prev_state1, 1:]
                path[1, i, delay - 1] = i & 1
                path_metric[1, i] = metric1

        # Check if we've reached the end of the current sub-sector
        sub_sector_end = (
            sub_sector_length
            + (sub_sector_length + num_interleaved_bits) * (current_sub_sector - 1)
            - 1
            + pre_padding_length
        )
        if (
            k - pre_padding_length
            == sub_sector_length + (sub_sector_length + num_interleaved_bits) * (current_sub_sector - 1) - 1
            and current_sub_sector != num_bits_rep_perm_period
        ):
            # Make decision on current sub-sector
            min_idx = int(np.argmin(path_metric[1]))
            for i in range(delay):
                detected_output[k - (delay - 1 - i)] = float(
                    path[1, min_idx, i]
                )
            mse[current_sub_sector - 1] = path_metric[1, min_idx]

            # Skip interleaved bits
            k += num_interleaved_bits

            # Re-initialize path metrics and paths
            path_metric[:, :] = 1e50
            path_metric[0, 0] = 0.0
            path_metric[1, 0] = 0.0
            path[:, :, :] = 0

            current_sub_sector += 1

        # Check for last sub-sector boundary
        if (
            k - pre_padding_length
            == (sub_sector_length + num_interleaved_bits) * (current_sub_sector - 1) - 1
            and current_sub_sector == num_bits_rep_perm_period
        ):
            # Last sub-sector: adjust delay to remaining length
            delay = sector_length - k - 1
            path = np.zeros((2, num_states, delay), dtype=np.int64)

        # Detect the last sub-sector
        if k == sector_length - 1:
            min_idx = int(np.argmin(path_metric[1]))
            for i in range(delay):
                detected_output[k - (delay - 1 - i)] = float(
                    path[1, min_idx, i]
                )
            mse[current_sub_sector - 1] = path_metric[1, min_idx]

        # Exchange old and new path/path metric values
        path_metric[0] = path_metric[1].copy()
        path_metric[1] = 0.0
        path[0] = path[1].copy()
        path[1] = 0

        k += 1

    return detected_output

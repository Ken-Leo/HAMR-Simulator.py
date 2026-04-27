"""Permutation-based tribit minimization.

Provides functions to count error events (tribits) in a bit sequence and
to find the optimal cyclic permutation shift that minimizes those events.

Based on EncodingFunctions.c CountPossibleErrorEvents and Permute.
"""

import numpy as np


def count_possible_error_events(input_word: np.ndarray, length: int) -> int:
    """Count the number of tribits in the bit sequence.

    A tribit is a run of 3 or more consecutive transitions (alternating
    0->1->0 or 1->0->1). Each additional transition beyond the first
    two contributes one error event.

    For example, a run of 4 consecutive transitions contains (4-1) = 2
    tribits.

    Args:
        input_word: Array of 0/1 values representing the bit sequence.
        length: Number of bits to examine from ``input_word``.

    Returns:
        The total number of tribit error events found in the sequence.

    Example:
        >>> import numpy as np
        >>> bits = np.array([0, 1, 0, 1, 0, 0, 0])
        >>> count_possible_error_events(bits, len(bits))
        2
    """
    num_error_events = 0
    i = 1
    while i < length:
        n_cons_trans = 0
        while i < length and abs(input_word[i] - input_word[i - 1]) == 1:
            n_cons_trans += 1
            i += 1
        if n_cons_trans >= 3:
            num_error_events += n_cons_trans - 1
        i += 1
    return num_error_events


def permute(
    input_word: np.ndarray,
    sector_length: int,
    p: np.ndarray,
    period: int,
) -> tuple:
    """Find the optimal cyclic permutation shift with the fewest tribits.

    Given a permutation pattern with cyclic period ``period``, this
    function tries all ``period`` cyclic shifts of the input codeword
    and selects the one that minimizes the number of tribit error events.

    The permutation is applied cyclically: position ``P[(j+i) % PLen]``
    in the permuted word receives the value from position ``P[j]`` in
    the original word, for each shift ``i`` from 0 to ``period-1``.

    Args:
        input_word: The NRZ codeword bits to permute.
        sector_length: Total length of the word.
        p: Permutation index array of length ``period``. Contains indices
            into ``input_word`` defining the cyclic permutation pattern.
        period: The number of distinct cyclic shifts to try (equal to
            the length of ``p``).

    Returns:
        A tuple ``(num_permutations_applied, permuted_word)`` where:

        - ``num_permutations_applied`` (int): The shift index (0 to
          ``period-1``) that produced the minimum tribit count.
        - ``permuted_word`` (np.ndarray): The best permuted bit array
          of length ``sector_length``.

    Example:
        >>> import numpy as np
        >>> word = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        >>> perm_idx = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        >>> n_shifts, best = permute(word, 8, perm_idx, 8)
        >>> isinstance(best, np.ndarray)
        True
        >>> len(best)
        8
    """
    p_length = len(p)

    # Evaluate the unshifted word (shift 0) as the initial minimum
    min_num_error_events = count_possible_error_events(input_word, sector_length)
    num_permutations = 0

    best_word = np.empty(sector_length, dtype=np.int64)
    best_word[:sector_length] = input_word[:sector_length]

    # Try each cyclic shift from 1 to period-1
    for i in range(1, period):
        # Apply permutation P^i: position P[(j+i) % PLen] gets InputWord[P[j]]
        for j in range(p_length):
            dst_idx = (j + i) % p_length
            best_word[p[dst_idx]] = input_word[p[j]]

        num_error_events = count_possible_error_events(best_word, sector_length)
        if num_error_events < min_num_error_events:
            min_num_error_events = num_error_events
            num_permutations = i

    # Build the final permuted word using the best shift
    result = np.empty(sector_length, dtype=np.int64)
    for j in range(p_length):
        dst_idx = (j + num_permutations) % p_length
        result[p[dst_idx]] = input_word[p[j]]

    return (num_permutations, result)

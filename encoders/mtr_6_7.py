"""6/7 MTR(2;8) encoder.

Translates 6 user bits into 7 code bits while enforcing the MTR(2;8)
constraint: no more than 2 consecutive transitions, and no run longer
than 8.

Includes a substitution pass that modifies boundary bits between
consecutive codewords to resolve constraint violations.

Based on EncodingFunctions.c Enc_6By7MTRCode.
"""

from __future__ import annotations

import math
import pathlib

import numpy as np

# ---------------------------------------------------------------------------
# Codeword lookup table (64 x 7)
# Loaded once at module initialization, equivalent to C's `static ReadFlag`.
# ---------------------------------------------------------------------------
_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_CODEWORD_PATH = _DATA_DIR / "Rate6By7MTR2-8.dat"

_Codewords: np.ndarray | None = None


def _load_codewords() -> np.ndarray:
    """Read the 6/7 MTR codeword file and return a (64, 7) integer array."""
    global _Codewords
    if _Codewords is not None:
        return _Codewords

    lines = _CODEWORD_PATH.read_text().strip().splitlines()
    rows = []
    for line in lines:
        stripped = line.strip()
        row = [int(ch) for ch in stripped]
        rows.append(row)
    _Codewords = np.array(rows, dtype=np.int64)
    return _Codewords


Codewords = _load_codewords()


def enc_6by7mtr_code(user_bits: np.ndarray, sector_length: int) -> np.ndarray:
    """Encode with 6/7 MTR(2;8) code.

    The encoder converts NRZ user bits to NRZI, maps every 6 NRZI bits
    to a 7-bit codeword, applies substitution rules to resolve boundary
    violations, then converts back to NRZ.

    The output is always ``nrzi_encoded_len + 1`` bits long.

    Args:
        user_bits: Array of 0/1 values representing the NRZ user data.
        sector_length: Desired sector length. Must be of the form 6Z+1;
            if not, it is auto-adjusted and a warning is printed.

    Returns:
        Encoded bit array of length ``nrzi_encoded_len + 1`` (matching
        the C implementation), containing NRZ codeword bits.

    Example:
        >>> import numpy as np
        >>> bits = np.array([0, 1, 0, 1, 0, 1, 0])
        >>> result = enc_6by7mtr_code(bits, len(bits))
        >>> len(result)
        8
    """
    # ---- Sector-length validation (6Z + 1) ----
    if sector_length % 6 != 1:
        original = sector_length
        sector_length = sector_length - (sector_length % 6) + 1
        print(
            f"WARNING: -> enc_6by7mtr_code() -> For 6/7 MTR(2;8) code, "
            f"sector length should be 6Z+1. "
            f"Changed from {original} to {sector_length}."
        )

    # ---- Truncate user_bits to sector_length ----
    user_bits = user_bits[:sector_length].copy()

    # ---- Allocate NRZI buffers ----
    nrzi_len = sector_length - 1
    rate = 6.0 / 7.0
    nrzi_encoded_len = int(math.ceil(nrzi_len / rate))

    nrzi_user = np.zeros(nrzi_len, dtype=np.int64)
    nrzi_encoded = np.zeros(nrzi_encoded_len, dtype=np.int64)

    # ---- NRZ -> NRZI conversion ----
    nrzi_user = np.abs(np.diff(user_bits))

    # ---- Encode NRZI user bits via codeword lookup ----
    num_blocks = nrzi_len // 6
    for i in range(num_blocks):
        start_src = i * 6
        start_dst = i * 7
        dec = 0
        for j in range(6):
            dec += int(math.pow(2, 5 - j)) * nrzi_user[start_src + j]
        nrzi_encoded[start_dst: start_dst + 7] = Codewords[dec]

    # ---- Substitution pass ----
    # Apply substitution rules at codeword boundaries to resolve MTR(2;8)
    # violations. Only one substitution is required per boundary.
    for i in range(num_blocks - 1):
        base = i * 7
        next_base = (i + 1) * 7

        # --- Substitution Type I ---
        # Pattern: [0,0,1] followed by [1,1,0]
        if (nrzi_encoded[base + 4] == 0 and nrzi_encoded[base + 5] == 0
                and nrzi_encoded[base + 6] == 1
                and nrzi_encoded[next_base] == 1
                and nrzi_encoded[next_base + 1] == 1
                and nrzi_encoded[next_base + 2] == 0):
            nrzi_encoded[base + 4] = 0
            nrzi_encoded[base + 5] = 1
            nrzi_encoded[base + 6] = 1
            nrzi_encoded[next_base] = 0
            nrzi_encoded[next_base + 1] = 0
            nrzi_encoded[next_base + 2] = 1
            continue

        # --- Substitution Type II ---
        # Pattern: [1,0,1] followed by [1,1,0]
        if (nrzi_encoded[base + 4] == 1 and nrzi_encoded[base + 5] == 0
                and nrzi_encoded[base + 6] == 1
                and nrzi_encoded[next_base] == 1
                and nrzi_encoded[next_base + 1] == 1
                and nrzi_encoded[next_base + 2] == 0):
            nrzi_encoded[base + 4] = 0
            nrzi_encoded[base + 5] = 1
            nrzi_encoded[base + 6] = 1
            nrzi_encoded[next_base] = 0
            nrzi_encoded[next_base + 1] = 1
            nrzi_encoded[next_base + 2] = 0
            continue

        # --- Substitution Type III ---
        # Pattern: [0,0,0] followed by [0,0,0]
        if (nrzi_encoded[base + 4] == 0 and nrzi_encoded[base + 5] == 0
                and nrzi_encoded[base + 6] == 0
                and nrzi_encoded[next_base] == 0
                and nrzi_encoded[next_base + 1] == 0
                and nrzi_encoded[next_base + 2] == 0):
            nrzi_encoded[base + 4] = 0
            nrzi_encoded[base + 5] = 1
            nrzi_encoded[base + 6] = 1
            nrzi_encoded[next_base] = 0
            nrzi_encoded[next_base + 1] = 0
            nrzi_encoded[next_base + 2] = 0
            continue

    # ---- NRZI encoded -> NRZ output ----
    # Assume the first NRZ encoded bit is 0
    encoded = np.zeros(nrzi_encoded_len + 1, dtype=np.int64)
    encoded[0] = 0
    for i in range(nrzi_encoded_len):
        if nrzi_encoded[i] == 0:
            encoded[i + 1] = encoded[i]
        else:
            encoded[i + 1] = (encoded[i] + 1) % 2

    return encoded

"""6/7 MTR(2;8) decoder.

Detects and undoes substitution patterns applied during encoding,
then performs codeword lookup to recover the original 6-bit user words.

Based on DecodingFunctions.c Dec_6By7MTRCode.
"""

from __future__ import annotations

import pathlib

import numpy as np

# ---------------------------------------------------------------------------
# Codeword lookup table (64 x 7) + decimal lookup
# ---------------------------------------------------------------------------
_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_CODEWORD_PATH = _DATA_DIR / "Rate6By7MTR2-8.dat"

_Codewords: np.ndarray | None = None
_Dec2Codeword: dict[int, np.ndarray] | None = None


def _load_codewords() -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Read the 6/7 MTR codeword file and return codewords and a reverse lookup."""
    global _Codewords, _Dec2Codeword
    if _Codewords is not None and _Dec2Codeword is not None:
        return _Codewords, _Dec2Codeword

    lines = _CODEWORD_PATH.read_text().strip().splitlines()
    rows = []
    dec_to_cw: dict[int, np.ndarray] = {}

    for i, line in enumerate(lines):
        stripped = line.strip()
        row = np.array([int(ch) for ch in stripped], dtype=np.int64)
        rows.append(row)
        # Decimal value (MSB first, 7 bits)
        dec = 0
        for j in range(7):
            dec += (2 ** (6 - j)) * int(row[j])
        dec_to_cw[dec] = row

    _Codewords = np.array(rows, dtype=np.int64)
    _Dec2Codeword = dec_to_cw
    return _Codewords, _Dec2Codeword


Codewords: np.ndarray
Dec2Codeword: dict[int, np.ndarray]
Codeword2Dec: dict[int, int]
Codewords, Dec2Codeword = _load_codewords()

_Codeword2Dec: dict[int, int] = {}
for i, cw in enumerate(Codewords):
    cw_dec = 0
    for j in range(7):
        cw_dec += (2 ** (6 - j)) * int(cw[j])
    _Codeword2Dec[cw_dec] = i
Codeword2Dec = _Codeword2Dec


def dec_6by7mtr_code(
    nrz_viterbi_output: np.ndarray,
    pre_padding_length: int,
    sector_length: int,
) -> np.ndarray:
    """Decode 6/7 MTR(2;8) code.

    Steps (matching C Dec_6By7MTRCode):
    1. Strip pre-padding.
    2. Convert NRZ to NRZI (differencing).
    3. Detect and undo substitution patterns at codeword boundaries.
    4. Map every 7 NRZI bits to a 6-bit user word via codeword lookup.
    5. Convert NRZI decoded bits back to NRZ.

    Args:
        nrz_viterbi_output: Full detector output including padding.
        pre_padding_length: Number of padding bits at the start.
        sector_length: Length of the coded sector (before padding).

    Returns:
        NRZ decoded output array of length ``sector_length``.
    """
    # Strip padding
    coded = nrz_viterbi_output[
        pre_padding_length: pre_padding_length + sector_length
    ].copy()

    # NRZ -> NRZI conversion
    nrzi_len = sector_length - 1
    nrzi = np.zeros(nrzi_len, dtype=np.int64)
    for i in range(1, sector_length):
        nrzi[i - 1] = abs(int(coded[i]) - int(coded[i - 1]))

    # Detect and undo substitutions
    decoded_rate = 6.0 / 7.0
    num_blocks = nrzi_len // 7

    for i in range(num_blocks - 1):
        base = i * 7
        next_base = (i + 1) * 7

        # Check for substitution pattern:
        # Current block ends with [0,1,1] at positions 4,5,6
        if (nrzi[base + 4] == 0 and nrzi[base + 5] == 1
                and nrzi[base + 6] == 1):

            # Type I: next block starts [1,1,0]
            if (nrzi[next_base] == 1 and nrzi[next_base + 1] == 1
                    and nrzi[next_base + 2] == 0):
                nrzi[base + 4] = 0
                nrzi[base + 5] = 0
                nrzi[base + 6] = 1
                nrzi[next_base] = 1
                nrzi[next_base + 1] = 1
                nrzi[next_base + 2] = 0
                continue

            # Type II: next block starts [1,1,0] with different current pattern
            if (nrzi[base + 4] == 1 and nrzi[base + 5] == 0
                    and nrzi[base + 6] == 1
                    and nrzi[next_base] == 1
                    and nrzi[next_base + 1] == 1
                    and nrzi[next_base + 2] == 0):
                nrzi[base + 4] = 1
                nrzi[base + 5] = 0
                nrzi[base + 6] = 1
                nrzi[next_base] = 1
                nrzi[next_base + 1] = 1
                nrzi[next_base + 2] = 0
                continue

            # Type III: next block starts [0,0,0]
            if (nrzi[next_base] == 0 and nrzi[next_base + 1] == 0
                    and nrzi[next_base + 2] == 0):
                nrzi[base + 4] = 0
                nrzi[base + 5] = 0
                nrzi[base + 6] = 0
                nrzi[next_base] = 0
                nrzi[next_base + 1] = 0
                nrzi[next_base + 2] = 0
                continue

    # Decode: every 7 NRZI bits -> 6 user bits
    nrzi_decoded = np.zeros(int(nrzi_len * decoded_rate), dtype=np.int64)

    for i in range(num_blocks):
        block = nrzi[i * 7: (i + 1) * 7]
        dec = 0
        for j in range(7):
            dec += (2 ** (6 - j)) * int(block[j])

        if dec in Codeword2Dec:
            nrzi_val = Codeword2Dec[dec]
            for k in range(6):
                nrzi_decoded[i * 6 + k] = (nrzi_val >> (5 - k)) & 1
        else:
            # Invalid codeword: fill with zeros
            for k in range(6):
                nrzi_decoded[i * 6 + k] = 0

    # NRZI decoded -> NRZ output
    decoded_output = np.zeros(sector_length, dtype=np.int64)
    decoded_output[0] = 0
    for i in range(len(nrzi_decoded)):
        if nrzi_decoded[i] == 0:
            decoded_output[i + 1] = decoded_output[i]
        else:
            decoded_output[i + 1] = (decoded_output[i] + 1) % 2

    return decoded_output

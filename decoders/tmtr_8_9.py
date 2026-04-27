"""8/9 TMTR(2/3;11) decoder.

Performs codeword lookup on the NRZI-detected bits to recover
the original 8-bit user words.

Based on DecodingFunctions.c Dec_8By9TMTRCode.
"""

from __future__ import annotations

import pathlib

import numpy as np

# ---------------------------------------------------------------------------
# Codeword lookup table (256 x 9) + decimal lookup
# ---------------------------------------------------------------------------
_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_CODEWORD_PATH = _DATA_DIR / "Rate8By9TMTR.dat"

_Codewords: np.ndarray | None = None
_Dec2Codeword: dict[int, np.ndarray] | None = None


def _load_codewords() -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Read the 8/9 TMTR codeword file and return codewords and a reverse lookup."""
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
        # Decimal value (MSB first, 9 bits)
        dec = 0
        for j in range(9):
            dec += (2 ** (8 - j)) * int(row[j])
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
    for j in range(9):
        cw_dec += (2 ** (8 - j)) * int(cw[j])
    _Codeword2Dec[cw_dec] = i
Codeword2Dec = _Codeword2Dec


def dec_8by9tmtr_code(
    nrz_viterbi_output: np.ndarray,
    pre_padding_length: int,
    sector_length: int,
) -> tuple[np.ndarray, int]:
    """Decode 8/9 TMTR(2/3;11) code.

    Steps (matching C Dec_8By9TMTRCode):
    1. Strip pre-padding.
    2. Convert NRZ to NRZI (differencing).
    3. Map every 9 NRZI bits to an 8-bit user word via codeword lookup.
    4. Convert NRZI decoded bits back to NRZ.
    5. Count invalid codewords.

    Args:
        nrz_viterbi_output: Full detector output including padding.
        pre_padding_length: Number of padding bits at the start.
        sector_length: Length of the coded sector (before padding).

    Returns:
        A tuple ``(decoded_output, invalid_codeword_count)`` where
        ``decoded_output`` is an int64 array of NRZ user bits and
        ``invalid_codeword_count`` is the number of invalid codewords.
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

    # Decode: every 9 NRZI bits -> 8 user bits
    decoded_rate = 8.0 / 9.0
    decoded_len = int(nrzi_len * decoded_rate)
    nrzi_decoded = np.zeros(decoded_len, dtype=np.int64)
    invalid_cw = 0
    num_blocks = nrzi_len // 9

    for i in range(num_blocks):
        block = nrzi[i * 9: (i + 1) * 9]
        dec = 0
        for j in range(9):
            dec += (2 ** (8 - j)) * int(block[j])

        if dec in Codeword2Dec:
            nrzi_val = Codeword2Dec[dec]
            for k in range(8):
                nrzi_decoded[i * 8 + k] = (nrzi_val >> (7 - k)) & 1
        else:
            invalid_cw += 1
            # Fill with zeros

    # NRZI decoded -> NRZ output
    decoded_output = np.zeros(sector_length, dtype=np.int64)
    decoded_output[0] = 0
    for i in range(decoded_len):
        if nrzi_decoded[i] == 0:
            decoded_output[i + 1] = decoded_output[i]
        else:
            decoded_output[i + 1] = (decoded_output[i] + 1) % 2

    return decoded_output, invalid_cw

"""4/5 RLL(0,2) decoder.

Converts the NRZI hard-output from the Viterbi/SOVA detector back
to NRZ user data by reversing the 4/5 RLL(0,2) encoding.

Based on DecodingFunctions.c Dec_4By5RLLCode.
"""

from __future__ import annotations

import pathlib

import numpy as np

# ---------------------------------------------------------------------------
# Codeword lookup table (16 x 5) + decimal lookup
# ---------------------------------------------------------------------------
_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_CODEWORD_PATH = _DATA_DIR / "Rate4By5RLLCode.dat"

_Codewords: np.ndarray | None = None
_Dec2Codeword: dict[int, np.ndarray] | None = None


def _load_codewords() -> tuple[np.ndarray, dict[int, np.ndarray]]:
    """Read the 4/5 RLL codeword file and return codewords and a reverse lookup."""
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
        # Compute decimal value (MSB first)
        dec = 0
        for j in range(5):
            dec += (2 ** (4 - j)) * int(row[j])
        dec_to_cw[dec] = row

    _Codewords = np.array(rows, dtype=np.int64)
    _Dec2Codeword = dec_to_cw
    return _Codewords, _Dec2Codeword


Codewords: np.ndarray
Dec2Codeword: dict[int, np.ndarray]
Codeword2Dec: dict[int, int]  # codeword decimal -> NRZI decimal (decoded value)
Codewords, Dec2Codeword = _load_codewords()

# Build reverse lookup: for each codeword, compute its decimal value
# and map it back to the original NRZI index (i.e., the decoded value).
_Codeword2Dec: dict[int, int] = {}
for i, cw in enumerate(Codewords):
    cw_dec = 0
    for j in range(5):
        cw_dec += (2 ** (4 - j)) * int(cw[j])
    _Codeword2Dec[cw_dec] = i
Codeword2Dec = _Codeword2Dec


def dec_4by5rll_code(
    nrz_viterbi_output: np.ndarray,
    pre_padding_length: int,
    sector_length: int,
) -> tuple[np.ndarray, int]:
    """Decode 4/5 RLL(0,2) code.

    Steps (matching C Dec_4By5RLLCode):
    1. Strip pre-padding from the Viterbi output.
    2. Convert NRZ to NRZI (differencing).
    3. Map every 5 NRZI bits to a 4-bit user word via codeword lookup.
    4. Convert NRZI decoded bits back to NRZ.
    5. Count invalid codewords.

    Args:
        nrz_viterbi_output: Full detector output including padding.
        pre_padding_length: Number of padding bits at the start.
        sector_length: Length of the coded sector (before padding).

    Returns:
        A tuple ``(decoded_output, invalid_codeword_count)`` where
        ``decoded_output`` is an int64 array of NRZ user bits and
        ``invalid_codeword_count`` is the number of invalid codewords
        encountered (decoded as zero on each).
    """
    # Strip padding
    coded = nrz_viterbi_output[
        pre_padding_length: pre_padding_length + sector_length
    ].copy()

    # NRZ -> NRZI conversion (differencing)
    nrzi_len = sector_length - 1
    nrzi = np.zeros(nrzi_len, dtype=np.int64)
    for i in range(1, sector_length):
        nrzi[i - 1] = abs(int(coded[i]) - int(coded[i - 1]))

    # Decode: every 5 NRZI bits -> 4 user bits
    decoded_rate = 4.0 / 5.0
    decoded_len = int(nrzi_len * decoded_rate)
    nrzi_decoded = np.zeros(decoded_len, dtype=np.int64)
    invalid_cw = 0
    num_blocks = nrzi_len // 5

    for i in range(num_blocks):
        # Decimal value of the 5-bit block
        block = nrzi[i * 5: (i + 1) * 5]
        dec = 0
        for j in range(5):
            dec += (2 ** (4 - j)) * int(block[j])

        if dec in Codeword2Dec:
            nrzi_val = Codeword2Dec[dec]
            for k in range(4):
                nrzi_decoded[i * 4 + k] = (nrzi_val >> (3 - k)) & 1
        else:
            invalid_cw += 1
            # Fill with zeros (matching C behaviour)

    # NRZI decoded -> NRZ output
    decoded_output = np.zeros(sector_length, dtype=np.int64)
    decoded_output[0] = 0
    for i in range(decoded_len):
        if nrzi_decoded[i] == 0:
            decoded_output[i + 1] = decoded_output[i]
        else:
            decoded_output[i + 1] = (decoded_output[i] + 1) % 2

    return decoded_output, invalid_cw

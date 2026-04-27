"""4/5 RLL(0,2) encoder.

Translates 4 user bits into 5 code bits ensuring at most 2 consecutive
transitions (run-length limited constraint d=0, k=2).

Based on EncodingFunctions.c Enc_4By5RLLCode.
"""

import math
import pathlib

import numpy as np

# ---------------------------------------------------------------------------
# Codeword lookup table (16 x 5)
# Loaded once at module initialization, equivalent to C's `static ReadFlag`.
# ---------------------------------------------------------------------------
_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_CODEWORD_PATH = _DATA_DIR / "Rate4By5RLLCode.dat"

_Codewords: np.ndarray | None = None


def _load_codewords() -> np.ndarray:
    """Read the 4/5 RLL codeword file and return a (16, 5) integer array."""
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


def enc_4by5rll_code(user_bits: np.ndarray, sector_length: int) -> np.ndarray:
    """Encode user bits with 4/5 RLL(0,2) code.

    The encoder converts NRZ user bits to NRZI, maps every 4 NRZI bits to
    a 5-bit codeword from the lookup table, then converts back to NRZ.

    The output is always ``nrzi_encoded_len + 1`` bits long, where
    ``nrzi_encoded_len = ceil((sector_length - 1) / (4/5))``.
    For standard sector lengths (4Z+1), this produces ``sector_length + 1``.

    Args:
        user_bits: Array of 0/1 values representing the NRZ user data.
        sector_length: Desired sector length. Must be of the form 4Z+1;
            if not, it is auto-adjusted and a warning is printed.

    Returns:
        Encoded bit array of length ``nrzi_encoded_len + 1`` (matching
        the C implementation), containing NRZ codeword bits.

    Example:
        >>> import numpy as np
        >>> bits = np.array([0, 1, 0, 1, 0])
        >>> result = enc_4by5rll_code(bits, len(bits))
        >>> len(result)
        6
    """
    # ---- Sector-length validation (4Z + 1) ----
    if sector_length % 4 != 1:
        original = sector_length
        sector_length = sector_length - (sector_length % 4) + 1
        print(
            f"WARNING: -> enc_4by5rll_code() -> For 4/5 RLL(0,2) code, "
            f"sector length should be 4Z+1, where Z is an integer. "
            f"Changed from {original} to {sector_length}."
        )

    # ---- Truncate user_bits to sector_length ----
    user_bits = user_bits[:sector_length].copy()

    # ---- Allocate NRZI buffers ----
    nrzi_len = sector_length - 1
    nrzi_encoded_len = int(math.ceil(nrzi_len / (4.0 / 5.0)))

    nrzi_user = np.zeros(nrzi_len, dtype=np.int64)
    nrzi_encoded = np.zeros(nrzi_encoded_len, dtype=np.int64)

    # ---- NRZ -> NRZI conversion ----
    # NRZI[i] = abs(user_bits[i+1] - user_bits[i])
    nrzi_user = np.abs(np.diff(user_bits))

    # ---- Encode NRZI user bits via codeword lookup ----
    num_blocks = nrzi_len // 4
    for i in range(num_blocks):
        start_src = i * 4
        start_dst = i * 5
        dec = 0
        for j in range(4):
            dec += int(math.pow(2, 3 - j)) * nrzi_user[start_src + j]
        nrzi_encoded[start_dst: start_dst + 5] = Codewords[dec]

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

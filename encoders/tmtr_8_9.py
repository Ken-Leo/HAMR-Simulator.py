"""8/9 TMTR(2/3;11) encoder.

Translates 8 user bits into 9 code bits while enforcing the
Time-Varying Maximum Transition Run TMTR(2/3;11) constraint.

Based on EncodingFunctions.c Enc_8By9TMTRCode.
"""

import math
import pathlib

import numpy as np

# ---------------------------------------------------------------------------
# Codeword lookup table (256 x 9)
# Loaded once at module initialization, equivalent to C's `static ReadFlag`.
# ---------------------------------------------------------------------------
_DATA_DIR = pathlib.Path(__file__).resolve().parent.parent / "data"
_CODEWORD_PATH = _DATA_DIR / "Rate8By9TMTR.dat"

_Codewords: np.ndarray | None = None


def _load_codewords() -> np.ndarray:
    """Read the 8/9 TMTR codeword file and return a (256, 9) integer array."""
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


def enc_8by9tmtr_code(user_bits: np.ndarray, sector_length: int) -> np.ndarray:
    """Encode with 8/9 TMTR(2/3;11) code.

    The encoder converts NRZ user bits to NRZI, maps every 8 NRZI bits
    to a 9-bit codeword from the lookup table, then converts back to NRZ.

    The output length is ``nrzi_encoded_len + 1``.

    Args:
        user_bits: Array of 0/1 values representing the NRZ user data.
        sector_length: Desired sector length. Must be of the form 8Z+1;
            if not, it is auto-adjusted and a warning is printed.

    Returns:
        Encoded bit array of length ``nrzi_encoded_len + 1`` (matching
        the C implementation), containing NRZ codeword bits.

    Example:
        >>> import numpy as np
        >>> bits = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0])
        >>> result = enc_8by9tmtr_code(bits, len(bits))
        >>> len(result)
        10
    """
    # ---- Sector-length validation (8Z + 1) ----
    if sector_length % 8 != 1:
        original = sector_length
        sector_length = sector_length - (sector_length % 8) + 1
        print(
            f"WARNING: -> enc_8by9tmtr_code() -> For 8/9 TMTR(2/3;11) code, "
            f"sector length should be 8Z+1. "
            f"Changed from {original} to {sector_length}."
        )

    # ---- Truncate user_bits to sector_length ----
    user_bits = user_bits[:sector_length].copy()

    # ---- Allocate NRZI buffers ----
    nrzi_len = sector_length - 1
    rate = 8.0 / 9.0
    nrzi_encoded_len = int(math.ceil(nrzi_len / rate))

    nrzi_user = np.zeros(nrzi_len, dtype=np.int64)
    nrzi_encoded = np.zeros(nrzi_encoded_len, dtype=np.int64)

    # ---- NRZ -> NRZI conversion ----
    nrzi_user = np.abs(np.diff(user_bits))

    # ---- Encode NRZI user bits via codeword lookup ----
    num_blocks = nrzi_len // 8
    for i in range(num_blocks):
        start_src = i * 8
        start_dst = i * 9
        dec = 0
        for j in range(8):
            dec += int(math.pow(2, 7 - j)) * nrzi_user[start_src + j]
        nrzi_encoded[start_dst: start_dst + 9] = Codewords[dec]

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

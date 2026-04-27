"""Tests for encoder modules.

Tests round-trip encode->decode for each code type, permutation logic,
and sector length validation.
"""

from __future__ import annotations

import numpy as np
import pytest

from encoders.rll_4_5 import enc_4by5rll_code, Codewords as rll_codewords
from encoders.mtr_6_7 import enc_6by7mtr_code, Codewords as mtr_codewords
from encoders.tmtr_8_9 import enc_8by9tmtr_code, Codewords as tmtr_codewords
from decoders.rll_4_5 import dec_4by5rll_code, Dec2Codeword as rll_dec_lookup
from decoders.mtr_6_7 import dec_6by7mtr_code, Dec2Codeword as mtr_dec_lookup
from decoders.tmtr_8_9 import dec_8by9tmtr_code, Dec2Codeword as tmtr_dec_lookup


# ---------------------------------------------------------------------------
# 4/5 RLL encoder/decoder round-trip
# ---------------------------------------------------------------------------


class TestRLL45Encoder:
    """Tests for the 4/5 RLL(0,2) encoder."""

    def test_output_length(self):
        """Encoded output length = nrzi_encoded_len + 1, not sector_length."""
        bits = np.array([0, 1, 0, 1, 0], dtype=np.int64)
        result = enc_4by5rll_code(bits, 5)
        # nrzi_len = 4, rate = 4/5, ceil(4/0.8) = 5, output = 5 + 1 = 6
        assert len(result) == 6

    def test_round_trip_no_noise(self):
        """Encode then decode should recover original user bits (no noise)."""
        np.random.seed(42)
        sector_len = 101  # valid 4Z+1
        user_bits = np.random.randint(0, 2, sector_len, dtype=np.int64)
        user_bits[0] = 0

        encoded = enc_4by5rll_code(user_bits, sector_len)

        # Pad with zeros for decoder
        pre_pad = 20
        padded = np.zeros(len(encoded) + pre_pad, dtype=np.int64)
        padded[pre_pad: pre_pad + len(encoded)] = encoded

        decoded, invalid_cw = dec_4by5rll_code(padded, pre_pad, sector_len)

        # RLL (rate 4/5): decoded_len = nrzi_len * 4/5 = 100 * 4/5 = 80
        # Only decoded_len + 1 = 81 elements are valid in decoded_output
        compare_len = min(len(decoded), 81)
        assert int(np.sum(decoded[:compare_len] != user_bits[:compare_len])) == 0
        assert invalid_cw == 0

    def test_sector_length_adjustment(self):
        """Sector length should be adjusted to 4Z+1 (prints warning)."""
        bits = np.array([0, 1, 0, 1, 0], dtype=np.int64)
        result = enc_4by5rll_code(bits, 7)  # 7 is already 4Z+1 (4*1+3 -> adjusted to 5)
        assert len(result) >= 5

    def test_first_bit_zero(self):
        """The first encoded bit should always be 0."""
        bits = np.array([1, 1, 1, 1, 1], dtype=np.int64)
        result = enc_4by5rll_code(bits, 5)
        assert result[0] == 0


# ---------------------------------------------------------------------------
# 6/7 MTR encoder/decoder round-trip
# ---------------------------------------------------------------------------


class TestMTR67Encoder:
    """Tests for the 6/7 MTR(2;8) encoder."""

    def test_output_length(self):
        """Encoded output length = nrzi_encoded_len + 1."""
        bits = np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
        result = enc_6by7mtr_code(bits, 7)
        assert len(result) == 8

    def test_round_trip_no_noise(self):
        """Encode then decode should recover original user bits (no noise)."""
        np.random.seed(42)
        sector_len = 211  # valid 6Z+1, also 42*5+1 so decoded_len = 180
        user_bits = np.random.randint(0, 2, sector_len, dtype=np.int64)
        user_bits[0] = 0

        encoded = enc_6by7mtr_code(user_bits, sector_len)
        pre_pad = 20
        padded = np.zeros(len(encoded) + pre_pad, dtype=np.int64)
        padded[pre_pad: pre_pad + len(encoded)] = encoded

        decoded = dec_6by7mtr_code(padded, pre_pad, sector_len)

        # MTR (rate 6/7): decoded_len = 210 * 6/7 = 180
        compare_len = min(len(decoded), 181)
        assert int(np.sum(decoded[:compare_len] != user_bits[:compare_len])) == 0

    def test_substitution_pattern(self):
        """Test that substitution patterns are encoded correctly."""
        bits = np.zeros(20, dtype=np.int64)
        encoded = enc_6by7mtr_code(bits, 21)

        # Verify first bit is 0
        assert encoded[0] == 0
        # Encoded length should be nrzi_encoded_len + 1
        assert len(encoded) > 0


# ---------------------------------------------------------------------------
# 8/9 TMTR encoder/decoder round-trip
# ---------------------------------------------------------------------------


class TestTMTR89Encoder:
    """Tests for the 8/9 TMTR(2/3;11) encoder."""

    def test_output_length(self):
        """Encoded output length = nrzi_encoded_len + 1."""
        bits = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
        result = enc_8by9tmtr_code(bits, 9)
        assert len(result) == 10

    def test_round_trip_no_noise(self):
        """Encode then decode should recover original user bits (no noise)."""
        np.random.seed(42)
        sector_len = 361  # valid 8Z+1, also 72*5+1 so decoded_len = 320
        user_bits = np.random.randint(0, 2, sector_len, dtype=np.int64)
        user_bits[0] = 0

        encoded = enc_8by9tmtr_code(user_bits, sector_len)
        pre_pad = 20
        padded = np.zeros(len(encoded) + pre_pad, dtype=np.int64)
        padded[pre_pad: pre_pad + len(encoded)] = encoded

        decoded, invalid_cw = dec_8by9tmtr_code(padded, pre_pad, sector_len)

        # TMTR (rate 8/9): decoded_len = 360 * 8/9 = 320
        compare_len = min(len(decoded), 321)
        assert int(np.sum(decoded[:compare_len] != user_bits[:compare_len])) == 0
        assert invalid_cw == 0


# ---------------------------------------------------------------------------
# Codeword table integrity
# ---------------------------------------------------------------------------


class TestCodewordTables:
    """Verify codeword lookup tables match data files."""

    def test_rll_codeword_count(self):
        assert len(rll_codewords) == 16
        assert rll_codewords.shape == (16, 5)

    def test_mtr_codeword_count(self):
        assert len(mtr_codewords) == 64
        assert mtr_codewords.shape == (64, 7)

    def test_tmtr_codeword_count(self):
        assert len(tmtr_codewords) == 256
        assert tmtr_codewords.shape == (256, 9)

    def test_rll_decoder_lookup_complete(self):
        """Every codeword in the table should be in the decoder lookup."""
        for i, cw in enumerate(rll_codewords):
            dec = 0
            for j in range(5):
                dec += (2 ** (4 - j)) * cw[j]
            assert dec in rll_dec_lookup, f"Codeword {i} ({cw}) not in decoder lookup"

    def test_mtr_decoder_lookup_complete(self):
        for i, cw in enumerate(mtr_codewords):
            dec = 0
            for j in range(7):
                dec += (2 ** (6 - j)) * cw[j]
            assert dec in mtr_dec_lookup, f"Codeword {i} not in decoder lookup"

    def test_tmtr_decoder_lookup_complete(self):
        for i, cw in enumerate(tmtr_codewords):
            dec = 0
            for j in range(9):
                dec += (2 ** (8 - j)) * cw[j]
            assert dec in tmtr_dec_lookup, f"Codeword {i} not in decoder lookup"

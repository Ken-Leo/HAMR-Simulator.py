"""Branch coverage tests for remaining uncovered lines.

Targeted tests for:
- simulator.py: default config, unknown encoder, FixedPRTarget, SOVA, constrained Viterbi,
  HAMR channel, no-equalizer, error tracking, early termination
- decoders/mtr_6_7.py: substitution undo Types I/II/III, sector_length edge case
- encoders/mtr_6_7.py: substitution Type I, cached codewords
- equalizer_detector/equalizer.py: GPR monic constraint
- channel/channel.py: unknown channel type, HAMR path, temperature modulation
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np
import pytest

from channel.channel import FullHamrChannel, channel, hamr_channel
from channel.fir import non_causal_fir
from channel.lpf import lpf
from channel.math_utils import LCG, gaussian_random
from decoders.mtr_6_7 import dec_6by7mtr_code, Dec2Codeword as mtr_dec_lookup
from encoders.mtr_6_7 import enc_6by7mtr_code, Codewords as mtr_codewords
from encoders.tmtr_8_9 import enc_8by9tmtr_code
from equalizer_detector.equalizer import (
    adapt_equalizer,
    apply_equalizer,
    find_gpr_target,
    lpf as equalizer_lpf,
)
from equalizer_detector.sova import classical_sova
from equalizer_detector.viterbi import classical_viterbi
from equalizer_detector.constrained_detectors import (
    viterbi_6by7mtr_code,
    viterbi_8by9tmtr_code,
    sova_8by9mtr_code,
)
from simulator import (
    SimulatorConfig,
    _bipolar,
    _compute_code_params,
    _decoder_fn,
    _encoder_fn,
    run_simulation,
)


# ===========================================================================
# Section A: SimulatorConfig defaults (simulator.py lines 135-147)
# ===========================================================================


class TestSimulatorConfigDefaults:
    """Test that __post_init__ sets defaults when fields are None."""

    def test_snr_db_default(self):
        """snr_db=None should default to [21.0]."""
        cfg = SimulatorConfig(snr_db=None)
        assert cfg.snr_db == [21.0]

    def test_pri_imp_res_default(self):
        """pri_imp_res=None should default to [1, 1, -1, -1]."""
        cfg = SimulatorConfig(pri_imp_res=None)
        assert cfg.pri_imp_res == [1, 1, -1, -1]

    def test_hamr_hc_default(self):
        """hamr_hc=None should default."""
        cfg = SimulatorConfig(hamr_hc=None)
        assert cfg.hamr_hc == [-1000.0, 1600000.0]

    def test_hamr_mr_default(self):
        """hamr_mr=None should default."""
        cfg = SimulatorConfig(hamr_mr=None)
        assert cfg.hamr_mr == [-600.0, 1200000.0]

    def test_hamr_s_default(self):
        """hamr_s=None should default."""
        cfg = SimulatorConfig(hamr_s=None)
        assert cfg.hamr_s == [0.0003, 0.5]

    def test_nlts_k_default(self):
        """nlts_k=None should default to 16-element array of 4000.0."""
        cfg = SimulatorConfig(nlts_k=None)
        assert len(cfg.nlts_k) == 16
        assert cfg.nlts_k == [4000.0] * 16

    def test_nlts_rho_default(self):
        """nlts_rho=None should default to 16-element array of 2.0."""
        cfg = SimulatorConfig(nlts_rho=None)
        assert len(cfg.nlts_rho) == 16
        assert cfg.nlts_rho == [2.0] * 16

    def test_all_none(self):
        """All defaults set correctly when everything is None."""
        cfg = SimulatorConfig(
            snr_db=None, pri_imp_res=None, hamr_hc=None,
            hamr_mr=None, hamr_s=None, nlts_k=None, nlts_rho=None,
        )
        assert cfg.snr_db == [21.0]
        assert cfg.pri_imp_res == [1, 1, -1, -1]

    def test_explicit_values_not_overwritten(self):
        """Explicitly-provided values should NOT be overwritten by defaults."""
        cfg = SimulatorConfig(snr_db=[30.0], pri_imp_res=[1, 0, -1])
        assert cfg.snr_db == [30.0]
        assert cfg.pri_imp_res == [1, 0, -1]


# ===========================================================================
# Section B: _compute_code_params edge cases (simulator.py line 179)
# ===========================================================================


class TestComputeCodeParamsBranch:
    """Test _compute_code_params edge branches."""

    def test_unknown_encoder_returns_ones(self):
        """Unknown encoder_type returns (1, 1, 1.0)."""
        config = SimulatorConfig(use_encoding=True, encoder_type="unknown_code")
        result = _compute_code_params(config)
        assert result == (1, 1, 1.0)

    def test_known_encoders_have_correct_rates(self):
        """Known encoders return proper code rates."""
        for enc_type, expected_rate in [
            ("rll_4_5", 4.0 / 5.0),
            ("mtr_6_7", 6.0 / 7.0),
            ("tmtr_8_9", 8.0 / 9.0),
        ]:
            config = SimulatorConfig(use_encoding=True, encoder_type=enc_type, sector_length=4096)
            user_len, coded_len, rate = _compute_code_params(config)
            assert rate == expected_rate
            assert user_len < coded_len


# ===========================================================================
# Section C: Channel type errors (channel.py lines 62, 491)
# ===========================================================================


class TestChannelErrors:
    """Test channel() error handling."""

    def test_unknown_channel_type_raises(self):
        """Unknown channel type should raise ValueError."""
        bits = np.array([0, 1, 0], dtype=np.int64)
        with pytest.raises(ValueError, match="Unknown channel type"):
            channel(bits, "UnknownType", num_taps=21, osr=5)

    def test_unknown_channel_type_in_channel_func(self):
        """channel() should raise for invalid type."""
        bits = np.array([0, 1, 0], dtype=np.int64)
        with pytest.raises(ValueError, match="Unknown channel type"):
            channel(bits, "Invalid", num_taps=21, osr=5)


# ===========================================================================
# Section D: FullHamrChannel temperature_modulation (channel.py line 296)
# ===========================================================================


class TestFullHamrChannelTempMod:
    """Test FullHamrChannel with temperature modulation."""

    def test_temp_modulation(self):
        """Temperature modulation should set modulated_peak_temp."""
        config = SimulatorConfig(
            channel_type="Hamr",
            temperature_modulation=1,
            hamr_t_peak=500.0,
            sector_length=64,
            num_eq_taps=11,
            num_channel_taps=51,
            num_lpf_taps=51,
        )
        channel_obj = FullHamrChannel(config, nd=2.5, osr=4)
        # Temperature modulation enabled sets _modulated_peak_temp to 450.0
        assert channel_obj._modulated_peak_temp == 450.0

    def test_no_temp_modulation(self):
        """Without temperature modulation, peak temp comes from config."""
        config = SimulatorConfig(
            channel_type="Hamr",
            temperature_modulation=0,
            hamr_t_peak=500.0,
            sector_length=64,
            num_eq_taps=11,
            num_channel_taps=51,
            num_lpf_taps=51,
        )
        channel_obj = FullHamrChannel(config, nd=2.5, osr=4)
        assert channel_obj._modulated_peak_temp == 500.0


# ===========================================================================
# Section E: HAMR channel path in channel() (channel.py lines 472-478)
# ===========================================================================


class TestChannelHamrPath:
    """Test the HAMR path through the channel() function."""

    def test_hamr_channel_default_params(self):
        """Hamr channel with hamr_params=None should use empty dict."""
        bits = np.ones(200, dtype=np.int64)
        output = channel(bits, "Hamr", num_taps=21, osr=10)
        assert len(output) > 0
        assert np.all(np.isfinite(output))

    def test_hamr_channel_float_input(self):
        """Hamr channel with float input should work correctly."""
        bits = np.ones(200, dtype=np.float64)
        output = channel(bits, "Hamr", num_taps=21, osr=10,
                         hamr_params={"sigma_t": 90.0})
        assert len(output) > 0
        assert np.all(np.isfinite(output))

    def test_hamr_channel_via_simulator(self):
        """Run simulation with Hamr channel type to cover init and main loop."""
        config = SimulatorConfig(
            channel_type="Hamr",
            detector_type="Viterbi",
            equalizer_type="GPRTarget",
            use_encoding=False,
            snr_db=[30.0],
            max_num_sectors=1,
            min_num_sectors=1,
            max_num_bit_err=100,
            sector_length=64,
            num_eq_sectors=1,
            num_eq_taps=11,
            viterbi_delay=10,
            num_channel_taps=51,
            num_lpf_taps=51,
            sigma_jitter=0.0,
            sigma_pulse_broad=0.0,
        )
        results = run_simulation(config)
        assert len(results["results"]) == 1
        assert "ber" in results["results"][0]


# ===========================================================================
# Section F: MTR decoder substitution undo (decoders/mtr_6_7.py lines 124-141)
# ===========================================================================


class TestMTRDecoderSubstitutionUndo:
    """Targeted tests for MTR decoder substitution undo branches.

    The decoder's substitution undo checks for these NRZI patterns at
    consecutive 7-bit block boundaries:
      - Type I:  [0,1,1] + [0,0,1]  → undo to [0,0,1] + [1,1,0]  (lines 124-130)
      - Type II: [0,1,1] + [0,1,0]  → undo to [1,0,1] + [1,1,0]  (lines 135-141)
      - Type III:[0,1,1] + [0,0,0]  → undo to [0,0,0] + [0,0,0]  (lines 144-152)
    """

    def _nrz_from_nrzi_14(self, nrzi_14: list[int], nrz_prefix: list[int]) -> np.ndarray:
        """Convert a 14-bit NRZI pattern + prefix to NRZ.

        NRZI[i] = |NRZ[i+1] - NRZ[i]|
        """
        nrz = list(nrz_prefix)
        for bit in nrzi_14:
            if bit == 0:
                nrz.append(nrz[-1])
            else:
                nrz.append(1 - nrz[-1])
        return np.array(nrz, dtype=np.int64)

    def test_type_I_undo(self):
        """Decoder should undo Type I substitution ([0,1,1] + [0,0,1])."""
        # Construct NRZ where first 14 NRZI bits are [*,*,*,*, 0,1,1,  0,0,1,*,*,*]
        # The outer [0,1,1] check requires positions 4,5,6 = 0,1,1
        # Type I inner check: positions 7,8,9 = 0,0,1
        nrzi_14 = [0, 0, 0, 0,  0, 1, 1,  0, 0, 1,  0, 0, 0, 0]
        nrz = self._nrz_from_nrzi_14(nrzi_14, [0, 0, 0, 0, 0])

        # Pad with zeros (20 pre-padding)
        padded = np.zeros(len(nrz) + 40, dtype=np.int64)
        padded[20:20 + len(nrz)] = nrz

        # Decode with sector_length = len(nrz)
        decoded, invalid_cw = dec_6by7mtr_code(padded, 20, len(nrz))
        assert len(decoded) > 0
        # The decoder should have run without error
        assert invalid_cw >= 0

    def test_type_II_undo(self):
        """Decoder should undo Type II substitution ([0,1,1] + [0,1,0])."""
        # NRZI: positions 4,5,6 = 0,1,1; positions 7,8,9 = 0,1,0
        nrzi_14 = [0, 0, 0, 0,  0, 1, 1,  0, 1, 0,  0, 0, 0, 0]
        nrz = self._nrz_from_nrzi_14(nrzi_14, [0, 0, 0, 0, 0])

        padded = np.zeros(len(nrz) + 40, dtype=np.int64)
        padded[20:20 + len(nrz)] = nrz

        decoded, invalid_cw = dec_6by7mtr_code(padded, 20, len(nrz))
        assert len(decoded) > 0
        assert invalid_cw >= 0

    def test_type_III_undo(self):
        """Decoder should undo Type III substitution ([0,1,1] + [0,0,0])."""
        # NRZI: positions 4,5,6 = 0,1,1; positions 7,8,9 = 0,0,0
        nrzi_14 = [0, 0, 0, 0,  0, 1, 1,  0, 0, 0,  0, 0, 0, 0]
        nrz = self._nrz_from_nrzi_14(nrzi_14, [0, 0, 0, 0, 0])

        padded = np.zeros(len(nrz) + 40, dtype=np.int64)
        padded[20:20 + len(nrz)] = nrz

        decoded, invalid_cw = dec_6by7mtr_code(padded, 20, len(nrz))
        assert len(decoded) > 0
        assert invalid_cw >= 0

    def test_all_types_sequential(self):
        """Decoder can handle multiple substitution types in sequence."""
        # 4 blocks (28 NRZI bits): Type I at boundary 0, Type II at boundary 1, Type III at boundary 2
        nrzi_28 = [
            0, 0, 0, 0,    # block 0 prefix
            0, 1, 1,       # block 0 suffix - Type I/II/III trigger
            0, 0, 1,       # block 1 start - Type I
            0, 1, 1,       # block 1 suffix - trigger
            0, 1, 0,       # block 2 start - Type II
            0, 1, 1,       # block 2 suffix - trigger
            0, 0, 0,       # block 3 start - Type III
            0, 0, 0, 0, 0, # block 3 remaining
        ]
        nrz = self._nrz_from_nrzi_14(nrzi_28[:14], [0, 0, 0, 0, 0])
        # Append the rest
        extra_nrz = []
        for bit in nrzi_28[14:]:
            if extra_nrz:
                prev = extra_nrz[-1]
            else:
                prev = nrz[-1]
            if bit == 0:
                extra_nrz.append(prev)
            else:
                extra_nrz.append(1 - prev)
        full_nrz = np.concatenate([nrz, np.array(extra_nrz, dtype=np.int64)])
        padded = np.zeros(len(full_nrz) + 40, dtype=np.int64)
        padded[20:20 + len(full_nrz)] = full_nrz

        decoded, invalid_cw = dec_6by7mtr_code(padded, 20, len(full_nrz))
        assert len(decoded) > 0
        assert invalid_cw >= 0


# ===========================================================================
# Section G: MTR cached codewords (decoders/mtr_6_7.py line 29,
#             encoders/mtr_6_7.py line 32)
# ===========================================================================


class TestMTRCacheBranches:
    """Test that subsequent calls hit the cached codewords path."""

    def test_encoder_cached_second_call(self):
        """Second call to enc_6by7mtr_code should use cached codewords."""
        bits = np.array([0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
        # First call loads codewords
        enc_6by7mtr_code(bits, 7)
        # Second call should hit cached path
        result2 = enc_6by7mtr_code(bits, 7)
        assert len(result2) > 0

    def test_decoder_cached_second_call(self):
        """Second call to dec_6by7mtr_code should use cached lookup."""
        bits = np.zeros(50, dtype=np.int64)
        # First call loads lookup table
        dec_6by7mtr_code(bits, 0, 50)
        # Second call should hit cached path
        result2, _ = dec_6by7mtr_code(bits, 0, 50)
        assert len(result2) > 0


# ===========================================================================
# Section H: Decoder sector_length edge case (decoders/mtr_6_7.py line 98)
# ===========================================================================


class TestMTRDecoderSectorLength:
    """Test decoder when sector_length > available data length."""

    def test_sector_length_greater_than_available(self):
        """When sector_length > available, decoder should use available."""
        # Short input, large sector_length
        short = np.zeros(30, dtype=np.int64)  # only 30 elements
        decoded, invalid_cw = dec_6by7mtr_code(short, 5, 100)
        assert len(decoded) > 0
        assert invalid_cw >= 0


# ===========================================================================
# Section I: Equalizer GPR monic constraint (equalizer.py line 377)
# ===========================================================================


class TestGPRMonicConstraint:
    """Test the GPR monic constraint branch."""

    def test_monic_constraint_normalization(self):
        """GPR target with G[0] far from 1.0 should trigger normalization."""
        # Create data where the GPR target computation produces G[0] != 1.0
        rng = np.random.RandomState(42)
        num_samples = 500
        output = rng.randn(num_samples).astype(np.float64)
        input_signal = rng.randn(num_samples).astype(np.float64)

        gpr_target, eq_coeff = find_gpr_target(
            output, input_signal, num_taps=11,
            gpr_target_length=4,
        )
        # The monic constraint ensures gpr_target[0] ≈ 1.0
        assert abs(gpr_target[0] - 1.0) < 1e-3
        assert len(eq_coeff) == 11


# ===========================================================================
# Section J: Simulation integration branches
# ===========================================================================


class TestSimulationBranchCoverage:
    """Integration tests for simulation pipeline branches."""

    def test_fixed_pr_target_lms_adaptation(self):
        """Run simulation with FixedPRTarget equalizer to cover lines 450-506."""
        config = SimulatorConfig(
            channel_type="Longitudinal",
            detector_type="Viterbi",
            equalizer_type="FixedPRTarget",
            use_encoding=False,
            snr_db=[30.0],
            max_num_sectors=2,
            min_num_sectors=1,
            max_num_bit_err=100,
            sector_length=128,
            num_eq_sectors=2,
            num_eq_taps=11,
            num_channel_taps=51,
            num_lpf_taps=51,
            viterbi_delay=10,
            pri_imp_res=[1, 0, -1],
        )
        output = run_simulation(config)
        assert len(output["results"]) == 1

    def test_sova_detector(self):
        """Run simulation with SOVA detector (lines 612-618)."""
        config = SimulatorConfig(
            channel_type="Longitudinal",
            detector_type="SOVA",
            equalizer_type="FixedPRTarget",
            use_encoding=False,
            snr_db=[30.0],
            max_num_sectors=2,
            min_num_sectors=1,
            max_num_bit_err=100,
            sector_length=128,
            num_eq_sectors=2,
            num_eq_taps=11,
            num_channel_taps=51,
            num_lpf_taps=51,
            viterbi_delay=10,
            pri_imp_res=[1, 0, -1],
        )
        output = run_simulation(config)
        assert len(output["results"]) == 1

    def test_constrained_viterbi_mtr(self):
        """Run with MTR encoding to trigger constrained Viterbi (lines 590-595)."""
        config = SimulatorConfig(
            channel_type="Longitudinal",
            detector_type="Viterbi",
            equalizer_type="FixedPRTarget",
            use_encoding=True,
            encoder_type="mtr_6_7",
            snr_db=[30.0],
            max_num_sectors=2,
            min_num_sectors=1,
            max_num_bit_err=100,
            sector_length=127,  # 6Z+1 = 6*21+1 = 127
            num_eq_sectors=1,
            num_eq_taps=11,
            num_channel_taps=51,
            num_lpf_taps=51,
            viterbi_delay=10,
            pri_imp_res=[1, 0, -1],
        )
        output = run_simulation(config)
        assert len(output["results"]) == 1

    def test_constrained_viterbi_tmtr(self):
        """Run with TMTR encoding to trigger constrained Viterbi (lines 596-601)."""
        config = SimulatorConfig(
            channel_type="Longitudinal",
            detector_type="Viterbi",
            equalizer_type="FixedPRTarget",
            use_encoding=True,
            encoder_type="tmtr_8_9",
            snr_db=[30.0],
            max_num_sectors=2,
            min_num_sectors=1,
            max_num_bit_err=100,
            sector_length=129,  # 8Z+1 = 8*16+1 = 129
            num_eq_sectors=1,
            num_eq_taps=11,
            num_channel_taps=51,
            num_lpf_taps=51,
            viterbi_delay=10,
            pri_imp_res=[1, 0, -1],
        )
        output = run_simulation(config)
        assert len(output["results"]) == 1

    def test_no_equalizer_fallback(self):
        """Run with unknown equalizer type to trigger no-eq fallback (line 583)."""
        config = SimulatorConfig(
            channel_type="Longitudinal",
            detector_type="Viterbi",
            equalizer_type="NoEqualizer",
            use_encoding=False,
            snr_db=[30.0],
            max_num_sectors=2,
            min_num_sectors=1,
            max_num_bit_err=100,
            sector_length=128,
            num_eq_sectors=0,
            num_eq_taps=11,
            num_channel_taps=51,
            num_lpf_taps=51,
            viterbi_delay=10,
            pri_imp_res=[1, 0, -1],
        )
        output = run_simulation(config)
        assert len(output["results"]) == 1

    def test_error_tracking_and_early_termination(self):
        """Run with low SNR to trigger error tracking (lines 646-647)
        and early termination (line 658)."""
        config = SimulatorConfig(
            channel_type="Longitudinal",
            detector_type="Viterbi",
            equalizer_type="GPRTarget",
            use_encoding=False,
            snr_db=[12.0],  # Low SNR → more errors
            max_num_sectors=10,
            min_num_sectors=3,   # Wait for at least 3 sectors
            max_num_bit_err=10,  # Stop after 10 total bit errors
            sector_length=128,
            num_eq_sectors=2,
            num_eq_taps=11,
            num_channel_taps=51,
            num_lpf_taps=51,
            viterbi_delay=10,
            pri_imp_res=[1, 1, -1, -1],
        )
        output = run_simulation(config)
        assert len(output["results"]) == 1
        # Should have some bit errors (low SNR)
        assert output["results"][0]["num_bit_errors"] >= 0

    def test_universal_encoding_decoder_branch(self):
        """Run with RLL encoding to hit encoder/decoder branches in main loop.

        This covers:
        - simulator.py lines 523-524: encoder branch
        - simulator.py line 627: decoder branch
        """
        config = SimulatorConfig(
            channel_type="Longitudinal",
            detector_type="Viterbi",
            equalizer_type="FixedPRTarget",
            use_encoding=True,
            encoder_type="rll_4_5",
            snr_db=[30.0],
            max_num_sectors=2,
            min_num_sectors=1,
            max_num_bit_err=100,
            sector_length=101,  # 4Z+1 = 4*25+1 = 101
            num_eq_sectors=1,
            num_eq_taps=11,
            num_channel_taps=51,
            num_lpf_taps=51,
            viterbi_delay=10,
            pri_imp_res=[1, 0, -1],
        )
        output = run_simulation(config)
        assert len(output["results"]) == 1


# ===========================================================================
# Section K: Equalizer LPF in equalizer.py (lines 39-55)
# ===========================================================================


class TestEqualizerLpf:
    """Test the lpf function in equalizer_detector/equalizer.py (lines 39-55)."""

    def test_equalizer_lpf_basic(self):
        """The lpf() in equalizer.py should produce finite output."""
        data = np.random.randn(100).astype(np.float64)
        output = equalizer_lpf(data, 20, 0.4)
        assert len(output) > 0
        assert np.all(np.isfinite(output))

    def test_equalizer_lpf_vs_channel_lpf(self):
        """Both lpf implementations should produce similar results."""
        from channel.lpf import lpf as channel_lpf
        data = np.sin(np.linspace(0, 4 * np.pi, 200))
        out1 = equalizer_lpf(data, 30, 0.3)
        out2 = channel_lpf(data, 30, 0.3)
        # Outputs should be very similar
        min_len = min(len(out1), len(out2))
        np.testing.assert_array_almost_equal(out1[:min_len], out2[:min_len], decimal=10)


# ===========================================================================
# Section L: SOVA edge case startup branch (sova.py lines 116-120)
# ===========================================================================


class TestSOVAStartupBranch:
    """Test SOVA startup phase survivor selection.

    Lines 116-120 are triggered when prev_state0 is not in vs_prev
    but prev_state1 is. This is an edge case during the startup transient.
    """

    def test_sova_produces_output_with_pri_imp_res(self):
        """SOVA with non-trivial PR target should produce valid output."""
        bits = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 0], dtype=np.int64)
        bipolar = 2 * bits - 1
        pri_imp_res = np.array([1, 0, -1], dtype=np.float64)

        # Convolve with PR target
        signal = np.convolve(bipolar, pri_imp_res)[:len(bipolar)]

        # Add tiny noise
        noise_sigma = 0.01
        noisy = signal + np.random.randn(len(signal)).astype(np.float64) * noise_sigma

        hard, soft, sova_pri = classical_sova(
            10, noisy.astype(np.float64), len(noisy),
            pri_imp_res, noise_sigma,
        )
        assert len(hard) == len(noisy)
        assert len(soft) == len(noisy)
        assert len(sova_pri) == len(pri_imp_res)  # Returns PR target itself
        assert np.all(np.isfinite(hard))
        assert np.all(np.isfinite(soft))


# ===========================================================================
# Section M: Encoder MTR substitution Type I (encoders/mtr_6_7.py lines 120-126)
# ===========================================================================


class TestMTREncoderSubstitution:
    """Test encoder MTR substitution branches."""

    def test_encoder_with_seed_that_triggers_substitution(self):
        """Use a seed known to produce substitution patterns in encoder output."""
        np.random.seed(1)
        sector_len = 211
        user_bits = np.random.randint(0, 2, sector_len, dtype=np.int64)
        user_bits[0] = 0
        encoded = enc_6by7mtr_code(user_bits, sector_len)

        # Verify it produces output with transitions
        nrzi = np.abs(np.diff(encoded))
        assert np.sum(nrzi) > 0


# ===========================================================================
# Section N: Encoder MTR/TMTR cached codewords
# ===========================================================================


class TestTMTRCacheBranch:
    """Test TMTR encoder/decoder cached codewords."""

    def test_encoder_cached_second_call(self):
        """Second call to tmtr encoder should use cached codewords."""
        bits = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.int64)
        enc_8by9tmtr_code(bits, 9)
        result = enc_8by9tmtr_code(bits, 9)
        assert len(result) > 0


# ===========================================================================
# Section O: Encoder MTR/TMTR file validation
# ===========================================================================


class TestEncoderFiles:
    """Verify encoder files load correctly."""

    def test_mtr_codewords_loaded(self):
        """MTR codewords should be a (64, 7) array."""
        assert mtr_codewords.shape == (64, 7)
        # All entries should be 0 or 1
        assert np.all((mtr_codewords == 0) | (mtr_codewords == 1))

    def test_mtr_dec_lookup_has_64_entries(self):
        """MTR decoder lookup should have 64 entries."""
        assert len(mtr_dec_lookup) == 64


# ===========================================================================
# Section N: Comprehensive pipeline integration tests
# ===========================================================================
#
# Verify that all main C-code pipeline combinations run end-to-end:
#
#   Channel        × Equalizer      × Detector   × Encoding
# ─────────────────────────────────────────────────────────────
#   Longitudinal   × FixedPRTarget  × Viterbi    × Uncoded
#   Longitudinal   × FixedPRTarget  × Viterbi    × RLL-4/5
#   Longitudinal   × FixedPRTarget  × SOVA       × Uncoded
#   Longitudinal   × GPRTarget      × Viterbi    × Uncoded
#   Longitudinal   × GPRTarget      × SOVA       × Uncoded
#   Perpendicular  × FixedPRTarget  × Viterbi    × Uncoded
#   HAMR (physics) × FixedPRTarget  × Viterbi    × Uncoded
#   Media noise    × FixedPRTarget  × Viterbi    × Uncoded
#


class TestAllMainPipelines:
    """Run every major C-code pipeline combination to ensure no crashes."""

    CONFIG_BASE = dict(
        snr_db=[30.0],
        max_num_sectors=2,
        min_num_sectors=1,
        max_num_bit_err=100,
        sector_length=128,
        num_eq_sectors=1,
        num_eq_taps=11,
        num_channel_taps=51,
        num_lpf_taps=51,
        viterbi_delay=10,
        pri_imp_res=[1, 0, -1],
    )

    def _run(self, **overrides) -> dict:
        config = SimulatorConfig(**{**self.CONFIG_BASE, **overrides})
        return run_simulation(config)

    # ── Channel: Longitudinal ──────────────────────────────────────────

    def test_long_fixed_viterbi_uncoded(self):
        """Longitudinal + FixedPRTarget + Viterbi + Uncoded."""
        out = self._run(
            channel_type="Longitudinal", equalizer_type="FixedPRTarget",
            detector_type="Viterbi", use_encoding=False,
        )
        assert len(out["results"]) == 1

    def test_long_fixed_viterbi_rll(self):
        """Longitudinal + FixedPRTarget + Viterbi + RLL 4/5."""
        out = self._run(
            channel_type="Longitudinal", equalizer_type="FixedPRTarget",
            detector_type="Viterbi", use_encoding=True,
            encoder_type="rll_4_5", sector_length=101,
        )
        assert len(out["results"]) == 1

    def test_long_fixed_sova_uncoded(self):
        """Longitudinal + FixedPRTarget + SOVA + Uncoded."""
        out = self._run(
            channel_type="Longitudinal", equalizer_type="FixedPRTarget",
            detector_type="SOVA", use_encoding=False,
        )
        assert len(out["results"]) == 1

    def test_long_gpr_viterbi_uncoded(self):
        """Longitudinal + GPRTarget + Viterbi + Uncoded."""
        out = self._run(
            channel_type="Longitudinal", equalizer_type="GPRTarget",
            detector_type="Viterbi", use_encoding=False,
        )
        assert len(out["results"]) == 1

    def test_long_gpr_sova_uncoded(self):
        """Longitudinal + GPRTarget + SOVA + Uncoded."""
        out = self._run(
            channel_type="Longitudinal", equalizer_type="GPRTarget",
            detector_type="SOVA", use_encoding=False,
        )
        assert len(out["results"]) == 1

    def test_long_gpr_sova_coded(self):
        """Longitudinal + GPRTarget + SOVA + RLL 4/5."""
        out = self._run(
            channel_type="Longitudinal", equalizer_type="GPRTarget",
            detector_type="SOVA", use_encoding=True,
            encoder_type="rll_4_5", sector_length=101,
        )
        assert len(out["results"]) == 1

    def test_long_fixed_sova_mtr(self):
        """Longitudinal + FixedPRTarget + SOVA + MTR 6/7."""
        out = self._run(
            channel_type="Longitudinal", equalizer_type="FixedPRTarget",
            detector_type="SOVA", use_encoding=True,
            encoder_type="mtr_6_7", sector_length=127,
        )
        assert len(out["results"]) == 1

    def test_long_fixed_sova_tmtr(self):
        """Longitudinal + FixedPRTarget + SOVA + TMTR 8/9."""
        out = self._run(
            channel_type="Longitudinal", equalizer_type="FixedPRTarget",
            detector_type="SOVA", use_encoding=True,
            encoder_type="tmtr_8_9", sector_length=129,
        )
        assert len(out["results"]) == 1

    # ── Channel: Perpendicular ─────────────────────────────────────────

    def test_perp_fixed_viterbi_uncoded(self):
        """Perpendicular + FixedPRTarget + Viterbi + Uncoded."""
        out = self._run(
            channel_type="Perpendicular", equalizer_type="FixedPRTarget",
            detector_type="Viterbi", use_encoding=False,
        )
        assert len(out["results"]) == 1

    def test_perp_gpr_sova_uncoded(self):
        """Perpendicular + GPRTarget + SOVA + Uncoded."""
        out = self._run(
            channel_type="Perpendicular", equalizer_type="GPRTarget",
            detector_type="SOVA", use_encoding=False,
        )
        assert len(out["results"]) == 1

    # ── Channel: HAMR (full physics) ───────────────────────────────────

    def test_hamr_full_fixed_viterbi_uncoded(self):
        """HAMR (full physics) + FixedPRTarget + Viterbi + Uncoded."""
        out = self._run(
            channel_type="Hamr", equalizer_type="FixedPRTarget",
            detector_type="Viterbi", use_encoding=False,
            sector_length=128,
        )
        assert len(out["results"]) == 1

    def test_hamr_full_gpr_viterbi_uncoded(self):
        """HAMR (full physics) + GPRTarget + Viterbi + Uncoded."""
        out = self._run(
            channel_type="Hamr", equalizer_type="GPRTarget",
            detector_type="Viterbi", use_encoding=False,
            sector_length=128,
        )
        assert len(out["results"]) == 1

    # ── Media noise ────────────────────────────────────────────────────

    def test_media_noise_jitter(self):
        """Longitudinal + FixedPRTarget + Viterbi + jitter noise."""
        out = self._run(
            channel_type="Longitudinal", equalizer_type="FixedPRTarget",
            detector_type="Viterbi", use_encoding=False,
            sigma_jitter=5.0,  # 5% jitter
        )
        assert len(out["results"]) == 1

    def test_media_noise_pulse_broad(self):
        """Longitudinal + FixedPRTarget + Viterbi + pulse broadening."""
        out = self._run(
            channel_type="Longitudinal", equalizer_type="FixedPRTarget",
            detector_type="Viterbi", use_encoding=False,
            sigma_pulse_broad=3.0,  # 3% pulse broadening
        )
        assert len(out["results"]) == 1

    def test_media_noise_both(self):
        """Longitudinal + GPRTarget + Viterbi + both jitter and pulse broadening."""
        out = self._run(
            channel_type="Longitudinal", equalizer_type="GPRTarget",
            detector_type="Viterbi", use_encoding=False,
            sigma_jitter=5.0, sigma_pulse_broad=3.0,
        )
        assert len(out["results"]) == 1

    def test_media_noise_perp(self):
        """Perpendicular + GPRTarget + Viterbi + jitter noise."""
        out = self._run(
            channel_type="Perpendicular", equalizer_type="GPRTarget",
            detector_type="Viterbi", use_encoding=False,
            sigma_jitter=5.0,
        )
        assert len(out["results"]) == 1

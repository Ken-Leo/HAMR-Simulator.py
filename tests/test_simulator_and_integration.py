"""Tests for simulator.py and FullHamrChannel integration."""

import numpy as np
import pytest

from channel.channel import FullHamrChannel
from channel.fir import causal_fir
from channel.lpf import lpf
from channel.math_utils import uniform_random, gaussian_random
from equalizer_detector.equalizer import find_gpr_target, apply_equalizer
from equalizer_detector.viterbi import classical_viterbi
from equalizer_detector.constrained_detectors import (
    classical_viterbi_for_temp_mod_with_perm,
)
from simulator import (
    SimulatorConfig,
    _compute_code_params,
    _encoder_fn,
    _decoder_fn,
    _bipolar,
    _causal_fir_simple,
)


class TestSimulatorConfig:
    def test_default_config(self):
        config = SimulatorConfig()
        assert config.channel_type == "Hamr"
        assert config.detector_type == "Viterbi"
        assert config.equalizer_type == "GPRTarget"
        assert config.use_encoding is False
        assert config.snr_db == [21.0]
        assert config.sector_length == 4096

    def test_custom_config(self):
        config = SimulatorConfig(
            channel_type="Longitudinal",
            detector_type="SOVA",
            use_encoding=True,
            encoder_type="rll_4_5",
            snr_db=[20.0, 25.0, 30.0],
        )
        assert config.channel_type == "Longitudinal"
        assert config.detector_type == "SOVA"
        assert config.use_encoding is True
        assert config.encoder_type == "rll_4_5"
        assert len(config.snr_db) == 3


class TestComputeCodeParams:
    def test_no_encoding(self):
        config = SimulatorConfig(use_encoding=False, sector_length=512)
        user_len, coded_len, rate = _compute_code_params(config)
        assert user_len == 512
        assert coded_len == 512
        assert rate == 1.0

    def test_rll_encoding(self):
        config = SimulatorConfig(
            use_encoding=True, encoder_type="rll_4_5", sector_length=512
        )
        user_len, coded_len, rate = _compute_code_params(config)
        assert rate == pytest.approx(4.0 / 5.0)
        assert coded_len == 512

    def test_mtr_encoding(self):
        config = SimulatorConfig(
            use_encoding=True, encoder_type="mtr_6_7", sector_length=512
        )
        user_len, coded_len, rate = _compute_code_params(config)
        assert rate == pytest.approx(6.0 / 7.0)

    def test_tmtr_encoding(self):
        config = SimulatorConfig(
            use_encoding=True, encoder_type="tmtr_8_9", sector_length=512
        )
        user_len, coded_len, rate = _compute_code_params(config)
        assert rate == pytest.approx(8.0 / 9.0)


class TestHelperFunctions:
    def test_bipolar_mapping(self):
        bits = np.array([0, 1, 0, 1])
        result = _bipolar(bits)
        expected = np.array([-1.0, 1.0, -1.0, 1.0])
        np.testing.assert_array_equal(result, expected)

    def test_causal_fir_simple(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        h = np.array([1.0, 1.0])
        result = _causal_fir_simple(data, h)
        assert len(result) == len(data)
        # First element: h[0]*data[0] = 1*1 = 1
        assert result[0] == pytest.approx(1.0)
        # Second element: h[0]*data[1] + h[1]*data[0] = 2+1 = 3
        assert result[1] == pytest.approx(3.0)

    def test_encoder_fn_selection(self):
        config = SimulatorConfig(use_encoding=True, encoder_type="rll_4_5")
        fn = _encoder_fn(config)
        assert fn is not None

        config = SimulatorConfig(use_encoding=False)
        fn = _encoder_fn(config)
        assert fn is None

    def test_decoder_fn_selection(self):
        config = SimulatorConfig(use_encoding=True, encoder_type="mtr_6_7")
        fn = _decoder_fn(config)
        assert fn is not None

        config = SimulatorConfig(use_encoding=False)
        fn = _decoder_fn(config)
        assert fn is None


class TestFullHamrChannel:
    def test_initialization(self):
        config = SimulatorConfig(channel_type="Hamr")
        ch = FullHamrChannel(config, nd=2.5, osr=10)
        assert ch.pw50 > 0
        assert ch.over_sampled_bit_length > 0
        assert ch._norm_factor > 0

    def test_pw50_reasonable(self):
        """PW50 for HAMR should be in the range 50-200 nm."""
        config = SimulatorConfig(channel_type="Hamr")
        ch = FullHamrChannel(config, nd=2.5, osr=10)
        assert 50 < ch.pw50 < 200

    def test_channel_call_produces_output(self):
        config = SimulatorConfig(channel_type="Hamr", sector_length=128)
        ch = FullHamrChannel(config, nd=2.5, osr=10)

        bits = np.array([0, 1] * 64, dtype=np.int64)
        padded = np.zeros(128 + 40, dtype=np.int64)
        padded[20 : 20 + 128] = bits

        output = ch(padded, sector_index=0, disable_media_noise=True)
        assert len(output) == len(padded) * 10
        assert np.any(output != 0)

    def test_normalization_factor_applied(self):
        """Channel output should be normalized to ~[-1, 1]."""
        config = SimulatorConfig(channel_type="Hamr", sector_length=128)
        ch = FullHamrChannel(config, nd=2.5, osr=10)

        bits = np.array([0] * 64 + [1] * 64, dtype=np.int64)
        padded = np.zeros(128 + 40, dtype=np.int64)
        padded[20 : 20 + 128] = bits

        output = ch(padded, sector_index=0, disable_media_noise=True)
        assert np.max(np.abs(output)) < 2.0


class TestTemperatureModulatedViterbi:
    def test_output_length(self):
        pri = np.array([1, 1, -1, -1], dtype=np.float64)
        sector_length = 200
        sub_sector_length = 50
        num_sub_sectors = 3
        num_interleaved = 5
        pre_padding = 10

        mse = np.zeros(num_sub_sectors)
        data = np.random.randn(sector_length)
        detected = classical_viterbi_for_temp_mod_with_perm(
            delay=20,
            equalized_channel_output=data,
            sector_length=sector_length,
            pri_imp_res=pri,
            mse=mse,
            num_bits_rep_perm_period=num_sub_sectors,
            sub_sector_length=sub_sector_length,
            num_interleaved_bits=num_interleaved,
            pre_padding_length=pre_padding,
        )
        assert len(detected) == sector_length

    def test_clean_signal_low_errors(self):
        """Clean PR target signal should have errors only at boundaries."""
        pri = np.array([1, 1, -1, -1], dtype=np.float64)
        sub_sector_length = 100
        num_sub_sectors = 3
        num_interleaved = 5
        pre_padding = 10
        sector_length = (sub_sector_length + num_interleaved) * num_sub_sectors - num_interleaved + 2 * pre_padding

        np.random.seed(42)
        bits = np.random.randint(0, 2, sector_length)
        bipolar = 2.0 * bits.astype(np.float64) - 1.0
        pr_output = causal_fir(bipolar, pri)

        mse = np.zeros(num_sub_sectors)
        detected = classical_viterbi_for_temp_mod_with_perm(
            delay=20,
            equalized_channel_output=pr_output,
            sector_length=sector_length,
            pri_imp_res=pri,
            mse=mse,
            num_bits_rep_perm_period=num_sub_sectors,
            sub_sector_length=sub_sector_length,
            num_interleaved_bits=num_interleaved,
            pre_padding_length=pre_padding,
        )

        # Errors should only be in pre-padding and interleaved regions
        errors = np.where(detected != bits)[0]
        for e in errors:
            rel = e - pre_padding
            if 0 <= rel < sub_sector_length:
                pytest.fail(f"Error in data region at position {e}")

    def test_mse_values_finite(self):
        """MSE values should be finite and positive."""
        pri = np.array([1, 1, -1, -1], dtype=np.float64)
        sector_length = 200
        sub_sector_length = 50
        num_sub_sectors = 3

        mse = np.zeros(num_sub_sectors)
        data = np.random.randn(sector_length) * 0.5
        classical_viterbi_for_temp_mod_with_perm(
            delay=20,
            equalized_channel_output=data,
            sector_length=sector_length,
            pri_imp_res=pri,
            mse=mse,
            num_bits_rep_perm_period=num_sub_sectors,
            sub_sector_length=sub_sector_length,
            num_interleaved_bits=5,
            pre_padding_length=10,
        )
        assert np.all(np.isfinite(mse))
        assert np.all(mse >= 0)


class TestGprEqualizerPipeline:
    def test_gpr_convergence_with_enough_sectors(self):
        """GPR equalizer should converge with sufficient adaptation sectors."""
        from channel.channel import longitudinal_channel

        lcg_bits = uniform_random(-500)
        lcg_noise = uniform_random(-600)
        sector_length = 512
        padded_length = sector_length + 40
        pri = np.array([1, 1, -1, -1], dtype=np.float64)

        concat_input, concat_output = [], []
        for s in range(10):
            bits = np.array(
                [int(lcg_bits.random() > 0.5) for _ in range(sector_length + 4)],
                dtype=np.int64,
            )
            padded = np.zeros(padded_length, dtype=np.int64)
            padded[20 : 20 + sector_length] = bits[:sector_length]

            ch_output = longitudinal_channel(padded, nd=2.5, num_taps=201, osr=10)
            noise = (
                np.array([gaussian_random(lcg_noise) for _ in range(len(ch_output))])
                * 0.001
            )
            ch_output[200 : sector_length * 10] += noise[200 : sector_length * 10]

            lpf_out = lpf(ch_output, 200, 1.0 / 10)
            ds_out = lpf_out[::10][:padded_length]

            concat_input.append(_bipolar(padded[20 : 20 + sector_length]))
            concat_output.append(ds_out[20 : 20 + sector_length])

        full_input = np.concatenate(concat_input)
        full_output = np.concatenate(concat_output)
        gpr_target, eq_coeff = find_gpr_target(
            full_output, full_input, 21, gpr_target_length=4
        )

        # GPR target should start with 1.0 (monic constraint)
        assert gpr_target[0] == pytest.approx(1.0, abs=1e-4)

        # Test with new sector
        bits = np.array(
            [int(lcg_bits.random() > 0.5) for _ in range(sector_length + 4)],
            dtype=np.int64,
        )
        padded = np.zeros(padded_length, dtype=np.int64)
        padded[20 : 20 + sector_length] = bits[:sector_length]

        ch_output = longitudinal_channel(padded, nd=2.5, num_taps=201, osr=10)
        noise_sigma = 0.2236  # 30dB
        noise = (
            np.array([gaussian_random(lcg_noise) for _ in range(len(ch_output))])
            * noise_sigma
        )
        ch_output[200 : sector_length * 10] += noise[200 : sector_length * 10]

        lpf_out = lpf(ch_output, 200, 1.0 / 10)
        ds_out = lpf_out[::10][:padded_length]
        equalized = apply_equalizer(ds_out, eq_coeff, 21)
        detected, _ = classical_viterbi(20, equalized, padded_length, gpr_target)

        errors = np.sum(detected[20 : 20 + sector_length] != bits[:sector_length])
        ber = errors / sector_length
        assert ber < 0.01, f"BER {ber} too high after GPR equalization"


class TestRunSimulation:
    def test_run_simulation_longitudinal(self):
        """Test run_simulation with Longitudinal channel (fast)."""
        from simulator import run_simulation

        config = SimulatorConfig(
            channel_type="Longitudinal",
            detector_type="Viterbi",
            equalizer_type="GPRTarget",
            use_encoding=False,
            snr_db=[30.0],
            max_num_sectors=2,
            min_num_sectors=1,
            max_num_bit_err=100,
            sector_length=256,
            num_eq_sectors=5,
            num_eq_taps=21,
            viterbi_delay=20,
            pri_imp_res=[1, 1, -1, -1],
        )

        results = run_simulation(config)
        assert "snr_values" in results
        assert "ber_per_snr" in results
        assert "ser_per_snr" in results
        assert len(results["snr_values"]) == 1
        assert results["snr_values"][0] == 30.0

    def test_run_simulation_multiple_snr(self):
        """Test run_simulation with multiple SNR points."""
        from simulator import run_simulation

        config = SimulatorConfig(
            channel_type="Longitudinal",
            detector_type="Viterbi",
            equalizer_type="GPRTarget",
            use_encoding=False,
            snr_db=[25.0, 30.0],
            max_num_sectors=2,
            min_num_sectors=1,
            sector_length=256,
            num_eq_sectors=5,
            num_eq_taps=21,
            viterbi_delay=20,
            pri_imp_res=[1, 1, -1, -1],
        )

        results = run_simulation(config)
        assert len(results["snr_values"]) == 2
        assert results["snr_values"] == [25.0, 30.0]

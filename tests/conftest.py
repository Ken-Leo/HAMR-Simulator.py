"""Shared test fixtures for the HAMR Simulator test suite."""

from __future__ import annotations

import numpy as np
import pytest

from encoders.rll_4_5 import Codewords as rll_codewords
from encoders.mtr_6_7 import Codewords as mtr_codewords
from encoders.tmtr_8_9 import Codewords as tmtr_codewords
from decoders.rll_4_5 import Codewords as rll_dec_codewords
from decoders.mtr_6_7 import Codewords as mtr_dec_codewords
from decoders.tmtr_8_9 import Codewords as tmtr_dec_codewords


@pytest.fixture
def rng_seed():
    """Fixed seed for reproducible tests."""
    return 42


@pytest.fixture
def rll_codeword_table():
    """Return the 4/5 RLL codeword table (16 x 5)."""
    return rll_codewords


@pytest.fixture
def mtr_codeword_table():
    """Return the 6/7 MTR codeword table (64 x 7)."""
    return mtr_codewords


@pytest.fixture
def tmtr_codeword_table():
    """Return the 8/9 TMTR codeword table (256 x 9)."""
    return tmtr_codewords


@pytest.fixture
def small_user_bits():
    """Generate small test user bits."""
    return np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=np.int64)


@pytest.fixture
def large_user_bits():
    """Generate a larger test bit sequence."""
    return np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1] * 256, dtype=np.int64)


@pytest.fixture
def pr_target():
    """EPR4 PR target: [1, 1, -1, -1]."""
    return np.array([1.0, 1.0, -1.0, -1.0], dtype=np.float64)


@pytest.fixture
def noiseless_equalized_signal(pr_target):
    """Create a noiseless equalized signal for a simple bit pattern."""
    bits = np.array([0, 1, 0, 1, 1, 0, 1, 0, 0, 1], dtype=np.int64)
    bipolar = 2.0 * bits.astype(np.float64) - 1.0

    # Compute expected channel output for PR target
    output = np.zeros(len(bipolar) + len(pr_target) - 1, dtype=np.float64)
    for i in range(len(bipolar)):
        for j in range(len(pr_target)):
            output[i + j] += pr_target[j] * bipolar[i]

    # Trim to sector length
    return output[:len(bits)]


@pytest.fixture
def default_sim_config():
    """Return a minimal SimulatorConfig for testing."""
    from simulator import SimulatorConfig
    return SimulatorConfig(
        snr_db=[30.0],  # High SNR for testing
        max_num_sectors=2,
        min_num_sectors=1,
        max_num_bit_err=10,
        use_encoding=True,
        encoder_type="rll_4_5",
        detector_type="Viterbi",
        equalizer_type="FixedPRTarget",
        pri_imp_res=[1, 1, -1, -1],
        channel_type="Longitudinal",
    )

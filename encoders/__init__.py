"""Encoders module for HAMR Simulator.

Provides run-length limited (RLL), Maximum Transition Run (MTR),
and Time-Varying Maximum Transition Run (TMTR) encoding functions,
plus permutation-based tribit minimization.
"""

from .mtr_6_7 import enc_6by7mtr_code
from .permutation import count_possible_error_events, permute
from .rll_4_5 import enc_4by5rll_code
from .tmtr_8_9 import enc_8by9tmtr_code

__all__ = [
    "enc_4by5rll_code",
    "enc_6by7mtr_code",
    "enc_8by9tmtr_code",
    "count_possible_error_events",
    "permute",
]

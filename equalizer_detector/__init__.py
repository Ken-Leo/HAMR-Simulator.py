"""Equalizer and detector module for the HAMR simulator.

Translates FIR filters, adaptive equalizer, Viterbi detector,
Soft-Output Viterbi Algorithm (SOVA), and code-constrained detectors
from the C reference implementation.
"""

from .equalizer import non_causal_fir, causal_fir, lpf, adapt_equalizer
from .viterbi import classical_viterbi
from .sova import classical_sova

# Constrained detectors are imported when available (stubbed for now)
try:
    from .constrained_detectors import (
        viterbi_6by7mtr_code,
        viterbi_8by9tmtr_code,
        sova_8by9mtr_code,
        classical_viterbi_for_temp_mod_with_perm,
    )

    __all__ = [
        "non_causal_fir",
        "causal_fir",
        "lpf",
        "adapt_equalizer",
        "classical_viterbi",
        "classical_sova",
        "viterbi_6by7mtr_code",
        "viterbi_8by9tmtr_code",
        "sova_8by9mtr_code",
        "classical_viterbi_for_temp_mod_with_perm",
    ]
except ImportError:
    __all__ = [
        "non_causal_fir",
        "causal_fir",
        "lpf",
        "adapt_equalizer",
        "classical_viterbi",
        "classical_sova",
    ]

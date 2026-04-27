"""Detector module for the HAMR simulator.

Re-exports Viterbi and SOVA detectors so that consumers can import
from a single ``detector`` submodule.
"""

from __future__ import annotations

from .sova import classical_sova
from .viterbi import classical_viterbi

__all__ = [
    "classical_viterbi",
    "classical_sova",
]

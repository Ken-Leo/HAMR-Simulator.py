"""Code-constrained detectors for MTR and TMTR codes.

Translates constrained Viterbi and SOVA detectors from
CustomDetectors.c. These are more complex and build on the
classical detectors by enforcing code constraints on the trellis.

TODO: Implement full constrained detector logic from C reference.
"""
from __future__ import annotations


def viterbi_6by7mtr_code(*args, **kwargs):
    """Placeholder for 6/7 MTR code-constrained Viterbi detector.

    TODO: Full implementation from CustomDetectors.c
    """
    raise NotImplementedError("6/7 MTR constrained Viterbi detector not yet implemented")


def viterbi_8by9tmtr_code(*args, **kwargs):
    """Placeholder for 8/9 TMTR code-constrained Viterbi detector.

    TODO: Full implementation from CustomDetectors.c
    """
    raise NotImplementedError("8/9 TMTR constrained Viterbi detector not yet implemented")


def sova_8by9mtr_code(*args, **kwargs):
    """Placeholder for 8/9 TMTR code-constrained SOVA detector.

    TODO: Full implementation from CustomDetectors.c
    """
    raise NotImplementedError("8/9 TMTR constrained SOVA detector not yet implemented")


def classical_viterbi_for_temp_mod_with_perm(*args, **kwargs):
    """Placeholder for Viterbi with temperature modulation and permutation tracking.

    TODO: Full implementation from CustomDetectors.c
    """
    raise NotImplementedError("Temperature-modulated Viterbi not yet implemented")

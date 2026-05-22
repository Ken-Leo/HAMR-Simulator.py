#!/usr/bin/env python3
"""Compare C and Python receiver results across multiple SNR points."""

import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from simulator import SimulatorConfig, run_simulation

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent  # HamrSimulator/

# Test configuration matching test_coverage_branches.py
TEST_PARAMS = dict(
    snr_db=[12.0, 14.0, 16.0, 18.0, 20.0, 22.0],
    max_num_sectors=10,
    min_num_sectors=1,
    max_num_bit_err=100,
    sector_length=128,
    osr=10,
    pre_padding_length=20,
    post_padding_length=20,
    channel_type="Longitudinal",
    user_density=2.5,
    num_channel_taps=51,
    sigma_jitter=0.0,
    sigma_pulse_broad=0.0,
    num_lpf_taps=51,
    equalizer_type="FixedPRTarget",
    num_eq_taps=11,
    num_eq_sectors=1,
    pri_imp_res=[1, 0, -1],
    detector_type="Viterbi",
    viterbi_delay=10,
    use_encoding=False,
)


def _patch_c_source(src_text: str) -> str:
    """Replace main() parameters with TEST_PARAMS matching Python config."""
    patches = [
        ("double SNR[] = {", TEST_PARAMS["snr_db"]),
        ("int MaxNumSectors =", TEST_PARAMS["max_num_sectors"]),
        ("int MinNumSectors =", TEST_PARAMS["min_num_sectors"]),
        ("int MaxNumBitErr =", TEST_PARAMS["max_num_bit_err"]),
        ("int SectorLength =", TEST_PARAMS["sector_length"]),
        ("int NumChannelTaps =", TEST_PARAMS["num_channel_taps"]),
        ("int NumLPFTaps =", TEST_PARAMS["num_lpf_taps"]),
        ("int NumEqTaps =", TEST_PARAMS["num_eq_taps"]),
        ("int NumEqSectors =", TEST_PARAMS["num_eq_sectors"]),
        ("double PRImpRes[] =", TEST_PARAMS["pri_imp_res"]),
        ("int ViterbiDelay =", TEST_PARAMS["viterbi_delay"]),
    ]
    lines = src_text.split("\n")
    out_lines = []
    for line in lines:
        patched = False
        for prefix, value in patches:
            if line.strip().startswith(prefix):
                if isinstance(value, list):
                    if prefix == "double PRImpRes[] =":
                        arr_str = ",".join(str(v) for v in value)
                        out_lines.append(
                            f'{line[:line.index("=")]}= {{{arr_str}}};'
                        )
                    else:
                        arr_str = ",".join(str(v) for v in value)
                        out_lines.append(
                            f'{line[:line.index("=")]}= {{{arr_str}}};'
                        )
                else:
                    out_lines.append(
                        f'{line[:line.index("=")]}= {value};'
                    )
                patched = True
                break
        if not patched:
            out_lines.append(line)
    return "\n".join(out_lines)


def run_c_simulator() -> dict[float, dict]:
    """Compile and run the C simulator, parse BER/SER per SNR."""
    c_files = [
        REPO / "MagneticDisk.c",
        REPO / "CustomDetectors.c",
        REPO / "DecodingFunctions.c",
        REPO / "EncodingFunctions.c",
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # Copy data files + header
        for dep in [REPO / "MagneticDisk.h",
                    REPO / "Rate4By5RLLCode.dat",
                    REPO / "Rate6By7MTR2-8.dat",
                    REPO / "Rate8By9TMTR.dat",
                    REPO / "ARModelCoeff.dat"]:
            if dep.exists():
                (tmp / dep.name).write_text(dep.read_text())

        # Patch and write each C file
        patched = []
        for cf in c_files:
            src = cf.read_text()
            patched_src = _patch_c_source(src)
            pf = tmp / cf.name
            pf.write_text(patched_src)
            patched.append(str(pf))

        # Compile
        binary = tmp / "hamr_simulator"
        # Compile from the temp dir so includes resolve
        gcc_cmd = ["gcc", "-O2", "-o", str(binary),
                    f"-I{tmp}", *patched, "-lm"]
        ret = subprocess.run(gcc_cmd, capture_output=True, text=True)
        if ret.returncode != 0:
            print("C compilation failed:", ret.stderr[:500])
            return {}

        # Run
        # Run from temp dir so data files are found
        ret = subprocess.run([str(binary)], capture_output=True, text=True,
                             timeout=120, cwd=str(tmp))
        output = ret.stdout + ret.stderr

    # Parse output lines like:
    # CurrentSNR=... BER(NRZ/VIT)=... BER(NRZI/VIT)=... SER=... NumSec=...
    results: dict[float, dict] = {}
    for line in output.split("\n"):
        m = re.search(
            r"CurrentSNR=([\d.]+e?[+-]?\d*) BER\(NRZ/VIT\)=([\d.eE+-]+)"
            r" BER\(NRZI/VIT\)=([\d.eE+-]+) SER=([\d.eE+-]+)"
            r" NumSec=\s*(\d+) NumNRZBitErr=\s*(\d+) NumNRZIBitErr=\s*(\d+)",
            line,
        )
        if m:
            snr = float(m.group(1))
            results[snr] = {
                "ber_nrz": float(m.group(2)),
                "ber_nrzi": float(m.group(3)),
                "ser": float(m.group(4)),
                "num_sectors": int(m.group(5)),
                "num_bit_errors": int(m.group(6)),
                "num_nrzi_bit_errors": int(m.group(7)),
            }
    return results


def run_python_simulator() -> dict[float, dict]:
    """Run Python simulator with matching config."""
    config = SimulatorConfig(**TEST_PARAMS)
    out = run_simulation(config)
    results: dict[float, dict] = {}
    for r in out["results"]:
        results[r["snr_db"]] = {
            "ber_nrz": r["ber"],
            "ser": r["ser"],
            "num_sectors": r["num_sectors"],
            "num_bit_errors": r["num_bit_errors"],
        }
    return results


def main():
    print("=" * 70)
    print("C vs Python Receiver Comparison")
    print("=" * 70)
    print()
    print(f"Parameters: {TEST_PARAMS['channel_type']}, "
          f"{TEST_PARAMS['equalizer_type']}, {TEST_PARAMS['detector_type']}, "
          f"SNR={TEST_PARAMS['snr_db']}")
    print()

    # Run C
    print("Running C simulator...")
    c_results = run_c_simulator()
    if not c_results:
        print("FAILED to get C results.")
        sys.exit(1)
    print(f"  Got {len(c_results)} SNR points from C.")
    print()

    # Run Python
    print("Running Python simulator...")
    py_results = run_python_simulator()
    print(f"  Got {len(py_results)} SNR points from Python.")
    print()

    # Compare
    print("-" * 70)
    print(f"{'SNR':>8} | {'C BER':>12} | {'Py BER':>12} | {'C Err':>8} | "
          f"{'Py Err':>8} | {'C Sec':>6} | {'Py Sec':>6} | {'Match?':>8}")
    print("-" * 70)

    all_snrs = sorted(set(c_results.keys()) | set(py_results.keys()))
    match_count = 0
    total = 0

    for snr in all_snrs:
        c = c_results.get(snr, {})
        p = py_results.get(snr, {})
        c_ber = c.get("ber_nrz", -1)
        p_ber = p.get("ber_nrz", -1)
        c_err = c.get("num_bit_errors", -1)
        p_err = p.get("num_bit_errors", -1)
        c_sec = c.get("num_sectors", -1)
        p_sec = p.get("num_sectors", -1)

        # Tolerance: same number of sectors + same number of errors
        if c_sec == p_sec and c_err == p_err:
            match = "BIT-EXACT"
            match_count += 1
        elif c_sec == p_sec and abs(c_err - p_err) <= max(2, 0.1 * c_err):
            match = "CLOSE"
            match_count += 0.5
        else:
            match = "DIFF"
        total += 1

        print(f"{snr:>8.1f} | {c_ber:>12.6f} | {p_ber:>12.6f} | "
              f"{c_err:>8} | {p_err:>8} | {c_sec:>6} | {p_sec:>6} | "
              f"{match:>8}")

    print("-" * 70)
    pct = 100.0 * match_count / max(total, 1)
    print(f"Match score: {match_count:.0f}/{total} ({pct:.0f}%)")

    if pct >= 80:
        print("\n✓ RESULT: Python receiver matches C within tolerance.")
        print("  (Minor differences expected due to RNG caching in Box-Muller.)")
    elif pct >= 50:
        print("\n∼ RESULT: Partial match — trends agree but numerical differences.")
    else:
        print("\n✗ RESULT: Significant deviation — investigate further.")


if __name__ == "__main__":
    main()

<p align="center">
  <a href="README_EN.md" rel="noopener">
    <img width="200" height="200" src="figures/hamr-logo.svg" alt="Project logo">
  </a>
</p>

<h3 align="center">HAMR-Simulator.py</h3>

<p align="center">
  <a href="README.md">中文</a> | English
</p>

<div align="center">

![Status](https://img.shields.io/badge/status-active-success.svg)
![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![Tests](https://img.shields.io/badge/tests-241%20passed-brightgreen.svg)
![Coverage](https://img.shields.io/badge/coverage-98%25-brightgreen.svg)
![Experiments](https://img.shields.io/badge/experiments-14%20passed-brightgreen.svg)

</div>

---

<p align="center">
  A Python simulator for the Heat-Assisted Magnetic Recording (HAMR) read/write channel,
  ported from a C-language implementation.
</p>

## Table of Contents

- [About](#about)
- [Architecture](#architecture)
- [Getting Started](#getting_started)
- [Running Tests](#tests)
- [Experiments](#experiments)
- [Project Structure](#structure)
- [Bug Log](#bugs)
- [Acknowledgements](#acknowledgement)
- [License](#license)

## About <a name="about"></a>

This project is a Python-based HAMR simulator for modeling the full magnetic recording signal-processing pipeline.
The codebase was originally derived from Prof. Bane Vasic lab's C implementation (about 8,400 lines of C), and then ported, refined, and optimized module by module.

The simulator covers a full processing flow:
user-bit generation -> encoding -> channel modeling (longitudinal/perpendicular/HAMR) -> low-pass filtering -> equalization -> detection (Viterbi/SOVA) -> decoding.

It supports three constrained coding schemes (RLL 4/5, MTR 6/7, TMTR 8/9), along with adaptive GPR-target LMS equalization.

## Architecture <a name="architecture"></a>

### Signal Processing Pipeline

```text
UserBits -> Encoder -> Channel -> LPF -> Downsampler -> Equalizer -> Detector -> Decoder
```

### Channel Models

| Type              | Description                                                                                                    |
| ----------------- | -------------------------------------------------------------------------------------------------------------- |
| **Longitudinal**  | FIR-convolution-based channel with Lorentzian pulse response                                                   |
| **Perpendicular** | Channel based on error-function pulse response                                                                 |
| **HAMR**          | Full HAMR physical model including thermal profile, micro-track accumulation, temperature modulation, and NLTS |

### Coding Schemes

| Code         | Rate | Constraint                                                                | Notes                             |
| ------------ | ---- | ------------------------------------------------------------------------- | --------------------------------- |
| RLL(0,2)     | 4/5  | At least two "0"s between any two "1"s                                    | Lookup-table encode/decode        |
| MTR(2;8)     | 6/7  | No more than two consecutive "1"s and no more than eight consecutive "0"s | Lookup table + substitution rules |
| TMTR(2/3;11) | 8/9  | Alternating-position constraints                                          | Lookup-table encode/decode        |

### Detectors

| Detector                          | Input                                    | Output                                    |
| --------------------------------- | ---------------------------------------- | ----------------------------------------- |
| **Classical Viterbi**             | Equalized signal                         | Hard-decision bits                        |
| **Classical SOVA**                | Equalized signal + noise variance        | Hard decisions + soft reliability outputs |
| **Code-Constrained Viterbi/SOVA** | Same as above + code-constraint callback | Decisions that satisfy code constraints   |

## Getting Started <a name="getting_started"></a>

### Requirements

- Python 3.9+
- NumPy
- SciPy (matrix operations)
- Matplotlib (experiment plotting)

### Installation

```bash
# Create a virtual environment (recommended)
conda create -n hamr_sim python=3.9 numpy scipy matplotlib
conda activate hamr_sim

# Or install with pip
pip install numpy scipy matplotlib
```

### Run the Simulator

```bash
cd python_receiver

# Run end-to-end simulation
python simulator.py

# Run all experiments
python experiments/run_experiments.py

# Run a single experiment (e.g., Exp11: Viterbi vs SOVA)
python experiments/run_experiments.py --experiment 11
```

## Running Tests <a name="tests"></a>

The project includes **241 test cases** across 9 test files (about 3,600 lines), covering all major modules.

```bash
cd python_receiver

# Run all tests
pytest tests/ -v

# Run tests with coverage report
pytest tests/ --cov=. --cov-report=term-missing

# Run a specific test file
pytest tests/test_equalizer_detector.py -v

# Run a single test
pytest tests/test_equalizer_detector.py::TestViterbi::test_viterbi_basic -v
```

### Coverage Scope

| Module                  | Coverage                                                                    |
| ----------------------- | --------------------------------------------------------------------------- |
| Encoders (RLL/MTR/TMTR) | All codeword lookup tables and replacement rules                            |
| Decoders                | Reverse lookup for all codewords and invalid-codeword counting              |
| Channel models          | Longitudinal/Perpendicular/HAMR + media noise                               |
| LPF/FIR filters         | Multiple filter lengths and decimation factors                              |
| LMS equalizer           | Convergence behavior and GPR target adaptation                              |
| Viterbi detector        | Multiple PR targets, delays, code constraints                               |
| SOVA detector           | Soft-output quality and probability propagation                             |
| End-to-end integration  | Full combinations of coding + channel + equalization + detection + decoding |

## Experiments <a name="experiments"></a>

The repository includes **14 built-in experiments** for module verification and figure generation:

| Experiment | Description                                                |
| ---------- | ---------------------------------------------------------- |
| **Exp1**   | Channel pulse responses (Longitudinal/Perpendicular/HAMR)  |
| **Exp2**   | LPF frequency response                                     |
| **Exp3**   | FIR filter validation                                      |
| **Exp4**   | RLL(4/5) code analysis (rate and run-length distribution)  |
| **Exp5**   | Viterbi BER vs SNR (clean channel + LMS-equalized channel) |
| **Exp6**   | SOVA soft-output analysis                                  |
| **Exp7**   | LMS equalizer convergence across SNR values                |
| **Exp8**   | GPR target adaptation                                      |
| **Exp9**   | Coding overhead analysis                                   |
| **Exp10**  | LCG random number generator analysis                       |
| **Exp11**  | Viterbi vs SOVA comparison                                 |
| **Exp12**  | End-to-end BER vs SNR (No coding/RLL/MTR/TMTR)             |
| **Exp13**  | Loopback BER comparison for all coding schemes             |
| **Exp14**  | Encoder-decoder identity validation (zero-noise)           |

Generated figures are stored in `experiments/assets/`.

## Project Structure <a name="structure"></a>

```text
python_receiver/
|-- simulator.py                    # Main simulation driver (~780 lines)
|-- channel/
|   |-- channel.py                  # Channel models (Longitudinal/Perpendicular/HAMR)
|   |-- hamr_channel.py             # Full HAMR physical model
|   |-- lpf.py                      # Low-pass filter
|   |-- fir.py                      # FIR filter
|   |-- media_noise.py              # Media noise (jitter/pulse broadening)
|   `-- math_utils.py               # LCG, Gaussian noise, matrix utilities
|-- encoders/
|   |-- rll_4_5.py                  # RLL(0,2) 4/5 encoder
|   |-- mtr_6_7.py                  # MTR(2;8) 6/7 encoder
|   |-- tmtr_8_9.py                 # TMTR(2/3;11) 8/9 encoder
|   `-- permutation.py              # Tribit minimization permutation
|-- decoders/
|   |-- rll_4_5.py                  # RLL 4/5 decoder
|   |-- mtr_6_7.py                  # MTR 6/7 decoder
|   `-- tmtr_8_9.py                 # TMTR 8/9 decoder
|-- equalizer_detector/
|   |-- viterbi.py                  # Classical Viterbi detector
|   |-- sova.py                     # SOVA soft-output detector
|   |-- equalizer.py                # LMS equalizer + GPR target adaptation
|   |-- constrained_detectors.py    # Code-constrained Viterbi/SOVA
|   `-- detector.py                 # Public detector API
|-- experiments/
|   |-- run_experiments.py          # 14-experiment suite
|   |-- REPORT.md                   # Detailed experiment report
|   `-- assets/                     # Generated figures
|-- tests/
|   |-- test_channel.py             # Channel model tests
|   |-- test_encoders.py            # Encoder tests
|   |-- test_decoders.py            # Decoder tests
|   |-- test_equalizer_detector.py  # Equalizer + detector tests
|   |-- test_constrained_detectors.py
|   |-- test_integration.py         # Integration tests
|   |-- test_coverage_branches.py   # Branch coverage tests (241 cases)
|   |-- test_permutation_and_math_utils.py
|   `-- test_simulator_and_integration.py
`-- data/                           # Codebook files (.dat)
```

## Bug Log <a name="bugs"></a>

A total of **13 bugs** were identified and fixed during porting and validation.

### Detector Bugs

| Bug                          | Location             | Symptom                                                            | Root Cause                                                         |
| ---------------------------- | -------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| Missing DC bias compensation | viterbi.py / sova.py | BER stuck around 0.37 for non-DC-free PR targets (e.g., PR[1,2,1]) | Branch metric missed subtracting `sum(pri)`                        |
| SOVA buffer contamination    | sova.py:93,104       | SOVA hard-decision BER remained around ~0.36                       | `metric0 = path_metric[0, ...]` should be `path_metric[prev, ...]` |

### Channel Model Bugs

| Bug                             | Location     | Symptom                            | Root Cause                                                                 |
| ------------------------------- | ------------ | ---------------------------------- | -------------------------------------------------------------------------- |
| Incorrect noise injection range | simulator.py | BER too high at high SNR           | Noise should be added only to the internal region, not the entire sequence |
| LPF input truncation            | simulator.py | Tail drop in filtered output       | LPF input should be `ch_output[:oss_len]`                                  |
| HAMR simplified-channel bug     | channel.py   | Incorrect HAMR physical parameters | Simplified channel did not pass parameters correctly                       |

### RNG and Numerical Bugs

| Bug                          | Location      | Symptom                          | Root Cause                                                                            |
| ---------------------------- | ------------- | -------------------------------- | ------------------------------------------------------------------------------------- |
| LCG cache conflict           | math_utils.py | Correlated noise across channels | Caching `gaussian_random` on LCG object violated `__slots__`                          |
| Gaussian generation mismatch | math_utils.py | Statistical distribution bias    | Box-Muller cache behavior was inconsistent with C version (fixed with CachedGaussian) |

### Experiment and Test Bugs

| Bug                                   | Location                  | Symptom                                       | Root Cause                                         |
| ------------------------------------- | ------------------------- | --------------------------------------------- | -------------------------------------------------- |
| GPR target parameter name mismatch    | test_coverage_branches.py | Test crash                                    | `look_ahead` parameter mismatch                    |
| Incorrect SOVA return-value unpacking | test_coverage_branches.py | Return-value count mismatch                   | SOVA returns 3 values but code unpacked 2          |
| Wrong encoded-length reference        | test_coverage_branches.py | Slice out-of-bounds                           | Should use `encoded_len` instead of `len(encoded)` |
| Incorrect nested result indexing      | test_coverage_branches.py | Dictionary indexing error                     | `results["results"]` should be `results`           |
| False BER=0 plotting crash            | run_experiments.py        | `semilogy` crash when all BER points are zero | Added minimum detectable BER floor                 |
| Outdated Exp5 artifact                | assets/                   | Stale plot remained                           | Removed obsolete `exp5_ber_snr_viterbi.png`        |

### Pending Clarifications

- In C code, `ClassicalViterbi_4By5RLLCode` is identical to `ClassicalViterbi` (no separate port needed).
- In C code, legacy `Viterbi()`/`SOVA()` implementations based on ValidState have been replaced by `ClassicalViterbi`/`ClassicalSOVA`.
- In C code, the temperature-modulation tribit detection block is disabled via `&& 0` (dead code).
- In C code, `ClassicalViterbiForTempMod` depends on a missing `ValidSequencesOfLength9.dat` file.

## Relationship to the C Codebase

| Metric                | C                        | Python                         |
| --------------------- | ------------------------ | ------------------------------ |
| Source files          | 6 (.c/.h)                | 26 modules                     |
| Lines of code         | ~8,400                   | ~11,300                        |
| Tests                 | None                     | 241 cases                      |
| Experiment validation | None                     | 14 experiments                 |
| Functional coverage   | Reference implementation | 100% of active C functionality |

The Python implementation covers all active functionality from the C code, excluding dead code, duplicated code, and deprecated legacy paths.

## Authors <a name="authors"></a>

- [@DeepSeek](https://deepseek.com) - Code review
- Qwen3.6-plus - Bug fixes
- Gemma4-31B-it - Initial implementation
- Qwen3.6-27B - Polishing, experiment design, and documentation

## Acknowledgements <a name="acknowledgement"></a>

- Thanks to Prof. Bane Vasic Lab for the foundational research.
- Thanks to open-source LLM communities for contributions to this project.

## License <a name="license"></a>

[Apache-2.0](LICENSE)

# HAMR Receiver Simulator -- Python Implementation

## Overview

This is a Python translation of the C HAMR (Heat-Assisted Magnetic Recording)
simulator from `MagneticDisk.c` and related files. It simulates the full
digital receiver pipeline for magnetic disk storage systems, including:

- **Encoding**: 4/5 RLL(0,2), 6/7 MTR(2;8), 8/9 TMTR(2/3;11) codes
- **Channel models**: Longitudinal, Perpendicular, HAMR
- **Filtering**: Low-pass FIR filter, non-causal and causal FIR
- **Equalization**: Fixed FIR, LMS adaptive equalizer, GPR target adaptation
- **Detection**: Classical Viterbi, SOVA (Soft-Output Viterbi Algorithm)
- **Decoding**: Reverse of the encoder codes

Pipeline:

```
UserBits -> Encoder -> Channel -> LPF -> Downsampler -> Equalizer -> Detector -> Decoder
```

## Prerequisites

```bash
cd python_receiver
pip install -r requirements.txt
```

Required packages: numpy, scipy, matplotlib, pytest, coverage

## Directory Structure

```
python_receiver/
├── encoders/          # RLL, MTR, TMTR encoders + permutation
│   ├── __init__.py
│   ├── rll_4_5.py     # 4/5 RLL(0,2) encoder
│   ├── mtr_6_7.py     # 6/7 MTR(2;8) encoder
│   ├── tmtr_8_9.py    # 8/9 TMTR(2/3;11) encoder
│   └── permutation.py # Temperature modulation permutation tracking
├── channel/           # Channel models and signal processing
│   ├── __init__.py
│   ├── channel.py     # Unified channel dispatcher
│   ├── hamr_channel.py # Detailed HAMR channel physics
│   ├── fir.py         # Non-causal and causal FIR filters
│   ├── lpf.py         # Low-pass windowed-sinc FIR filter
│   ├── math_utils.py  # LCG RNG, Gaussian, autocorrelation, matrix ops
│   └── media_noise.py # Media noise model (stub)
├── equalizer_detector/# Equalizer + detectors
│   ├── __init__.py
│   ├── equalizer.py   # FIR, LMS adaptive equalizer, GPR target
│   ├── viterbi.py     # Classical Viterbi PRML detector
│   ├── sova.py        # Soft-Output Viterbi Algorithm
│   └── constrained_detectors.py  # Code-constrained detectors (stub)
├── decoders/          # RLL, MTR, TMTR decoders
│   ├── __init__.py
│   ├── rll_4_5.py     # 4/5 RLL(0,2) decoder
│   ├── mtr_6_7.py     # 6/7 MTR(2;8) decoder
│   └── tmtr_8_9.py    # 8/9 TMTR(2/3;11) decoder
├── utils/             # Shared utilities (stub)
├── data/              # Codeword lookup table .dat files
├── tests/             # Unit and integration tests
│   ├── conftest.py    # Shared fixtures
│   ├── test_encoders.py
│   ├── test_decoders.py
│   ├── test_channel.py
│   ├── test_equalizer_detector.py
│   └── test_integration.py
├── simulator.py       # Top-level simulation orchestrator
├── requirements.txt   # Python dependencies
├── WORKFLOW.md        # This file
├── experiments/       # Comprehensive experiment suite
│   ├── run_experiments.py  # 13 validation experiments
│   ├── REPORT.md           # Academic test report
│   ├── assets/             # Generated figures
│   └── results/            # Experiment JSON results
├── results/           # Simulation output data (generated)
│   ├── simulation_summary.csv
│   └── test_report.txt
└── figures/           # Plots and visualizations (generated)
    └── ber_vs_snr.png
```

## Running the Simulator

### Quick Start (default settings)

```bash
cd python_receiver
python simulator.py
```

Runs a single SNR point (21 dB) with default HAMR channel, Viterbi detector,
no encoding. Outputs `results/simulation_summary.csv` and `figures/ber_vs_snr.png`.

### With Custom Configuration (Python API)

```python
from simulator import run_simulation, SimulatorConfig

# Example: BER sweep with perpendicular channel, RLL encoding
config = SimulatorConfig(
    snr_db=[18.0, 20.0, 22.0, 24.0, 26.0],
    channel_type="Perpendicular",
    detector_type="Viterbi",
    use_encoding=True,
    encoder_type="rll_4_5",
    max_num_sectors=100,
    max_num_bit_err=50,
    osr=10,
    sector_length=4096,
    num_eq_taps=21,
    equalizer_type="GPRTarget",
)

results = run_simulation(config)
print(f"BER per SNR: {results['ber_per_snr']}")
print(f"SER per SNR: {results['ser_per_snr']}")
```

### Running as Module

```bash
cd python_receiver
python -m simulator
```

### Configuration Options

#### Channel Types
- `"Longitudinal"` -- Longitudinal recording channel
- `"Perpendicular"` -- Perpendicular recording channel
- `"Hamr"` -- HAMR channel model (default)

#### Detector Types
- `"Viterbi"` -- Classical Viterbi PRML detector (default)
- `"SOVA"` -- Soft-Output Viterbi Algorithm

#### Equalizer Types
- `"FixedPRTarget"` -- Fixed FIR equalizer with PR target coefficients
- `"GPRTarget"` -- GPR target adaptive equalizer (default), computes
  optimal coefficients from high-SNR training sectors

#### Encoder Types
- `"none"` -- No encoding (raw bits, default)
- `"rll_4_5"` -- 4/5 RLL(0,2) code
- `"mtr_6_7"` -- 6/7 MTR(2;8) code
- `"tmtr_8_9"` -- 8/9 TMTR(2/3;11) code

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `snr_db` | `[21.0]` | SNR values in dB to sweep |
| `max_num_sectors` | `10` | Maximum sectors per SNR point |
| `min_num_sectors` | `10` | Minimum sectors before early termination |
| `max_num_bit_err` | `100` | Max total bit errors before early termination |
| `osr` | `10` | Oversampling rate |
| `sector_length` | `4096` | Sector length in bits |
| `pre_padding_length` | `20` | Padding bits before sector |
| `post_padding_length` | `20` | Padding bits after sector |
| `num_channel_taps` | `201` | Channel filter taps |
| `num_eq_taps` | `21` | Equalizer taps |
| `viterbi_delay` | `20` | Viterbi detection delay |
| `pri_imp_res` | `[1,1,-1,-1]` | PR target (EPR4) |

## Running Tests

```bash
cd python_receiver

# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=. --cov-report=term-missing

# Run a single test file
pytest tests/test_encoders.py -v

# Run a single test class
pytest tests/test_equalizer_detector.py::TestViterbiDetector -v

# Run with JUnit XML output (for CI)
pytest tests/ -v --junitxml=results/test_results.xml
```

### Test Categories

| Test File | Tests | Description |
|---|---|---|
| `test_encoders.py` | 11 | Encoder round-trip (encode -> decode), codeword table integrity |
| `test_decoders.py` | 9 | Decoder round-trip, invalid codeword handling, substitution undo |
| `test_channel.py` | 17 | Channel models, FIR filters, LPF, RNG, math utilities |
| `test_equalizer_detector.py` | 14 | Viterbi, SOVA, equalizer convergence, FIR filters |
| `test_integration.py` | 8 | Full pipeline: LPF->Viterbi, equalizer adaptation, detector comparison |

### Output Files

#### Results (`results/`)
- `simulation_summary.csv` -- BER, SER, sector counts per SNR point (generated by simulator)
- `test_report.txt` -- Full test coverage and failure report

#### Figures (`figures/`)
- `ber_vs_snr.png` -- BER curve vs SNR (log-scale Y axis)

## Architecture

### Encoding Pipeline

```
User bits (0/1)
    |
    v
NRZI conversion: NRZI[i] = abs(user_bits[i] - user_bits[i-1])
    |
    v
Codeword lookup: Every N bits -> N-bit codeword from .dat table
    |
    v
NRZ accumulation: NRZ[i+1] = NRZ[i] XOR codeword[i]
    |
    v
Encoded bits (bipolar: 0 -> -1, 1 -> +1)
```

Each code has a sector length validation (4Z+1 for RLL, 6Z+1 for MTR, 8Z+1 for TMTR).

### Decoding Pipeline

```
Detector output (NRZ)
    |
    v
NRZ -> NRZI conversion: Differencing adjacent bits
    |
    v
Codeword lookup: Every N bits -> find codeword index (original NRZI value)
    |
    v
NRZI -> NRZ: Accumulate NRZI values
    |
    v
Decoded user bits
```

Key: The decoder uses a **Codeword2Dec** reverse mapping -- the codeword's
position index in the table IS the original NRZI value, not the codeword's
decimal representation.

### Channel Models

- **Longitudinal**: Simple Gaussian-shaped isolated peak model
- **Perpendicular**: Includes demagnetizing effects and spacing loss
- **HAMR**: Full physics model with temperature profile, permeability,
  recalculation factor, and microtrack effects

### Equalization

1. **FixedPRTarget**: Apply pre-computed FIR coefficients
2. **GPRTarget**: Two-phase approach:
   - Phase 1: Run high-SNR training sectors, compute GPR target and
     equalizer coefficients via least-squares solution
   - Phase 2: Apply fixed coefficients to all simulation sectors

### Detection

- **Viterbi**: PRML detector using trellis with squared Euclidean distance
  metrics, ping-pong path metric buffers, and traceback.
- **SOVA**: Soft-output variant that tracks probability of wrong detection
  for each bit, producing both hard decisions and soft confidence values.

## C Reference Mapping

| Python Module | C Source File | Line Range |
|---|---|---|
| `simulator.py` | MagneticDisk.c | main() ~3356-4688 |
| `channel/channel.py` | MagneticDisk.c | Channel functions |
| `channel/fir.py` | MagneticDisk.c | NonCausalFIR, CausalFIR |
| `channel/lpf.py` | MagneticDisk.c | LPF (windowed-sinc) |
| `equalizer_detector/equalizer.py` | MagneticDisk.c | AdaptEqualizer |
| `equalizer_detector/viterbi.py` | MagneticDisk.c | Viterbi detector |
| `equalizer_detector/sova.py` | MagneticDisk.c | SOVA detector |
| `encoders/rll_4_5.py` | EncodingFunctions.c | Enc_4By5RLLCode |
| `encoders/mtr_6_7.py` | EncodingFunctions.c | Enc_6By7MTRCode |
| `encoders/tmtr_8_9.py` | EncodingFunctions.c | Enc_8By9TMTRCode |
| `decoders/rll_4_5.py` | DecodingFunctions.c | Dec_4By5RLLCode |
| `decoders/mtr_6_7.py` | DecodingFunctions.c | Dec_6By7MTRCode |
| `decoders/tmtr_8_9.py` | DecodingFunctions.c | Dec_8By9TMTRCode |

## Development

### Adding a New Test

```bash
# Create test file following naming convention
touch tests/test_new_module.py

# Run: pytest tests/test_new_module.py -v
```

### Code Style

- Follow PEP 8 conventions
- Type annotations on all function signatures
- Format with black, sort imports with isort, lint with ruff

```bash
black .
isort .
ruff check .
```

## Experiment Suite

A comprehensive suite of 13 experiments validates the Python translation and collects data for the academic test report.

```bash
cd python_receiver
python experiments/run_experiments.py
```

### Experiments

| # | Name | Description |
|---|------|-------------|
| 1 | Channel Impulse Response | Longitudinal vs Perpendicular channel shapes |
| 2 | LPF Frequency Response | Windowed-sinc filter response for orders 20/50/100 |
| 3 | FIR Filter Verification | Compare against manual convolution reference |
| 4 | RLL(4/5) Code Analysis | Codeword table and transition density |
| 5 | BER vs SNR (Viterbi) | Channel performance at 16-30 dB SNR |
| 6 | SOVA Soft Output | Confidence vs correctness correlation |
| 7 | Equalizer Convergence | LMS MSE reduction verification |
| 8 | GPR Target Adaptation | Cross-correlation target computation |
| 9 | Encoding Overhead | Bit overhead comparison across codes |
| 10 | LCG RNG Analysis | Statistical uniformity verification |
| 11 | Viterbi vs SOVA | Hard vs soft detection comparison |
| 12 | Full Pipeline | End-to-end with all encoding schemes |
| 13 | All Codes BER | Round-trip BER for all encoder/decoder pairs |

### Output

- **Report:** `experiments/REPORT.md` (full academic test report)
- **Figures:** `experiments/assets/*.png`
- **Results:** `experiments/results/experiment_results.json`

### Test Results

- **Unit tests:** 66/66 passed
- **Code coverage:** 66% overall, 80%+ for all core modules
- **Experiments:** 13/13 passed
- **Total runtime:** ~60 seconds

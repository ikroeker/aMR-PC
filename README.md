# aMR-PC

**Arbitrary Multi-Resolution Polynomial Chaos python toolbox**

![test status](https://github.com/ikroeker/aMR-PC/actions/workflows/python-package.yml/badge.svg?event=push)

A Python toolbox for **data-driven uncertainty quantification** using arbitrary multi-resolution polynomial chaos (aMR-PC) expansion and multi-wavelet methods. The package provides tools for constructing polynomial chaos surrogates from data, performing multi-resolution decomposition, computing Sobol sensitivity indices, and Bayesian inference diagnostics.

---

## Features

- Arbitrary polynomial chaos (aPC) basis construction from data moments
- Multi-wavelet construction via the Le Maitre et al. (2004) algorithm
- Multi-resolution decomposition and adaptive refinement
- aMR-PC expansion via least-squares and quadrature-based approaches
- Sobol global sensitivity indices for PC and aMR-PC expansions
- Statistical tools: Gaussian likelihood, Bayesian Model Evidence (BME), KL-divergence, entropy
- Numba JIT acceleration for performance-critical routines (with graceful fallback)

---

## Requirements

- **Python** >= 3.10
- **NumPy**
- **SciPy**
- **Pandas**
- **Numba**

---

## Installation

Install in development mode from a local clone:

```bash
pip install -e .
```

The `-e` switch means that pip will only link the source files to the directory
where all your Python packages are installed, so that any changes in the source
code take effect directly without reinstallation.

Or install directly from GitHub:

```bash
pip install git+https://github.com/ikroeker/aMR-PC.git
```

---

## Code sources

aMR-PC Python code is hosted on
[GitLab](https://git.iws.uni-stuttgart.de/ikroeker/ik_amr-pc) and
[GitHub](https://github.com/ikroeker/aMR-PC).

---

## Repository structure

```
aMRPC/                   # Main Python package
  __init__.py            # Package init, version (1.0.4)
  datatools.py           # Core data management, MR decomposition, reconstruction
  iodata.py              # File I/O routines, naming conventions
  polytools.py           # aPC polynomial basis, Gaussian quadrature
  sobol.py               # Sobol sensitivity indices
  stat.py                # Statistics: likelihood, BME, entropy, KL-divergence
  utils.py               # Multi-index generation, multi-key management
  wavetools.py           # Multi-wavelet construction (WaveTools class)
tests/
  data/                  # Test input data (InputParameters.txt)
  py/                    # Test files (test_00 through test_14)
```

---

## Package content

### Module overview

| Module | Responsibility |
|---|---|
| `utils.py` | Multi-index generation (graded lexicographic ordering), dictionary key management for multi-resolution elements identified by tuple keys `(aNr, Nri, src)` |
| `polytools.py` | Raw moments, Hankel matrices, aPC/PC polynomial coefficients, alpha-beta recurrence, Jacobi matrix, Gaussian quadrature roots/weights, normalization |
| `wavetools.py` | `WaveTools` class: multi-wavelet and scaling function construction on (0,1) via Le Maitre et al. algorithm |
| `datatools.py` | Central orchestration: Hankel generation from data, roots/weights, MR-element management, adaptivity, aMR-PC decomposition (LS and quadrature), function reconstruction |
| `sobol.py` | Sobol sensitivity indices for PC and aMR-PC expansions, total sensitivity indices |
| `stat.py` | Gaussian likelihood, Bayesian Model Evidence (BME), log-likelihood, KL-divergence, entropy |
| `iodata.py` | Read/write evaluation points, pickle/numpy I/O, filename generation conventions |

### Key concepts

- **Multi-resolution elements** are identified by tuple keys `(aNr, Nri, src)` where `aNr` = resolution level, `Nri` = element index, `src` = source/dimension.
- **Multi-keys** are tuples of tuples for multi-dimensional problems.
- **Numba JIT** (`@njit`, `@jit`) is used extensively for performance-critical functions with graceful fallback when Numba is unavailable.

---

## Usage

Import individual modules from the package:

```python
import aMRPC.datatools as dt
import aMRPC.polytools as pt
import aMRPC.wavetools as wt
import aMRPC.sobol as sobol
import aMRPC.stat as stat
import aMRPC.utils as utils
```

The test files in `tests/py/` serve as practical usage examples. In particular:

- `test_05_datatools.py` demonstrates Hankel matrix generation from data, MR decomposition, and aMR-PC expansion
- `test_06_sobol.py` demonstrates Sobol sensitivity analysis using the Ishigami benchmark function
- `test_07_stat.py` demonstrates likelihood computation, BME, and KL-divergence

---

## Testing

Run the full test suite:

```bash
pytest
```

Run tests with warnings treated as errors (as CI does):

```bash
pytest -W error
```

Lint with flake8:

```bash
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

Tests are ordered by dependency: basic tests (`test_00` through `test_07`) cover core module functionality, and extended tests (`test_08` through `test_14`) cover edge cases and advanced usage.

---

## Related publications

**Please cite the article:**

Ilja Kröker, Sergey Oladyshkin,
*Arbitrary multi-resolution multi-wavelet-based polynomial chaos expansion for data-driven uncertainty quantification.*
Reliability Engineering & System Safety,
Volume 222, 2022, 108376, ISSN 0951-8320,
https://doi.org/10.1016/j.ress.2022.108376.

Also used in the following publications:

Rebecca Kohlhaas; Ilja Kröker; Sergey Oladyshkin; Wolfgang Nowak
*Gaussian active learning on multi-resolution arbitrary polynomial chaos emulator: concept for bias correction, assessment of surrogate reliability and its application to the carbon dioxide benchmark*
Comput Geosci (2023).
https://doi.org/10.1007/s10596-023-10199-1

Ilja Kroeker, Sergey Oladyshkin, Iryna Rybak
*Global sensitivity analysis using multi-resolution polynomial chaos expansion for coupled Stokes-Darcy flow problems.*
https://doi.org/10.1007/s10596-023-10236-z

Ilja Kröker, Tim Brünnette, Nils Wildt, Maria Fernanda Morales Oreamuno, Rebecca Kohlhaas, Sergey Oladyshkin, Wolfgang Nowak
*Bayesian³ Active learning for regularized arbitrary multi-element polynomial chaos using information theory*
https://doi.org/10.1615/int.j.uncertaintyquantification.2024052675

---

## Author

Ilja Kröker
[ORCID: 0000-0003-0360-5307](https://orcid.org/0000-0003-0360-5307)
[Institute for Modelling Hydraulic and Environmental Systems (IWS)](https://www.iws.uni-stuttgart.de),
[Department of Stochastic Simulation and Safety Research for Hydrosystems (LS3)](https://www.iws.uni-stuttgart.de/ls3/),
University of Stuttgart

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

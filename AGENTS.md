# AGENTS.md

## Project Overview

**aMR-PC** (Arbitrary Multi-Resolution Polynomial Chaos) is a Python toolbox for data-driven uncertainty quantification. It implements arbitrary multi-resolution polynomial chaos expansion and multi-wavelet methods.

- **Language**: Python >= 3.10
- **Dependencies**: NumPy, SciPy, Pandas, Numba
- **License**: MIT

## Repository Structure

```
aMRPC/               # Main Python package
  __init__.py        # Package init, version
  datatools.py       # Core data management, MR decomposition, reconstruction
  iodata.py          # File I/O routines, naming conventions
  polytools.py       # aPC polynomial basis, Gaussian quadrature
  sobol.py           # Sobol sensitivity indices
  stat.py            # Statistics: likelihood, BME, entropy, KL-divergence
  utils.py           # Multi-index generation, multi-key management
  wavetools.py       # Multi-wavelet construction (WaveTools class)
tests/
  data/              # Test input data
  py/                # Test files (test_00 through test_14)
```

## Key Concepts

- **Multi-resolution elements** are identified by tuple keys `(aNr, Nri, src)` where `aNr` = resolution level, `Nri` = element index, `src` = source/dimension.
- **Multi-keys** are tuples of tuples for multi-dimensional problems.
- **Numba JIT** (`@njit`, `@jit`) is used extensively for performance-critical functions with graceful fallback when Numba is unavailable.

## Build and Test

```bash
# Install in development mode
pip install -e .

# Run tests
pytest

# Run tests with warnings as errors (as CI does)
pytest -W error

# Lint
flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
```

## CI/CD

- **GitLab CI** (`.gitlab-ci.yml`): build + test stages on Python 3.10. Test job runs `pytest -W error` and only triggers on changes to `aMRPC/**` or `tests/**`.
- **GitHub Actions** (`.github/workflows/python-package.yml`): flake8 lint + `pytest -W error` across Python 3.10–3.13, on push/PR to `master`.
- CI treats warnings as errors, so run `pytest -W error` locally before pushing.

## Module Guide

| Module | Responsibility |
|--------|---------------|
| `utils.py` | Multi-index generation, dictionary key management, helper functions |
| `polytools.py` | Hankel matrices, aPC polynomial coefficients, Gaussian quadrature, normalization |
| `wavetools.py` | `WaveTools` class: multi-wavelet construction via Le Maitre et al. algorithm |
| `datatools.py` | Central orchestration: Hankel generation from data, roots/weights, MR-element management, adaptivity, aMR-PC decomposition, function reconstruction |
| `sobol.py` | Sobol sensitivity indices for PC and aMR-PC expansions, total sensitivity |
| `stat.py` | Gaussian likelihood functions, Bayesian Model Evidence, KL-divergence, entropy |
| `iodata.py` | Read/write evaluation points, pickle/numpy I/O, filename conventions |

## Coding Conventions

- Primarily procedural/functional style; `WaveTools` in `wavetools.py` is the only class.
- Dictionary-based data structures keyed by multi-resolution tuples.
- Numba-decorated functions use `cache=True` and `nogil=True`.

## Testing Notes

- No pytest config file exists; tests are collected in filename order. Basic tests (`test_00`-`test_07`) are meant to run before extended tests (`test_08`-`test_14`).
- Test files do `import context` first (see `tests/py/context.py`) to insert the repo root on `sys.path`, then `import aMRPC.<module>`. New test files must keep the `import context` line before any `aMRPC` import.
- Test input data lives in `tests/data/`.

## Context7 Skills

The following Context7 skills are installed locally (in `.agents/skills/` and `.claude/skills/`, gitignored) to provide domain-specific guidance for AI agents:

| Skill | Source | Purpose |
|-------|--------|---------|
| `numpy` | `/microsoft/debugpy` | NumPy best practices: arrays, broadcasting, vectorization |
| `pytest` | `/microsoft/debugpy` | Test writing: fixtures, parametrize, plugins |
| `python` | `/alinaqi/claude-bootstrap` | Python development: ruff, mypy, pytest, TDD, type safety |
| `github-actions` | `/tartinerlabs/skills` | CI/CD workflow creation, auditing, SHA pinning |

To reinstall skills after a fresh clone:

```bash
npx ctx7 skills install /microsoft/debugpy numpy --universal -y
npx ctx7 skills install /microsoft/debugpy pytest --universal -y
npx ctx7 skills install /alinaqi/claude-bootstrap python --universal -y
npx ctx7 skills install /tartinerlabs/skills github-actions --universal -y
```

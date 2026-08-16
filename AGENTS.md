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
- **Sherman-Morrison-Woodbury identity** (`gen_cov_mx_4lh`, `gen_cov_mx_4lh_noex` in `datatools.py`) is used to invert the likelihood covariance matrix `cov_mx = Q + phi @ R @ phi.T` (n x n, n = number of samples), routed by the relative size of n and p (p = number of polynomials, `phi.shape[1]`): Woodbury (inverting the much smaller `P = phi.T @ Q_inv @ phi + R_inv`, p x p) when `p < n`; a direct Cholesky inverse of `cov_mx` when `p >= n` (Woodbury gives no benefit there). `np.linalg.pinv(cov_mx)` is used only as a last-resort exception fallback (e.g. if `P` or `cov_mx` is not positive definite).
- The regularized `reg_n`/`reg_t` methods of the `gen_amrpc_dec_*` functions (`gen_amrpc_dec_ls`, `gen_amrpc_dec_ls_mask(_aux)`, `gen_amrpc_dec_mk_ls`, `gen_reg_t_aux` in `datatools.py`) invert the (SPD by construction) p x p normal-equations matrix `P = phi.T @ phi / sigma_n^2 [+ I / sigma_p^2]` directly via its Cholesky factorization (`inv(cholesky(P))`), with `np.linalg.pinv(P)` used only as a last-resort exception fallback. Unlike `gen_cov_mx_4lh`, Woodbury does not apply here since `P` is already the small p x p matrix (there is no larger n x n matrix to avoid inverting).
- **Batched column solve**: `gen_amrpc_dec_ls`, `gen_amrpc_dec_mk_ls`, `gen_amrpc_dec_ls_mask(_aux)`, and `gen_reg_t_aux` (`datatools.py`) solve all `x_len` spatial/time columns of `data` in a single batched matrix product (`operator @ data_block`, or a single `np.linalg.lstsq`/`scipy.linalg.lstsq` call with a 2-D right-hand side) instead of looping per column, since the solve operator (`P_inv`, `mx_inv`, `pinv(phi)`, ...) is constant across `idx_x`. `return_std`/`return_cov` outputs are filled via broadcasting for the same reason. This gives ~5-35x speedups (BLAS-2 `gemv` loop -> BLAS-3 `gemm`/batched factorization) with no GPU/CUDA needed (evaluated and rejected as not worth it for this codebase's problem sizes and numpy `float64`-only linear algebra — see git history). `cmp_var_4_mk` similarly replaces a per-sample double loop over active polynomial indices with a batched matmul + reduction. Degenerate `n_s <= 1` edge cases are kept as plain loops (negligible size, not worth the added risk).
- **Known pre-existing issues** (not fixed as part of the vectorization, flagged for future work): (1) `gen_amrpc_dec_ls_mask_aux` unconditionally allocates `ret_std_cov_4s` of shape `(n_s, p_max, p_max, x_len)` regardless of `cov_mode`, even when no covariance/std output is requested (`cov_mode == 0`); this `O(n_s * p_max^2 * x_len)` allocation can dominate runtime and OOM at scale. (2) `gen_amrpc_dec_ls_mask_aux` raises a shape-mismatch error when called with a **partial** `alpha_mask` together with `cov_mode in (1, 2)` and `meth_mode in (2, 4, 5)`, since `P_inv`/`cov_op` is sized by the number of active mask entries rather than `p_max`.

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

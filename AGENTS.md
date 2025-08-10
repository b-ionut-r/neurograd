# Repository Guidelines

## Project Structure & Module Organization
- `neurograd/`: Core package. Key areas: `tensor.py` (autograd core), `functions/` (ops: math, linalg, conv, reductions), `nn/` (modules, losses, metrics), `optim/` (SGD/Adam/RMSProp), `amp/` (autocast, GradScaler), `utils/` (device, data, grad_check, graph).
- Notebooks live at repo root (e.g., `lenet5.ipynb`). Build outputs in `build/` and `dist/`.
- Add tests under `tests/` (create if missing). Keep example data small and deterministic.

## Dev Setup, Build, and Test
- Create env and install dev deps: `python -m venv .venv && source .venv/bin/activate && pip install -e .[dev]`
- Optional GPU/CuPy: `pip install -e .[gpu]` (CUDA 12.x).
- Run tests: `pytest -q` (coverage: `pytest --cov=neurograd`).
- Lint/format/type-check: `black . && flake8 neurograd && mypy neurograd`.
- Build package: `python setup.py sdist bdist_wheel` (artifacts in `dist/`).

## Coding Style & Naming Conventions
- Python ≥3.7. Use 4-space indentation and type hints for public APIs.
- Black formatting, Flake8 linting; keep lines black-friendly (~88 cols).
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.
- Keep public API stable; document breaking changes in PR description and commit message.

## Testing Guidelines
- Framework: pytest. Place files as `tests/test_<area>.py`; name tests `test_*`.
- Cover new operators and modules with forward + backward checks. For new ops, include gradient checks:
  ```python
  from neurograd.utils.grad_check import gradient_check
  assert gradient_check(model, X, y, loss_fn="MSE")
  ```
- Aim for meaningful coverage on core modules; use small tensors and fixed seeds.

## Commit & Pull Request Guidelines
- Prefer Conventional Commits (seen in history): `feat:`, `fix:`, `refactor:`, `docs:`, `test:`. Use scoped messages where helpful (e.g., `feat(amp): ...`).
- Commits: focused, atomic, imperative mood; explain rationale when non-obvious.
- PRs must include: clear description, linked issues, test coverage, API notes, and any perf impact. Add minimal examples or notebook references when relevant.

## Configuration & Runtime Tips
- Backend auto-detects device: `import neurograd as ng; print(ng.DEVICE)`.
- Visualization helpers require matplotlib; GPU features require CuPy. Guard optional imports and provide CPU fallbacks.

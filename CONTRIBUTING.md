# Contributing to HDVEC

We welcome contributions! Please follow these guidelines to keep the project healthy and consistent.

## Development setup

- Python 3.10+
- Recommended: [uv](https://github.com/astral-sh/uv)

```bash
uv venv -p 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
pre-commit install
```

## Quality checks

```bash
ruff check .
black --check .
mypy .
pytest -q
```

## Pull requests

- Include tests for new features and bug fixes.
- Keep changes focused; open separate PRs for unrelated changes.
- Update documentation (README/docs) when public APIs change.
- Ensure CI passes.

## Coding style & types

- Follow the existing style; `ruff` and `black` enforce most rules.
- Add precise type hints. Prefer `numpy.typing.NDArray` where practical.

## Release process

- We follow Semantic Versioning. Changelog entries should be added under `Unreleased` and moved to a version section on release.

## Conduct

- Be respectful and constructive. See `GOVERNANCE.md` for project roles and decisions.

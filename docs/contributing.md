# Contributing

Contributions are welcome: please check the [open issues](https://github.com/NeuroLIAA/pyxations/issues) and open a pull request if you'd like to help.

## Local development setup

```bash
# Clone the repository
git clone https://github.com/NeuroLIAA/pyxations.git
cd pyxations

# Create a virtual environment and install in editable mode with dev extras
uv venv
uv pip install -e '.[dev]'
```

`uv` is recommended but plain `pip` works the same way.

## Running tests

```bash
uv run pytest
```

The test suite covers BIDS conversion, derivative computation, hierarchy
loading, scanpath visualization, and multipanel plots. Small source-only
fixtures live under `examples/`; pytest generates raw BIDS, derivatives, and
figures in temporary directories.

BIDS raw and derivative writer tests use the official validator. Install
[Deno](https://docs.deno.com/runtime/getting_started/installation/) and run:

```bash
deno install -ERWN -g -n bids-validator jsr:@bids/validator@3.0.1
uv run pytest tests/test_0001_dataset_to_bids.py
```

Without Deno or an installed `bids-validator` command, structural writer tests
still run but official validation tests are skipped. CI always installs and
runs the pinned validator.

For coverage:

```bash
uv run pytest --cov=pyxations
```

## Building the docs locally

```bash
uv pip install -e '.[docs]'
mkdocs serve
```

`mkdocs serve` watches `docs/` and `mkdocs.yml` and rebuilds on save. The API reference is generated from source docstrings via [mkdocstrings](https://mkdocstrings.github.io/) (NumPy style).

## Coverage

Run the same coverage check used by CI:

```bash
pytest --cov=pyxations --cov-report=term-missing --cov-report=xml
```

CI enforces an 80% project-wide floor and uploads the report used by the
coverage badge in the README. The floor should increase as meaningful tests
are added, with 95% retained as the long-term target; it should not be raised
by excluding supported modules or adding tests that merely execute lines
without asserting behavior.

## Repository layout

- `pyxations/`: the package source.
    - `bids_formatting.py`, `pre_processing.py`: high-level entry points.
    - `bids.py`: vendor readers and canonical raw BIDS conversion.
    - `methods/eyemovement/`: REMoDNaV, Engbert–Kliegl detectors.
    - `analysis/`: `Experiment` and paradigm-specific helpers.
    - `visualization/`: plotting utilities.
    - `export/`: canonical BIDS derivative reader and writer.
- `tests/`: pytest suite.
- `docs/`: MkDocs sources for this site.
- `notebooks/`: runnable end-to-end examples.
- `examples/`: small source recordings used by the quickstart and integration
  tests; generated datasets are intentionally not committed.

## Coding style

- Format with `black` and lint with `ruff` (both included in the `dev` extra).
- Public functions should have NumPy-style docstrings so they render in the API reference.
- Add or update a test when changing behavior; keep the existing tests green.

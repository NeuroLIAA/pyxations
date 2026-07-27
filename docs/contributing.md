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

The test suite covers BIDS conversion, derivative computation, scanpath visualization and multipanel plots. Test fixtures live under `tests/data/`.

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

## Repository layout

- `pyxations/`: the package source.
    - `bids_formatting.py`, `pre_processing.py`: high-level entry points.
    - `formats/`: per-vendor parsers (EyeLink, Tobii, Gazepoint, WebGazer).
    - `methods/eyemovement/`: REMoDNaV, Engbert–Kliegl detectors.
    - `analysis/`: `Experiment` and paradigm-specific helpers.
    - `visualization/`: plotting utilities.
    - `export/`: Feather and HDF5 writers.
- `tests/`: pytest suite.
- `docs/`: MkDocs sources for this site.
- `notebooks/`: runnable end-to-end examples.
- `tests/data/eyelink_dataset/`: small bundled dataset used by the quickstart.

## Coding style

- Format with `black` and lint with `ruff` (both included in the `dev` extra).
- Public functions should have NumPy-style docstrings so they render in the API reference.
- Add or update a test when changing behavior; keep the existing tests green.

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

BIDS raw and derivative writer tests use the official validator, in
`test_0001_dataset_to_bids.py`, `test_0002_compute_derivatives.py` and
`test_bids_edge_cases.py`. Install
[Deno](https://docs.deno.com/runtime/getting_started/installation/) and run:

```bash
deno install -ERWN -g -n bids-validator jsr:@bids/validator@3.0.1
uv run pytest
```

Without Deno or an installed `bids-validator` command, structural writer tests
still run but official validation tests are skipped, and the run reports them
as skipped rather than passed. CI always installs and runs the pinned
validator, so those tests are never skipped there.

## Building the docs locally

```bash
uv pip install -e '.[docs]'
mkdocs serve
```

`mkdocs serve` watches `docs/` and `mkdocs.yml` and rebuilds on save. Note that
it does **not** watch the package source, so a change to a docstring only shows
up after restarting the server.

The API reference lives in `docs/api/`, one page per pipeline stage, and is
generated from source docstrings via
[mkdocstrings](https://mkdocstrings.github.io/) (NumPy style). A module is only
rendered if some page includes a `::: pyxations.<module>` directive, so a new
module needs an entry there as well as a `nav` entry in `mkdocs.yml`.

### Tutorials

The notebooks under `notebooks/` are **executed on every build**, so their
figures and tables are real output rather than something committed by hand.
That makes them an end-to-end test: if a change breaks a tutorial, the docs
build fails. Notebooks are therefore committed without stored outputs.

`docs/tutorials` is a symlink to `notebooks/`, because MkDocs only serves files
under `docs/`. Every notebook it reaches is executed whether or not it appears
in the `nav`, so one that cannot run unattended has to be listed under
`exclude_docs` in `mkdocs.yml`; `driving_animation.ipynb` is excluded because
it needs a dataset that is not committed.

Do **not** cache the executed notebooks in CI. The cache is keyed on notebook
content, so a tutorial broken by a change in the package would still be served
from cache and the build would pass. A full build takes about 20 seconds.

## Coverage

Run the same coverage check used by CI:

```bash
uv run pytest --cov=pyxations --cov-report=term-missing --cov-report=xml --cov-fail-under=95
```

CI enforces a 95% project-wide floor and uploads the report used by the
coverage badge in the README. Supported modules are not excluded from the
measurement; tests should assert behavior rather than merely execute lines.

## Repository layout

- `pyxations/`: the package source.
    - `bids_formatting.py`, `pre_processing.py`: high-level entry points.
    - `bids.py`: vendor readers and canonical raw BIDS conversion.
    - `behavior.py`, `psychopy.py`: behavioral table and PsychoPy log readers.
    - `tables.py`: the in-memory table container and BIDS TSV read/write.
    - `methods/eyemovement/`: REMoDNaV, Engbert–Kliegl detectors.
    - `analysis/`: `Experiment` and paradigm-specific helpers.
    - `visualization/`: plotting utilities.
    - `export/`: canonical BIDS derivative reader and writer.
- `tests/`: pytest suite.
- `docs/`: MkDocs sources for this site; `docs/api/` is the API reference.
- `notebooks/`: runnable end-to-end examples.
- `examples/`: small source recordings used by the quickstart and integration
  tests; generated datasets are intentionally not committed.

## Coding style

- Format with `black` and lint with `ruff` (both included in the `dev` extra).
- Add or update a test when changing behavior; keep the existing tests green.
- Every public class, function and method needs a NumPy-style docstring. This
  is enforced by `tests/test_docstring_coverage.py`, which also checks that
  section headings are underlined with dashes rather than followed by a colon,
  and that no module with public API is missing from the API reference.

Public API here means module-level classes and functions plus the public
methods of those classes. Nested helper functions are excluded: they are an
implementation detail and are never rendered.

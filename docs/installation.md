## Installation

Install the base package with `uv` or `pip`:

```bash
uv pip install pyxations
# or
pip install pyxations
```

Add only the optional features you need:

```bash
pip install "pyxations[remodnav]"
pip install "pyxations[multimatch]"
pip install "pyxations[video]"
```

Install all optional features with:

```bash
pip install "pyxations[all]"
```

For a reproducible checkout of the repository, use the committed universal
lock file:

```bash
uv sync --extra dev
```

The lock constrains the development environment; it does not impose exact
versions on users who install the library with pip.

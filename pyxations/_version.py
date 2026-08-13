"""Single source of truth for the package version.

The version is declared once, in ``pyproject.toml``, and read back here from
the installed distribution metadata. Anything that needs to report a version
imports it from this module, so a release bump cannot leave a stale copy
behind in the code.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pyxations")
except PackageNotFoundError:
    # Running straight from a source tree that was never installed. There is no
    # distribution to ask, and guessing a number would be worse than saying so.
    __version__ = "unknown"

__all__ = ["__version__"]

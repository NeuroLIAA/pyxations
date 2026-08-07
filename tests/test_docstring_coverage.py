"""Tests for public API documentation policy.

The public API is every module-level class and function that does not start
with an underscore, plus the public methods of those classes. Nested closures
are deliberately excluded: they are an implementation detail and are never
rendered in the documentation.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "pyxations"
DOCS = ROOT / "docs"

# Section headings that numpydoc underlines with dashes. A docstring that
# introduces one of them with a colon instead is using a different convention.
NUMPY_SECTIONS = (
    "Parameters",
    "Returns",
    "Yields",
    "Raises",
    "Attributes",
    "Examples",
    "Notes",
    "See Also",
)
FOREIGN_SECTION = re.compile(
    rf"^\s*(?:{'|'.join(NUMPY_SECTIONS)}|Args|Arguments|Keyword Args):",
    re.MULTILINE,
)


def _source_files() -> list[Path]:
    return sorted(path for path in PACKAGE.rglob("*.py") if "build" not in path.parts)


def _module_name(path: Path) -> str:
    relative = path.relative_to(ROOT).with_suffix("")
    return ".".join(relative.parts).removesuffix(".__init__")


def _public_api(path: Path) -> list[tuple[int, str, str | None]]:
    """Return ``(lineno, qualified_name, docstring)`` for one module's API."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    api: list[tuple[int, str, str | None]] = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.name.startswith("_"):
                api.append((node.lineno, node.name, ast.get_docstring(node)))
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            api.append((node.lineno, node.name, ast.get_docstring(node)))
            for member in node.body:
                if isinstance(
                    member, ast.FunctionDef | ast.AsyncFunctionDef
                ) and not member.name.startswith("_"):
                    api.append(
                        (
                            member.lineno,
                            f"{node.name}.{member.name}",
                            ast.get_docstring(member),
                        )
                    )
    return api


def test_public_api_is_fully_documented():
    undocumented = [
        f"{path.relative_to(ROOT)}:{lineno} {name}"
        for path in _source_files()
        for lineno, name, docstring in _public_api(path)
        if not docstring
    ]

    assert not undocumented, "Public API without a docstring:\n" + "\n".join(
        undocumented
    )


def test_docstring_sections_follow_numpy_style():
    foreign = [
        f"{path.relative_to(ROOT)}:{lineno} {name}"
        for path in _source_files()
        for lineno, name, docstring in _public_api(path)
        if docstring and FOREIGN_SECTION.search(docstring)
    ]

    assert not foreign, (
        "Docstring sections must be underlined with dashes, not followed by a "
        "colon:\n" + "\n".join(foreign)
    )


def test_every_module_with_public_api_is_in_the_api_reference():
    rendered: set[str] = set()
    for page in DOCS.rglob("*.md"):
        rendered.update(
            re.findall(r"^::: (\S+)", page.read_text(encoding="utf-8"), re.MULTILINE)
        )

    missing = [
        _module_name(path)
        for path in _source_files()
        if _public_api(path) and _module_name(path) not in rendered
    ]

    assert not missing, (
        "Modules with public API that the documentation never renders:\n"
        + "\n".join(missing)
    )

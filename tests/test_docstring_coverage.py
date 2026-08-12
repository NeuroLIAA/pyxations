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
SECTION = re.compile(
    r"^(?P<title>[A-Za-z][A-Za-z ]+)\n-{3,}\n(?P<body>.*?)(?=\n[A-Za-z][A-Za-z ]+\n-{3,}\n|\Z)",
    re.MULTILINE | re.DOTALL,
)
PARAMETER = re.compile(r"^(?P<names>\S[^\n:]*?)\s*:\s*\S.*$", re.MULTILINE)


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


def _public_nodes(path: Path):
    """Yield each public callable together with its qualified name."""

    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.name.startswith("_"):
                yield node, node.name
        elif isinstance(node, ast.ClassDef) and not node.name.startswith("_"):
            for member in node.body:
                if isinstance(
                    member, ast.FunctionDef | ast.AsyncFunctionDef
                ) and not member.name.startswith("_"):
                    yield member, f"{node.name}.{member.name}"


def _sections(docstring: str) -> dict[str, str]:
    return {
        match.group("title"): match.group("body").strip()
        for match in SECTION.finditer(docstring)
    }


def _parameters(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    parameters = [
        argument.arg
        for argument in (
            *node.args.posonlyargs,
            *node.args.args,
            *node.args.kwonlyargs,
        )
        if argument.arg not in {"self", "cls"}
    ]
    if node.args.vararg:
        parameters.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg:
        parameters.append(f"**{node.args.kwarg.arg}")
    return parameters


def _documented_parameters(body: str) -> set[str]:
    documented = set()
    for match in PARAMETER.finditer(body):
        documented.update(name.strip() for name in match.group("names").split(","))
    return documented


class _DirectReturnVisitor(ast.NodeVisitor):
    """Find value returns without descending into nested callables."""

    def __init__(self):
        self.has_value_return = False

    def visit_Return(self, node):
        if node.value is not None and not (
            isinstance(node.value, ast.Constant) and node.value.value is None
        ):
            self.has_value_return = True

    def visit_FunctionDef(self, node):
        pass

    def visit_AsyncFunctionDef(self, node):
        pass

    def visit_Lambda(self, node):
        pass

    def visit_ClassDef(self, node):
        pass


def _has_value_return(node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    visitor = _DirectReturnVisitor()
    for statement in node.body:
        visitor.visit(statement)
    return visitor.has_value_return


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


def test_public_parameters_are_documented():
    problems = []
    for path in _source_files():
        for node, name in _public_nodes(path):
            expected = set(_parameters(node))
            if not expected:
                continue
            sections = _sections(ast.get_docstring(node) or "")
            documented = _documented_parameters(sections.get("Parameters", ""))
            if expected != documented:
                problems.append(
                    f"{path.relative_to(ROOT)}:{node.lineno} {name}: "
                    f"expected {sorted(expected)}, documented {sorted(documented)}"
                )

    assert not problems, "Public parameters are not fully documented:\n" + "\n".join(
        problems
    )


def test_public_return_sections_match_implementation():
    problems = []
    for path in _source_files():
        for node, name in _public_nodes(path):
            sections = _sections(ast.get_docstring(node) or "")
            documented = "Returns" in sections or "Yields" in sections
            implemented = _has_value_return(node)
            if documented != implemented:
                problems.append(
                    f"{path.relative_to(ROOT)}:{node.lineno} {name}: "
                    f"value return={implemented}, return section={documented}"
                )

    assert not problems, "Public return contracts do not match the code:\n" + "\n".join(
        problems
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

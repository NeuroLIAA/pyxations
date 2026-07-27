"""Tests for installation requirement and optional-feature policy."""

from __future__ import annotations

import ast
from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
OPTIONAL_IMPORTS = {"cv2", "multimatch_gaze", "remodnav"}


def _project_metadata() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]


def _package_name(requirement: str) -> str:
    for delimiter in (">=", "==", "<=", "~=", ">", "<", "!=", "["):
        requirement = requirement.split(delimiter, 1)[0]
    return requirement.strip().lower()


def _top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    return imported


def test_base_runtime_dependencies_are_minimal():
    dependencies = _project_metadata()["dependencies"]

    assert {_package_name(item) for item in dependencies} == {
        "matplotlib",
        "numpy",
        "polars",
    }


def test_runtime_dependencies_are_not_exactly_pinned():
    dependencies = _project_metadata()["dependencies"]

    assert dependencies
    assert all("==" not in requirement for requirement in dependencies)
    assert all(">=" in requirement for requirement in dependencies)


def test_feature_dependencies_have_dedicated_extras():
    extras = _project_metadata()["optional-dependencies"]

    assert {_package_name(item) for item in extras["remodnav"]} == {"remodnav"}
    assert {_package_name(item) for item in extras["multimatch"]} == {
        "multimatch-gaze"
    }
    assert {_package_name(item) for item in extras["video"]} == {
        "opencv-python"
    }


def test_all_extra_is_union_of_feature_extras():
    extras = _project_metadata()["optional-dependencies"]
    feature_packages = {
        _package_name(item)
        for group in ("remodnav", "multimatch", "video")
        for item in extras[group]
    }

    assert {_package_name(item) for item in extras["all"]} == feature_packages
    assert feature_packages <= {_package_name(item) for item in extras["dev"]}


def test_base_import_boundaries_do_not_eagerly_import_optional_packages():
    base_import_modules = (
        ROOT / "pyxations" / "__init__.py",
        ROOT / "pyxations" / "bids_formatting.py",
        ROOT / "pyxations" / "analysis" / "generic.py",
        ROOT / "pyxations" / "visualization" / "visualization.py",
    )

    for module_path in base_import_modules:
        assert not (_top_level_imports(module_path) & OPTIONAL_IMPORTS), module_path

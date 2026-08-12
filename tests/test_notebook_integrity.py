"""Repository-level checks for the executable documentation notebooks."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TUTORIALS = ROOT / "docs" / "tutorials"
PUBLISHED = {
    "eyelink_example.ipynb",
    "gazepoint_example.ipynb",
    "multimatch_example.ipynb",
    "tobii_example.ipynb",
    "webgazer_example.ipynb",
}


def _notebooks():
    return sorted(TUTORIALS.glob("*.ipynb"))


def test_tutorials_have_one_cross_platform_canonical_location():
    assert TUTORIALS.is_dir()
    assert not TUTORIALS.is_symlink()
    assert not (ROOT / "notebooks").exists()
    assert PUBLISHED <= {path.name for path in _notebooks()}


def test_notebooks_have_stable_ids_clean_outputs_and_generic_kernel_metadata():
    problems = []
    for path in _notebooks():
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = notebook["cells"]
        cell_ids = [cell.get("id") for cell in cells]

        if notebook.get("nbformat") != 4 or notebook.get("nbformat_minor", 0) < 5:
            problems.append(f"{path.name}: expected notebook format 4.5 or newer")
        if any(not cell_id for cell_id in cell_ids):
            problems.append(f"{path.name}: one or more cells have no ID")
        if len(cell_ids) != len(set(cell_ids)):
            problems.append(f"{path.name}: cell IDs are not unique")
        if any(cell.get("outputs") for cell in cells if cell["cell_type"] == "code"):
            problems.append(f"{path.name}: generated outputs are committed")

        metadata = notebook.get("metadata", {})
        kernel = metadata.get("kernelspec", {})
        language = metadata.get("language_info", {})
        if kernel.get("name") != "python3" or kernel.get("display_name") != "Python 3":
            problems.append(f"{path.name}: kernel metadata is environment-specific")
        if language != {"name": "python", "version": "3"}:
            problems.append(f"{path.name}: language metadata is environment-specific")

    assert not problems, "Notebook normalization problems:\n" + "\n".join(problems)

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import polars as pl
import pytest

matplotlib.use("Agg")

from pyxations.visualization.samples import SampleVisualization


def _samples() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "tSample": [0.0, 10.0, 20.0],
            "X": [0.10, 0.25, 0.50],
            "Y": [0.20, 0.40, 0.60],
        }
    )


def test_sample_visualization_requires_polars_dataframe():
    with pytest.raises(TypeError, match="polars.DataFrame"):
        SampleVisualization({"X": [1.0], "Y": [2.0]})


def test_sample_visualization_validates_screen_dimensions():
    frame = _samples()

    with pytest.raises(ValueError, match="screen_width"):
        SampleVisualization(frame, screen_width=0)
    with pytest.raises(ValueError, match="screen_height"):
        SampleVisualization(frame, screen_height=np.inf)


def test_gaze_arrays_are_numpy_and_scale_percent_coordinates():
    visualizer = SampleVisualization(_samples(), screen_width=1000, screen_height=500)

    x, y = visualizer._gaze_arrays(in_percent=True)

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    np.testing.assert_allclose(x, [100.0, 250.0, 500.0])
    np.testing.assert_allclose(y, [100.0, 200.0, 300.0])


def test_plot_writes_png_from_polars_samples(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    visualizer = SampleVisualization(_samples(), screen_width=1000, screen_height=500)

    visualizer.plot(display=False, scanpath_file_name="scanpath", in_percent=True)

    assert (tmp_path / "scanpath.png").is_file()


def test_plot_rejects_missing_or_non_numeric_columns():
    missing_time = SampleVisualization(pl.DataFrame({"X": [1.0], "Y": [2.0]}))
    with pytest.raises(ValueError, match="tSample"):
        missing_time.plot(display=False)

    non_numeric = SampleVisualization(
        pl.DataFrame({"tSample": [0], "X": ["left"], "Y": [2.0]})
    )
    with pytest.raises(TypeError, match="'X'.*numeric"):
        non_numeric.plot(display=False)


def test_empty_samples_are_rejected():
    visualizer = SampleVisualization(
        pl.DataFrame(schema={"tSample": pl.Float64, "X": pl.Float64, "Y": pl.Float64})
    )

    with pytest.raises(ValueError, match="At least one gaze sample"):
        visualizer.plot(display=False)


def test_animation_uses_one_frame_per_sample_and_writes_requested_path(
    tmp_path, monkeypatch
):
    captured: dict[str, object] = {}

    class FakeAnimation:
        def __init__(self, *, fig, func, frames, interval):
            captured["frames"] = list(frames)
            captured["interval"] = interval
            # Exercise the last update to ensure NumPy arrays are accepted.
            func(captured["frames"][-1])

        def save(self, *, filename, writer):
            captured["filename"] = Path(filename)
            captured["writer"] = writer
            Path(filename).write_bytes(b"GIF89a")

    monkeypatch.setattr(
        "pyxations.visualization.samples.animation.FuncAnimation", FakeAnimation
    )
    output = tmp_path / "gaze.gif"
    visualizer = SampleVisualization(_samples())

    visualizer.animate(display=False, out_file=output)

    assert captured["frames"] == [1, 2, 3]
    assert captured["interval"] == 1
    assert captured["writer"] == "pillow"
    assert captured["filename"] == output
    assert output.is_file()

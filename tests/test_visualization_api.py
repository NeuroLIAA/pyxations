import sys
import types
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

import pyxations.visualization.visualization as visualization_module
from pyxations.visualization.visualization import Visualization


def _plot_tables():
    fixations = pl.DataFrame(
        {
            "trial_number": [0, 0],
            "phase": ["search", "memory"],
            "tStart": [0.0, 100.0],
            "duration": [50.0, 50.0],
            "xAvg": [10.0, 20.0],
            "yAvg": [15.0, 25.0],
        }
    )
    saccades = pl.DataFrame(
        {
            "trial_number": [0, 0],
            "phase": ["search", "memory"],
            "tStart": [50.0, 150.0],
            "ampDeg": [2.0, 3.0],
            "vPeak": [100.0, 120.0],
            "deg": [0.0, 90.0],
            "dir": ["right", "down"],
        }
    )
    samples = pl.DataFrame(
        {
            "trial_number": [0, 0, 0, 0],
            "phase": ["search", "search", "memory", "memory"],
            "tSample": [0.0, 50.0, 100.0, 150.0],
            "X": [10.0, 12.0, 20.0, 22.0],
            "Y": [15.0, 17.0, 25.0, 27.0],
        }
    )
    return fixations, saccades, samples


def test_scanpath_handles_phases_images_and_validation(tmp_path):
    fixations, saccades, samples = _plot_tables()
    image_path = tmp_path / "stimulus.png"
    plt.imsave(image_path, np.zeros((4, 4, 3), dtype=np.uint8))
    output = tmp_path / "nested" / "plots"
    visualization = Visualization(tmp_path, "remodnav")

    visualization.scanpath(
        fixations,
        screen_height=100,
        screen_width=100,
        folder_path=output,
        saccades=saccades,
        samples=samples,
        phase_data={
            "search": {
                "img_paths": [image_path],
                "img_plot_coords": [(0, 0, 50, 50)],
                "bbox": (5, 5, 15, 15),
            }
        },
        display=False,
    )

    assert (output / "scanpath_0_search.png").exists()
    assert (output / "scanpath_0_memory.png").exists()

    visualization.scanpath(
        fixations.head(0),
        screen_height=100,
        screen_width=100,
        display=False,
    )
    with pytest.raises(ValueError, match="provided together"):
        visualization.scanpath(
            fixations,
            screen_height=100,
            screen_width=100,
            tmin=0,
        )
    with pytest.raises(ValueError, match="required columns"):
        visualization.scanpath(
            fixations.drop("xAvg"),
            screen_height=100,
            screen_width=100,
        )


def test_scanpath_plots_unsegmented_fixations_and_warns(tmp_path):
    """Recordings with no named phase must still produce a figure."""
    fixations, saccades, samples = _plot_tables()
    unphased = {"phase": pl.lit("")}
    output = tmp_path / "plots"
    visualization = Visualization(tmp_path, "remodnav")

    with pytest.warns(UserWarning, match="No named trial phase"):
        visualization.scanpath(
            fixations.with_columns(**unphased),
            screen_height=100,
            screen_width=100,
            folder_path=output,
            saccades=saccades.with_columns(**unphased),
            samples=samples.with_columns(**unphased),
            display=False,
        )

    # The rows are kept rather than filtered away, so a figure exists.
    assert (output / "scanpath_0_unphased.png").exists()


def test_scanpath_still_drops_unphased_rows_when_phases_exist(tmp_path):
    """Rows between trials are skipped whenever the recording was segmented."""
    fixations, _, _ = _plot_tables()
    between_trials = fixations.head(1).with_columns(phase=pl.lit(""))
    output = tmp_path / "plots"
    visualization = Visualization(tmp_path, "remodnav")

    visualization.scanpath(
        pl.concat([fixations, between_trials]),
        screen_height=100,
        screen_width=100,
        folder_path=output,
        display=False,
    )

    assert (output / "scanpath_0_search.png").exists()
    assert (output / "scanpath_0_memory.png").exists()
    assert not (output / "scanpath_0_unphased.png").exists()


def test_scanpath_handles_more_than_255_fixations(tmp_path):
    """Fixation labels must not be cast through an eight-bit colour index."""
    count = 300
    fixations = pl.DataFrame(
        {
            "trial_number": [0] * count,
            "phase": ["search"] * count,
            "tStart": np.arange(count, dtype=float) * 10,
            "duration": [5.0] * count,
            "xAvg": np.linspace(0, 99, count),
            "yAvg": np.linspace(99, 0, count),
        }
    )
    output = tmp_path / "plots"

    Visualization(tmp_path, "remodnav").scanpath(
        fixations,
        screen_height=100,
        screen_width=100,
        folder_path=output,
        display=False,
    )

    assert (output / "scanpath_0_search.png").exists()


def test_visualization_summary_plots_and_multipanel(tmp_path):
    fixations, saccades, _ = _plot_tables()
    visualization = Visualization(tmp_path, "engbert")
    fig, axes = plt.subplots(2, 2)

    visualization.fix_duration(fixations, axs=axes[0, 0])
    visualization.sacc_amplitude(saccades, axs=axes[0, 1])
    visualization.sacc_main_sequence(saccades, axs=axes[1, 1], hline=80)
    visualization.sacc_direction(saccades, axs=axes[1, 0], figs=fig)
    plt.close(fig)

    with pytest.raises(ValueError, match="saccades direction"):
        visualization.sacc_direction(saccades.drop("deg"))

    visualization.plot_multipanel(fixations, saccades, display=False)
    assert (tmp_path / "engbert" / "multipanel_search.png").exists()
    assert (tmp_path / "engbert" / "multipanel_memory.png").exists()


def test_animation_without_video_and_input_validation(tmp_path, monkeypatch):
    monkeypatch.setattr("matplotlib.pyplot.show", lambda: None)
    visualization = Visualization(tmp_path, "remodnav")
    samples = pl.DataFrame(
        {
            "trial_number": [2, 2, 2],
            "tSample": [0.0, 500.0, 1000.0],
            "X": [10.0, 20.0, 30.0],
            "Y": [15.0, 25.0, 35.0],
        }
    )

    result = visualization.plot_animation(
        samples,
        screen_height=100,
        screen_width=100,
        folder_path=tmp_path,
        fps=2,
        output_format="html",
        display=False,
    )
    assert result is None
    assert (tmp_path / "animation_2.html").exists()

    assert (
        visualization.plot_animation(
            samples,
            screen_height=100,
            screen_width=100,
            fps=2,
            output_format="matplotlib",
            display=False,
        )
        is None
    )

    with pytest.raises(ValueError, match="output_format"):
        visualization.plot_animation(samples, 100, 100, output_format="avi")
    with pytest.raises(ValueError, match="scale_factor"):
        visualization.plot_animation(samples, 100, 100, scale_factor=0)
    with pytest.raises(ValueError, match="fps"):
        visualization.plot_animation(samples, 100, 100, fps=0)
    with pytest.raises(ValueError, match="provided together"):
        visualization.plot_animation(samples, 100, 100, tmin=0)
    with pytest.raises(ValueError, match="gaze columns"):
        visualization.plot_animation(
            samples.drop(["X", "Y"]).with_columns(pl.lit(1).alias("other")),
            100,
            100,
        )
    with pytest.raises(ValueError, match="finite gaze"):
        visualization.plot_animation(
            samples.with_columns(
                pl.lit(float("nan")).alias("X"),
                pl.lit(float("nan")).alias("Y"),
            ),
            100,
            100,
        )
    with pytest.raises(FileNotFoundError, match="Background image"):
        visualization.plot_animation(
            samples,
            100,
            100,
            background_image_path=tmp_path / "missing.png",
        )


class _FakeAnimation:
    fail_with = None
    saves: ClassVar[list] = []

    def __init__(self, fig, func, frames, interval, **kwargs):
        self.fig = fig
        self.func = func
        self.frames = list(range(frames)) if isinstance(frames, int) else list(frames)
        self.interval = interval
        func(self.frames[0])
        func(self.frames[-1])

    def save(self, path, writer, fps):
        if self.fail_with is not None:
            raise self.fail_with("writer failed")
        self.saves.append((path, writer, fps))

    def to_jshtml(self):
        return "<div>animation</div>"


class _FakeCapture:
    def __init__(self, path, *, readable=True):
        self.path = path
        self.readable = readable
        self.read_count = 0
        self.released = False
        self.position = None

    def get(self, prop):
        import cv2

        if prop == cv2.CAP_PROP_FPS:
            return 20.0
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return 2
        return 0

    def set(self, prop, value):
        self.position = (prop, value)

    def read(self):
        self.read_count += 1
        if not self.readable or self.read_count > 2:
            return False, None
        return True, np.zeros((10, 10, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def test_animation_video_and_save_formats(tmp_path, monkeypatch):
    import cv2
    import matplotlib.animation

    _FakeAnimation.fail_with = None
    _FakeAnimation.saves = []
    captures = []

    def capture(path):
        result = _FakeCapture(path)
        captures.append(result)
        return result

    monkeypatch.setattr(matplotlib.animation, "FuncAnimation", _FakeAnimation)
    monkeypatch.setattr(cv2, "VideoCapture", capture)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    samples = pl.DataFrame(
        {
            "trial_number": [2.0, 2.0],
            "tSample": [0.0, 1000.0],
            "RX": [10.0, 500.0],
            "RY": [10.0, 500.0],
        }
    )
    visualization = Visualization(tmp_path, "remodnav")

    visualization.plot_animation(
        samples,
        100,
        100,
        video_path=video,
        folder_path=tmp_path,
        seconds_to_show=0.05,
        output_format="mp4",
        display=False,
    )

    assert _FakeAnimation.saves[-1][1:] == ("ffmpeg", 20.0)
    assert _FakeAnimation.saves[-1][0].endswith("animation_2.mp4")
    assert captures[-1].released

    background = tmp_path / "background.png"
    background.write_bytes(b"placeholder")
    monkeypatch.setattr(
        visualization_module.mpimg,
        "imread",
        lambda path: np.ones((2, 2, 3), dtype=np.float64),
    )
    visualization.plot_animation(
        samples.rename({"RX": "LX", "RY": "LY"}),
        100,
        100,
        background_image_path=background,
        folder_path=tmp_path,
        tmin=0,
        tmax=1000,
        seconds_to_show=0.1,
        fps=2,
        output_format="gif",
        display=False,
    )
    assert _FakeAnimation.saves[-1][1:] == ("pillow", 2)
    assert _FakeAnimation.saves[-1][0].endswith("animation_2_0_1000.gif")


def test_animation_video_and_writer_failures(tmp_path, monkeypatch):
    import cv2
    import matplotlib.animation

    monkeypatch.setattr(matplotlib.animation, "FuncAnimation", _FakeAnimation)
    video = tmp_path / "video.mp4"
    video.write_bytes(b"placeholder")
    samples = pl.DataFrame({"tSample": [0.0, 100.0], "X": [1.0, 2.0], "Y": [1.0, 2.0]})
    visualization = Visualization(tmp_path, "remodnav")

    monkeypatch.setattr(
        cv2,
        "VideoCapture",
        lambda path: _FakeCapture(path, readable=False),
    )
    with pytest.raises(RuntimeError, match="first frame"):
        visualization.plot_animation(samples, 100, 100, video_path=video)

    monkeypatch.setattr(cv2, "VideoCapture", lambda path: _FakeCapture(path))
    _FakeAnimation.fail_with = OSError
    with pytest.raises(RuntimeError, match="Failed to save MP4"):
        visualization.plot_animation(
            samples,
            100,
            100,
            folder_path=tmp_path,
            output_format="mp4",
            display=False,
        )
    with pytest.raises(RuntimeError, match="Failed to save GIF"):
        visualization.plot_animation(
            samples,
            100,
            100,
            folder_path=tmp_path,
            output_format="gif",
            display=False,
        )
    _FakeAnimation.fail_with = None

    with pytest.raises(FileNotFoundError, match="Video file"):
        visualization.plot_animation(
            samples,
            100,
            100,
            video_path=tmp_path / "missing.mp4",
        )
    with pytest.raises(ValueError, match="No samples"):
        visualization.plot_animation(samples, 100, 100, tmin=200, tmax=300)


def test_animation_optional_dependency_and_html_display(tmp_path, monkeypatch):
    import matplotlib.animation

    monkeypatch.setattr(matplotlib.animation, "FuncAnimation", _FakeAnimation)
    display_module = types.ModuleType("IPython.display")
    display_module.HTML = lambda content: {"html": content}
    ipython_module = types.ModuleType("IPython")
    ipython_module.display = display_module
    monkeypatch.setitem(sys.modules, "IPython", ipython_module)
    monkeypatch.setitem(sys.modules, "IPython.display", display_module)
    samples = pl.DataFrame({"tSample": [0.0, 100.0], "X": [1.0, 2.0], "Y": [1.0, 2.0]})
    visualization = Visualization(tmp_path, "remodnav")

    result = visualization.plot_animation(
        samples,
        100,
        100,
        output_format="html",
        display=True,
    )
    assert result == {"html": "<div>animation</div>"}

    def missing_cv2(*args):
        error = ModuleNotFoundError("No module named 'cv2'")
        error.name = "cv2"
        raise error

    monkeypatch.setattr(visualization_module, "import_module", missing_cv2)
    with pytest.raises(ImportError, match=r"pyxations\[video\]"):
        visualization.plot_animation(samples, 100, 100)

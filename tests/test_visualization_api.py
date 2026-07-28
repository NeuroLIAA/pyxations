import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pytest

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
    pytest.importorskip("cv2")
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

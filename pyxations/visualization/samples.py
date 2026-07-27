"""Visualizations for sample-level gaze data."""

from __future__ import annotations

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import polars as pl


class SampleVisualization:
    """Plot and animate sample-level gaze coordinates.

    Parameters
    ----------
    samples_df
        Polars dataframe containing gaze-coordinate columns. ``X`` and ``Y``
        are required by both plotting methods; ``tSample`` is additionally
        required by :meth:`plot`.
    screen_width
        Screen width in pixels.
    screen_height
        Screen height in pixels.
    """

    def __init__(
        self,
        samples_df: pl.DataFrame,
        screen_width: float = 1366,
        screen_height: float = 768,
    ) -> None:
        if not isinstance(samples_df, pl.DataFrame):
            raise TypeError(
                "SampleVisualization requires a polars.DataFrame; "
                f"received {type(samples_df).__name__}."
            )
        if not np.isfinite(screen_width) or screen_width <= 0:
            raise ValueError("screen_width must be a positive finite number.")
        if not np.isfinite(screen_height) or screen_height <= 0:
            raise ValueError("screen_height must be a positive finite number.")

        self.samples = samples_df
        self.screen_width = float(screen_width)
        self.screen_height = float(screen_height)

    def _numeric_column(self, name: str) -> np.ndarray:
        """Return one dataframe column as a one-dimensional float array."""
        if name not in self.samples.columns:
            raise ValueError(
                f"Sample dataframe is missing required column {name!r}."
            )

        try:
            values = self.samples.get_column(name).cast(pl.Float64, strict=True)
        except Exception as exc:
            raise TypeError(
                f"Sample column {name!r} must contain numeric values."
            ) from exc

        array = np.asarray(values.to_numpy(), dtype=float)
        if array.ndim != 1:
            raise ValueError(f"Sample column {name!r} must be one-dimensional.")
        return array

    def _gaze_arrays(self, *, in_percent: bool) -> tuple[np.ndarray, np.ndarray]:
        """Return gaze coordinates in pixels as NumPy arrays."""
        x = self._numeric_column("X")
        y = self._numeric_column("Y")

        if in_percent:
            x = x * self.screen_width
            y = y * self.screen_height

        return x, y

    @staticmethod
    def _require_samples(x: np.ndarray, y: np.ndarray) -> None:
        if x.size == 0 or y.size == 0:
            raise ValueError("At least one gaze sample is required.")

    def plot(
        self,
        display: bool = True,
        scanpath_file_name: str | Path = "scanpath",
        in_percent: bool = False,
    ) -> None:
        """Save a scanpath and gaze-over-time plot as a PNG image."""
        x, y = self._gaze_arrays(in_percent=in_percent)
        self._require_samples(x, y)
        timestamps = self._numeric_column("tSample")

        fig, axs = plt.subplots(
            nrows=2,
            ncols=1,
            height_ratios=(4, 1),
            figsize=(10, 6),
        )
        ax_main = axs[0]
        ax_gaze = axs[1]

        ax_main.set_xlim(0, self.screen_width)
        ax_main.set_ylim(0, self.screen_height)
        ax_main.plot(x, y, "--", color="C0", zorder=1)

        ax_gaze.plot(timestamps, x, label="X")
        ax_gaze.plot(timestamps, y, label="Y")
        ax_gaze.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        ax_gaze.set_ylabel("Gaze")
        ax_gaze.set_xlabel("Time [ms]")

        plt.tight_layout()
        file_path = Path(f"{scanpath_file_name}.png")
        fig.savefig(file_path)
        if display:
            plt.show()
        plt.close(fig)

    def animate(
        self,
        display: bool = True,
        in_percent: bool = False,
        out_file: str | Path = "output.gif",
    ) -> None:
        """Save an animated gaze trace as a GIF image."""
        x, y = self._gaze_arrays(in_percent=in_percent)
        self._require_samples(x, y)

        fig, ax = plt.subplots()
        ax.set_xlim(0, self.screen_width)
        ax.set_ylim(0, self.screen_height)

        scat = ax.scatter(x[0], y[0], c="b", s=5, label="a")
        line = ax.plot(x[0], y[0], label="b")[0]
        ax.legend()

        def update(frame: int):
            # ``frame`` is a sample count, not a zero-based sample index.
            x_frame = x[:frame]
            y_frame = y[:frame]
            scat.set_offsets(np.column_stack((x_frame, y_frame)))
            line.set_xdata(x_frame)
            line.set_ydata(y_frame)
            return scat, line

        gaze_animation = animation.FuncAnimation(
            fig=fig,
            func=update,
            frames=range(1, len(x) + 1),
            interval=1,
        )
        if display:
            plt.show()

        gaze_animation.save(filename=Path(out_file), writer="pillow")
        plt.close(fig)

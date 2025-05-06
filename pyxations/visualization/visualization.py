import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mplcolors
import polars as pl
from pyxations.bids_formatting import EYE_MOVEMENT_DETECTION_DICT
from pathlib import Path


class Visualization():
    def __init__(self, derivatives_folder_path,events_detection_algorithm):
        self.derivatives_folder_path = Path(derivatives_folder_path)
        if events_detection_algorithm not in EYE_MOVEMENT_DETECTION_DICT and events_detection_algorithm != 'eyelink':
            raise ValueError(f"Detection algorithm {events_detection_algorithm} not found.")
        self.events_detection_folder = Path(events_detection_algorithm+'_events')

    def scanpath(
        self,
        fixations: pl.DataFrame,
        screen_height: int,
        screen_width: int,
        folder_path: str | Path | None = None,
        tmin: int | None = None,
        tmax: int | None = None,
        saccades: pl.DataFrame | None = None,
        samples: pl.DataFrame | None = None,
        phase_data: dict[str, dict] | None = None,
        display: bool = True,
    ):
        """
        Fast scan‑path visualiser.

        • **Vectorised**: no per‑row Python loops  
        • **Single pass** phase grouping  
        • Uses `BrokenBarHCollection` for fixation spans  
        • Optional asynchronous PNG write via ThreadPoolExecutor (drop‑in‑ready, see comment)

        Parameters
        ----------
        fixations
            Polars DataFrame with at least `tStart`, `duration`, `xAvg`, `yAvg`, `phase`.
        screen_height, screen_width
            Stimulus resolution in pixels.
        folder_path
            Directory where 1 PNG per phase will be stored.  If *None*, nothing is saved.
        tmin, tmax
            Time window in **ms**.  If both `None`, the whole trial is plotted.
        saccades
            Polars DataFrame with `tStart`, `phase`, …  (optional).
        samples
            Polars DataFrame with gaze traces (`tSample`, `LX`, `LY`, `RX`, `RY` or
            `X`, `Y`) (optional).
        phase_data
            Per‑phase extras::

                {
                    "search": {
                        "img_paths": [...],
                        "img_plot_coords": [(x1,y1,x2,y2), ...],
                        "bbox": (x1,y1,x2,y2),
                    },
                    ...
                }

        display
            If *False* the figure canvas is never shown (faster for batch jobs).
        """


        # ------------- small helpers ------------------------------------------------
        def _make_axes(plot_samples: bool):
            if plot_samples:
                fig, (ax_main, ax_gaze) = plt.subplots(
                    2, 1, height_ratios=(4, 1), figsize=(10, 6), sharex=False
                )
            else:
                fig, ax_main = plt.subplots(figsize=(10, 6))
                ax_gaze = None
            ax_main.set_xlim(0, screen_width)
            ax_main.set_ylim(screen_height, 0)
            return fig, ax_main, ax_gaze

        def _maybe_cache_img(path: str):
            if path not in _img_cache:
                _img_cache[path] = mpimg.imread(path)
            return _img_cache[path]

        # ---------------------------------------------------------------------------
        plot_saccades = saccades is not None
        plot_samples = samples is not None
        _img_cache: dict[str, np.ndarray] = {}

        trial_idx = fixations["trial_number"][0]

        # ---- time filter ----------------------------------------------------------
        if tmin is not None and tmax is not None:
            fixations = fixations.filter(pl.col("tStart").is_between(tmin, tmax))
            if plot_saccades:
                saccades = saccades.filter(pl.col("tStart").is_between(tmin, tmax))
            if plot_samples:
                samples = samples.filter(pl.col("tSample").is_between(tmin, tmax))

        # remove empty phase markings
        fixations = fixations.filter(pl.col("phase") != "")
        if plot_saccades:
            saccades = saccades.filter(pl.col("phase") != "")
        if plot_samples:
            samples = samples.filter(pl.col("phase") != "")

        # ---- split once by phase --------------------------------------------------
        fix_by_phase = fixations.partition_by("phase", as_dict=True)
        sac_by_phase = (
            saccades.partition_by("phase", as_dict=True) if plot_saccades else {}
        )
        samp_by_phase = (
            samples.partition_by("phase", as_dict=True) if plot_samples else {}
        )

        # colour map shared across phases
        cmap = plt.cm.rainbow

        # ---- build & draw ---------------------------------------------------------
        # optional async saver (uncomment if you save hundreds of files)
        from concurrent.futures import ThreadPoolExecutor
        saver = ThreadPoolExecutor(max_workers=4) if folder_path else None

        if not display:
            plt.ioff()

        for phase, phase_fix in fix_by_phase.items():
            if phase_fix.is_empty():
                continue

            # ---------- vectors (zero‑copy) -----------------
            fx, fy, fdur = phase_fix.select(["xAvg", "yAvg", "duration"]).to_numpy().T
            n_fix = fx.size
            fix_idx = np.arange(1, n_fix + 1)

            norm = mplcolors.BoundaryNorm(np.arange(1, n_fix + 2), cmap.N)

            # saccades
            sac_t = (
                sac_by_phase[phase]["tStart"].to_numpy()
                if plot_saccades and phase in sac_by_phase
                else np.empty(0)
            )

            # samples
            if plot_samples and phase in samp_by_phase and samp_by_phase[phase].height:
                samp_phase = samp_by_phase[phase]
                t0 = samp_phase["tSample"][0]
                ts = (samp_phase["tSample"].to_numpy() - t0) 
                get = samp_phase.get_column
                lx = get("LX").to_numpy() if "LX" in samp_phase.columns else None
                ly = get("LY").to_numpy() if "LY" in samp_phase.columns else None
                rx = get("RX").to_numpy() if "RX" in samp_phase.columns else None
                ry = get("RY").to_numpy() if "RY" in samp_phase.columns else None
                gx = get("X").to_numpy() if "X" in samp_phase.columns else None
                gy = get("Y").to_numpy() if "Y" in samp_phase.columns else None
            else:
                t0 = None

            # ---------- figure -----------------------------
            fig, ax_main, ax_gaze = _make_axes(plot_samples and t0 is not None)
            # scatter fixations
            sc = ax_main.scatter(
                fx,
                fy,
                c=fix_idx,
                s=fdur,
                cmap=cmap,
                norm=norm,
                alpha=0.5,
                zorder=2,
            )
            fig.colorbar(
                sc,
                ax=ax_main,
                ticks=[1, n_fix // 2 if n_fix > 2 else n_fix, n_fix],
                fraction=0.046,
                pad=0.04,
            ).set_label("# of fixation")

            # ---------- stimulus imagery / bbox ------------
            if phase_data and phase[0] in phase_data:
                pdict = phase_data[phase[0]]
                coords = pdict.get("img_plot_coords") or []
                bbox = pdict.get('bbox',None) 
                for img_path, box in zip(pdict.get("img_paths", []), coords):

                    ax_main.imshow(_maybe_cache_img(img_path), extent=[box[0], box[2], box[3], box[1]], zorder=0)
                if bbox is not None:
                    x1, y1, x2, y2 = bbox
                    ax_main.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color='red', linewidth=1.5, zorder=3)

            # ---------- gaze traces ------------------------
            if ax_gaze is not None:
                if lx is not None:
                    ax_main.plot(lx, ly, "--", color="C0", zorder=1)
                    ax_gaze.plot(ts, lx, label="Left X")
                    ax_gaze.plot(ts, ly, label="Left Y")
                if rx is not None:
                    ax_main.plot(rx, ry, "--", color="k", zorder=1)
                    ax_gaze.plot(ts, rx, label="Right X")
                    ax_gaze.plot(ts, ry, label="Right Y")
                if gx is not None:
                    ax_main.plot(gx, gy, "--", color="k", zorder=1, alpha=0.6)
                    ax_gaze.plot(ts, gx, label="X")
                    ax_gaze.plot(ts, gy, label="Y")

                # fixation spans
                bars   = np.c_[phase_fix['tStart'].to_numpy() - t0,
                            phase_fix['duration'].to_numpy()]
                height = ax_gaze.get_ylim()[1] - ax_gaze.get_ylim()[0]
                colors = cmap(norm(fix_idx))

                # Draw all bars in one call; no BrokenBarHCollection import needed
                ax_gaze.broken_barh(bars, (0, height), facecolors=colors, alpha=0.4)
                # saccades
                if sac_t.size:
                    ymin, ymax = ax_gaze.get_ylim()
                    ax_gaze.vlines(
                        sac_t - t0,
                        ymin,
                        ymax,
                        colors="red",
                        linestyles="--",
                        linewidth=0.8,
                    )

                # tidy gaze axis
                h, l = ax_gaze.get_legend_handles_labels()
                by_label = {lab: hdl for hdl, lab in zip(h, l)}
                ax_gaze.legend(
                    by_label.values(),
                    by_label.keys(),
                    loc="center left",
                    bbox_to_anchor=(1, 0.5),
                )
                ax_gaze.set_ylabel("Gaze")
                ax_gaze.set_xlabel("Time [s]")

            fig.tight_layout()

            # ---------- save / show ------------------------
            if folder_path:
                scan_name = f"scanpath_{trial_idx}"
                if tmin is not None and tmax is not None:
                    scan_name += f"_{tmin}_{tmax}"
                out = Path(folder_path) / f"{scan_name}_{phase[0]}.png"
                fig.savefig(out, dpi=150)
                if saver:  saver.submit(fig.savefig, out, dpi=150)

            if display:
                plt.show()
            plt.close(fig)

        if not display:
            plt.ion()


    def fix_duration(self,fixations:pl.DataFrame,axs=None):
        
        ax = axs
        if ax is None:
            fig, ax = plt.subplots()

        ax.hist(fixations.select(pl.col('duration')).to_numpy().ravel(), bins=100, edgecolor='black', linewidth=1.2, density=True)
        ax.set_title('Fixation duration')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Density')


    def sacc_amplitude(self,saccades:pl.DataFrame,axs=None):

        ax = axs
        if ax is None:
            fig, ax = plt.subplots()

        saccades_amp = saccades.select(pl.col('ampDeg')).to_numpy().ravel()
        ax.hist(saccades_amp, bins=100, range=(0, 20), edgecolor='black', linewidth=1.2, density=True)
        ax.set_title('Saccades amplitude')
        ax.set_xlabel('Amplitude (deg)')
        ax.set_ylabel('Density')


    def sacc_direction(self,saccades:pl.DataFrame,axs=None,figs=None):

        ax = axs
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(polar=True)
        else:
            ax.set_axis_off()
            ax = figs.add_subplot(2, 2, 3, projection='polar')
        if 'deg' not in saccades.columns or 'dir' not in saccades.columns:
            raise ValueError('Compute saccades direction first by using saccades_direction function from the PreProcessing module.')
        # Convert from deg to rad
        saccades_rad = saccades.select(pl.col('deg')).to_numpy().ravel() * np.pi / 180

        n_bins = 24
        ang_hist, bin_edges = np.histogram(saccades_rad, bins=24, density=True)
        bin_centers = [np.mean((bin_edges[i], bin_edges[i+1])) for i in range(len(bin_edges) - 1)]

        bars = ax.bar(bin_centers, ang_hist, width=2*np.pi/n_bins, bottom=0.0, alpha=0.4, edgecolor='black')
        ax.set_title('Saccades direction')
        ax.set_yticklabels([])

        for r, bar in zip(ang_hist, bars):
            bar.set_facecolor(plt.cm.Blues(r / np.max(ang_hist)))


    def sacc_main_sequence(self,saccades:pl.DataFrame,axs=None, hline=None):

        ax = axs
        if ax is None:
            fig, ax = plt.subplots()
        # Logarithmic bins
        XL = np.log10(25)  # Adjusted to fit the xlim
        YL = np.log10(1000)  # Adjusted to fit the ylim

        saccades_peak_vel = saccades.select(pl.col('vPeak')).to_numpy().ravel()
        saccades_amp = saccades.select(pl.col('ampDeg')).to_numpy().ravel()

        # Create a 2D histogram with logarithmic bins
        ax.hist2d(saccades_amp, saccades_peak_vel, bins=[np.logspace(-1, XL, 50), np.logspace(0, YL, 50)])

        if hline:
            ax.hlines(y=hline, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors='grey', linestyles='--', label=hline)
            ax.legend()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title('Main sequence')
        ax.set_xlabel('Amplitude (deg)')
        ax.set_ylabel('Peak velocity (deg)')
         # Set the limits of the axes
        ax.set_xlim(0.1, 25)
        ax.set_ylim(10, 1000)
        ax.set_aspect('equal')


    def plot_multipanel(
            self,
            fixations: pl.DataFrame,
            saccades: pl.DataFrame,
            display: bool = True
        ) -> None:
        """
        Create a 2×2 multi‑panel diagnostic plot for every non‑empty
        phase label and save it as PNG in
        <derivatives_folder_path>/<events_detection_folder>/plots/.
        """
        # ── paths & matplotlib style ────────────────────────────────
        folder_path: Path = (
            self.derivatives_folder_path
            / self.events_detection_folder
            / "plots"
        )
        folder_path.mkdir(parents=True, exist_ok=True)
        plt.rcParams.update({"font.size": 12})

        # ── drop practice / invalid trials ─────────────────────────
        fixations = fixations.filter(pl.col("trial_number") != -1)
        saccades  = saccades.filter(pl.col("trial_number") != -1)

        # ── collect valid phase labels (skip empty string) ─────────
        phases = (
            fixations
            .select(pl.col("phase").filter(pl.col("phase") != ""))
            .unique()           # unique values in this Series
            .to_series()
            .to_list()          # plain Python list of strings
        )

        # ── one figure per phase ───────────────────────────────────
        for phase in phases:
            fix_phase   = fixations.filter(pl.col("phase") == phase)
            sacc_phase  = saccades.filter(pl.col("phase") == phase)

            fig, axs = plt.subplots(2, 2, figsize=(12, 7))

            self.fix_duration(fix_phase , axs=axs[0, 0])
            self.sacc_main_sequence(sacc_phase, axs=axs[1, 1])
            self.sacc_direction(sacc_phase, axs=axs[1, 0], figs=fig)
            self.sacc_amplitude(sacc_phase, axs=axs[0, 1])

            fig.tight_layout()
            plt.savefig(folder_path / f"multipanel_{phase}.png")
            if display:
                plt.show()
            plt.close()
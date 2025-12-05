import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mplcolors
import polars as pl
from pyxations.bids_formatting import EYE_MOVEMENT_DETECTION_DICT
from pathlib import Path
from collections import OrderedDict

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

        def _maybe_cache_img(path):
            """Load image from disk with a small LRU cache."""

            # Cache hit: move to the end (most recently used)
            if path in _img_cache:
                img = _img_cache.pop(path)
                _img_cache[path] = img
                return img

            # Cache miss: load image
            img = mpimg.imread(path)

            # Optional: reduce memory if image is float64 in [0, 1]
            if isinstance(img, np.ndarray) and img.dtype == np.float64:
                img = (img * 255).clip(0, 255).astype(np.uint8)

            # Insert into cache
            _img_cache[path] = img

            # If cache too big, drop least recently used item
            if len(_img_cache) > _MAX_CACHE_ITEMS:
                _img_cache.popitem(last=False)  # pops the oldest inserted item

            return img

        # ---------------------------------------------------------------------------
        plot_saccades = saccades is not None
        plot_samples = samples is not None
        _img_cache = OrderedDict()
        _MAX_CACHE_ITEMS = 8  # or 5, 10, etc. Tune as you like.

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
    
    def plot_animation(
        self,
        samples: pl.DataFrame,
        screen_height: int,
        screen_width: int,
        video_path: str | Path | None = None,
        background_image_path: str | Path | None = None,
        folder_path: str | Path | None = None,
        tmin: int | None = None,
        tmax: int | None = None,
        seconds_to_show: float | None = None,
        scale_factor: float = 0.5,
        gaze_radius: int = 10,
        gaze_color: tuple = (255, 0, 0),
        fps: float | None = None,
        output_format: str = "matplotlib",
        display: bool = True,
    ):
        """
        Create an animated visualization of eye-tracking data.

        When a video is provided, the animation syncs gaze samples with video frames.
        When no video is provided, gaze points are animated on a grey background or
        a provided background image, using the sample timestamps for timing.

        Parameters
        ----------
        samples
            Polars DataFrame with gaze samples. Must contain 'tSample' and gaze
            position columns ('X', 'Y' or 'LX', 'LY', 'RX', 'RY').
        screen_height, screen_width
            Stimulus resolution in pixels.
        video_path
            Path to a video file. If provided, gaze is overlaid on video frames.
        background_image_path
            Path to a background image. Only used when video_path is None.
            If both are None, a grey background is used.
        folder_path
            Directory where the animation will be saved. If None, nothing is saved.
            The file format depends on `output_format`.
        tmin, tmax
            Time window in **ms**. If both None, the whole trial is plotted.
        seconds_to_show
            Limit the animation to the first N seconds. If None, shows all available data.
        scale_factor
            Resolution scaling factor (1.0 = original, 0.5 = half resolution).
        gaze_radius
            Radius of the gaze point circle in pixels (before scaling).
        gaze_color
            RGB tuple for gaze point color.
        fps
            Frames per second for the animation. If None:
            - With video: uses the video's native FPS
            - Without video: defaults to 60 FPS
        output_format
            Output format for saved animations:
            - "html": Interactive HTML file (default, works in browsers)
            - "mp4": Video file (requires ffmpeg)
            - "gif": Animated GIF file (requires pillow)
            - "matplotlib": Show in matplotlib GUI window (blocking)
        display
            If True and output_format is "html", returns an HTML object for notebooks.
            If output_format is "matplotlib", this is ignored (always shows window).
            If False, only saves to file (if folder_path is provided).

        Returns
        -------
        IPython.display.HTML or None
            Returns HTML animation if display=True and output_format="html", otherwise None.
        """
        try:
            import cv2
            from matplotlib.animation import FuncAnimation
            import matplotlib as mpl
            mpl.rcParams['animation.embed_limit'] = 100
        except ImportError as e:
            raise ImportError(
                f"Missing required dependency for animation: {e}. "
                "Please install cv2 (opencv-python)."
            )

        # Validate output_format
        valid_formats = ["html", "mp4", "gif", "matplotlib"]
        if output_format not in valid_formats:
            raise ValueError(f"output_format must be one of {valid_formats}, got '{output_format}'")

        # ---- Determine gaze columns ----
        if "X" in samples.columns and "Y" in samples.columns:
            x_col, y_col = "X", "Y"
        elif "LX" in samples.columns and "LY" in samples.columns:
            x_col, y_col = "LX", "LY"
        elif "RX" in samples.columns and "RY" in samples.columns:
            x_col, y_col = "RX", "RY"
        else:
            raise ValueError("Samples DataFrame must contain gaze columns (X, Y) or (LX, LY) or (RX, RY)")

        # ---- Time filter ----
        if tmin is not None and tmax is not None:
            samples = samples.filter(pl.col("tSample").is_between(tmin, tmax))

        if samples.is_empty():
            raise ValueError("No samples available after time filtering")

        # ---- Drop NaN gaze values ----
        samples = samples.filter(pl.col(x_col).is_not_null() & pl.col(y_col).is_not_null())

        # ---- Calculate scaled dimensions ----
        scaled_width = int(screen_width * scale_factor)
        scaled_height = int(screen_height * scale_factor)

        trial_idx = samples["trial_number"][0] if "trial_number" in samples.columns else 0

        # ================= WITH VIDEO =================
        if video_path is not None:
            video_path = Path(video_path)
            if not video_path.exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")

            cap = cv2.VideoCapture(str(video_path))
            video_fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if fps is None:
                fps = video_fps

            # Calculate time to frame mapping
            t_start = samples["tSample"].min()
            t_end = samples["tSample"].max()
            trial_duration = t_end - t_start

            # Create frame-to-time mapping
            frame_edges = np.linspace(t_start, t_end, total_frames + 1)
            frame_times = ((frame_edges[:-1] + frame_edges[1:]) / 2).astype(int)

            # Build a lookup: frame_index -> list of gaze points
            samples_np = samples.select([x_col, y_col, "tSample"]).to_numpy()
            gaze_by_frame = {i: [] for i in range(total_frames)}

            for x, y, t in samples_np:
                # Find the closest frame
                frame_idx = np.searchsorted(frame_times, t, side='right') - 1
                frame_idx = max(0, min(frame_idx, total_frames - 1))
                gaze_by_frame[frame_idx].append((x, y))

            # Limit frames if seconds_to_show is set
            frames_to_show = total_frames
            if seconds_to_show is not None:
                frames_to_show = min(int(fps * seconds_to_show), total_frames)

            # Reset video
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            # Create figure
            fig, ax = plt.subplots(figsize=(10 * scale_factor, 6 * scale_factor))
            ax.axis('off')

            # Initialize with first frame
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise RuntimeError("Could not read first frame from video")

            frame_resized = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            im = ax.imshow(frame_rgb)

            def update_frame_video(frame_idx):
                ret, frame = cap.read()
                if not ret:
                    return [im]

                frame_resized = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

                # Draw gaze points for this frame
                for gx, gy in gaze_by_frame.get(frame_idx, []):
                    scaled_x = int(gx * scale_factor)
                    scaled_y = int(gy * scale_factor)
                    if 0 <= scaled_x < scaled_width and 0 <= scaled_y < scaled_height:
                        radius = max(3, int(gaze_radius * scale_factor))
                        cv2.circle(frame_rgb, (scaled_x, scaled_y), radius=radius, color=gaze_color, thickness=-1)

                im.set_array(frame_rgb)
                return [im]

            anim = FuncAnimation(fig, update_frame_video, frames=frames_to_show,
                                 interval=1000/fps, blit=True, repeat=True)

        # ================= WITHOUT VIDEO =================
        else:
            if fps is None:
                fps = 60  # Default FPS for sample-based animation

            # Prepare background
            if background_image_path is not None:
                bg_path = Path(background_image_path)
                if not bg_path.exists():
                    raise FileNotFoundError(f"Background image not found: {bg_path}")
                bg_img = mpimg.imread(str(bg_path))
                if bg_img.dtype == np.float64:
                    bg_img = (bg_img * 255).clip(0, 255).astype(np.uint8)
                # Resize background to match screen dimensions then scale
                bg_img = cv2.resize(bg_img, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
            else:
                # Grey background
                bg_img = np.ones((scaled_height, scaled_width, 3), dtype=np.uint8) * 128

            # Get time range
            t_start = samples["tSample"].min()
            t_end = samples["tSample"].max()
            trial_duration = t_end - t_start

            # Limit duration if seconds_to_show is set
            if seconds_to_show is not None:
                t_end = min(t_end, t_start + int(seconds_to_show * 1000))
                samples = samples.filter(pl.col("tSample") <= t_end)
                trial_duration = t_end - t_start

            # Calculate total frames based on duration and fps
            total_frames = int((trial_duration / 1000) * fps)
            if total_frames < 1:
                total_frames = 1

            # Create time bins for each animation frame
            frame_times = np.linspace(t_start, t_end, total_frames + 1)

            # Build gaze lookup by frame
            samples_np = samples.select([x_col, y_col, "tSample"]).to_numpy()
            gaze_by_frame = {i: [] for i in range(total_frames)}

            for x, y, t in samples_np:
                frame_idx = np.searchsorted(frame_times, t, side='right') - 1
                frame_idx = max(0, min(frame_idx, total_frames - 1))
                gaze_by_frame[frame_idx].append((x, y))

            # Create figure
            fig, ax = plt.subplots(figsize=(10 * scale_factor, 6 * scale_factor))
            ax.axis('off')

            # Initialize with background
            im = ax.imshow(bg_img.copy())

            def update_frame_no_video(frame_idx):
                # Start with fresh background copy
                frame_rgb = bg_img.copy()

                # Draw gaze points for this frame
                for gx, gy in gaze_by_frame.get(frame_idx, []):
                    scaled_x = int(gx * scale_factor)
                    scaled_y = int(gy * scale_factor)
                    if 0 <= scaled_x < scaled_width and 0 <= scaled_y < scaled_height:
                        radius = max(3, int(gaze_radius * scale_factor))
                        cv2.circle(frame_rgb, (scaled_x, scaled_y), radius=radius, color=gaze_color, thickness=-1)

                im.set_array(frame_rgb)
                return [im]

            anim = FuncAnimation(fig, update_frame_no_video, frames=total_frames,
                                 interval=1000/fps, blit=True, repeat=True)

        # ================= SAVE / DISPLAY =================
        result = None
        trial_idx_val = trial_idx
        
        # Build output filename
        anim_name = f"animation_{trial_idx_val}"
        if tmin is not None and tmax is not None:
            anim_name += f"_{tmin}_{tmax}"

        # Handle different output formats
        if output_format == "matplotlib":
            # Show in matplotlib GUI window (blocking)
            plt.show()
            # Cleanup video capture if used
            if video_path is not None:
                cap.release()
            return None

        elif output_format == "mp4":
            if folder_path:
                folder_path = Path(folder_path)
                folder_path.mkdir(parents=True, exist_ok=True)
                out_path = folder_path / f"{anim_name}.mp4"
                try:
                    anim.save(str(out_path), writer='ffmpeg', fps=fps)
                    print(f"Animation saved to: {out_path}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to save MP4. Make sure ffmpeg is installed. Error: {e}"
                    )
            plt.close(fig)

        elif output_format == "gif":
            if folder_path:
                folder_path = Path(folder_path)
                folder_path.mkdir(parents=True, exist_ok=True)
                out_path = folder_path / f"{anim_name}.gif"
                try:
                    anim.save(str(out_path), writer='pillow', fps=fps)
                    print(f"Animation saved to: {out_path}")
                except Exception as e:
                    raise RuntimeError(
                        f"Failed to save GIF. Make sure pillow is installed. Error: {e}"
                    )
            plt.close(fig)

        else:  # html (default)
            if folder_path:
                folder_path = Path(folder_path)
                folder_path.mkdir(parents=True, exist_ok=True)
                out_path = folder_path / f"{anim_name}.html"
                with open(out_path, 'w') as f:
                    f.write(anim.to_jshtml())
                print(f"Animation saved to: {out_path}")

            if display:
                try:
                    from IPython.display import HTML
                    plt.close(fig)
                    result = HTML(anim.to_jshtml())
                except ImportError:
                    print("IPython not available. Use output_format='matplotlib' for GUI display.")
                    plt.close(fig)
            else:
                plt.close(fig)

        # Cleanup video capture if used
        if video_path is not None:
            cap.release()

        return result

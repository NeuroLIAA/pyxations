import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mplcolors
import numpy as np
import polars as pl
from concurrent.futures import ProcessPoolExecutor
from pyxations.bids_formatting import EYE_MOVEMENT_DETECTION_DICT
from pathlib import Path


class Visualization():
    def __init__(self, derivatives_folder_path,events_detection_algorithm):
        self.derivatives_folder_path = Path(derivatives_folder_path)
        if events_detection_algorithm not in EYE_MOVEMENT_DETECTION_DICT and events_detection_algorithm != 'eyelink':
            raise ValueError(f"Detection algorithm {events_detection_algorithm} not found.")
        self.events_detection_folder = Path(events_detection_algorithm+'_events')

    def scanpath(self,fixations:pl.DataFrame,screen_height:int, screen_width:int,folder_path:str=None,
                 tmin:int=None, tmax:int=None, saccades:pl.DataFrame=None,samples:pl.DataFrame=None, phase_data:dict=None, display:bool=True):
        """
        Plots the scanpath, including fixations, saccades, and optionally an image background and gaze samples.

        Parameters
        ----------
        fixations : pd.DataFrame
            DataFrame containing fixation data with the following columns:
            'tStart', 'tEnd', 'duration', 'xAvg', 'yAvg'.
        screen_width : int
            Horizontal resolution of the screen in pixels.
        screen_height : int
            Vertical resolution of the screen in pixels.
        folder_path : str
            Path to the folder where the plots will be saved.
        tmin : int, optional
            The minimum time for filtering the data.
        tmax : int, optional
            The maximum time for filtering the data.
        saccades : pd.DataFrame, optional
            DataFrame containing saccade data with the following columns:
            'tStart', 'tEnd', 'ampDeg', 'vPeak', 'xStart', 'xEnd', 'yStart', 'yEnd'.
        samples : pd.DataFrame, optional
            DataFrame containing gaze samples data with the following columns:
            'tSample', 'LX', 'LY', 'RX', 'RY'.
        phase_data: dict, optional
            This dictionary should have the phase as key and as value it should have a dictionary with the following:
                img_paths : list of strs, optional
                    Paths to image files to be plotted.
                img_plot_coords : list of tuples, optional
                    Tuples with the coordinates of the images to plot. The tuples should be in the format (x1, y1, x2, y2) where
                    (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
                bbox : tuple, optional
                    Tuple with the coordinates of the bounding box to plot. The tuple should be in the format (x1, y1, x2, y2) where
                    (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.
        display: bool
            Show plot


        Either trial_index or trial_label or tmix and tmax must be provided.

        """
        plot_saccades = not saccades is None
        plot_samples = not samples is None
        trial_index = fixations.select(pl.col('trial_number')).to_numpy()[0]
        
        if folder_path:
            scanpath_file_name = 'scanpath' + f'_{trial_index}'+ f'_{tmin}_{tmax}'*(tmin is not None and tmax is not None) 
            
       
        #----- Filter saccades, fixations and samples to defined time interval -----#
        if tmax is not None and tmin is not None:
            filtered_fixations = fixations.filter(pl.col('tStart').is_between(tmin, tmax))


            if plot_saccades:
                filtered_saccades = saccades.filter(pl.col('tStart').is_between(tmin, tmax))

            if plot_samples:
                filtered_samples = samples.filter(pl.col('tSample').is_between(tmin, tmax))
        else:
            filtered_fixations = fixations
            if plot_saccades:
                filtered_saccades = saccades
            if plot_samples:
                filtered_samples = samples
        
        # Filter the data where the "phase" is not empty
        filtered_fixations = filtered_fixations.filter(pl.col('phase') != '')
        if plot_saccades:
            filtered_saccades = filtered_saccades.filter(pl.col('phase') != '')
        if plot_samples:
            filtered_samples = filtered_samples.filter(pl.col('phase') != '')
        
        for phase in filtered_fixations.select(pl.col('phase')).unique().to_numpy():
            phase_fixations = filtered_fixations.filter(pl.col('phase') == phase)
            if plot_saccades:
                phase_saccades = filtered_saccades.filter(pl.col('phase') == phase)
            if plot_samples:
                phase_samples = filtered_samples.filter(pl.col('phase') == phase)

            #----- Define figure and axes -----#
            if plot_samples:
                fig, axs = plt.subplots(nrows=2, ncols=1, height_ratios=(4, 1),figsize=(10, 6))
                ax_main = axs[0]
                ax_gaze = axs[1]
            else:
                fig, ax_main = plt.subplots(figsize=(10, 6))

            ax_main.set_xlim(0, screen_width)
            ax_main.set_ylim(0, screen_height)


            #----- Plot fixations as dots if any in time interval -----#
            # Colormap: Get fixation durations for scatter circle size
            sizes = phase_fixations.select(pl.col('duration')).to_numpy()
            
            # Define rainwbow cmap for fixations
            cmap = plt.cm.rainbow
            
            # Define the bins and normalize
            fix_num = list(range(1,len(phase_fixations)+1))
            bounds = np.linspace(1, fix_num[-1] + 1, fix_num[-1] + 1)
            norm = mplcolors.BoundaryNorm(bounds, cmap.N)

            
            # Plot
            ax_main.scatter(phase_fixations.select(pl.col('xAvg')).to_numpy(), phase_fixations.select(pl.col('yAvg')).to_numpy(), c=fix_num, s=sizes, cmap=cmap, norm=norm, alpha=0.5, zorder=2)

            # Colorbar
            PCM = ax_main.get_children()[0]  # When the fixations dots for color mappable were ploted (first)
            cb = plt.colorbar(PCM, ax=ax_main, ticks=[fix_num[0], fix_num[int(len(fix_num)/2)], fix_num[-1] + 1], fraction=0.046, pad=0.04)
            cb.ax.set_yticklabels([fix_num[0], fix_num[int(len(fix_num)/2)], fix_num[-1]])
            cb.set_label('# of fixation')

            #----- Plot image if provided -----#
            if phase_data is not None:
                phase_data_current = phase_data.get(phase, {})
                img_paths = phase_data_current.get('img_paths',None)
                bbox = phase_data_current.get('bbox',None)
                if img_paths is not None:
                    for i, img_path in enumerate(img_paths):
                        # Load search image
                        img = mpimg.imread(img_path)
                        img_bbox = phase_data_current.get('img_plot_coords', None)[i]

                        # Define box in axes to plot image, because the image should be centered even if it doesn't have the same resolution as the screen
                        image_box_extent = [img_bbox[0], img_bbox[2], img_bbox[1], img_bbox[3]]
                        # Plot
                        ax_main.imshow(img, extent=image_box_extent, zorder=0)
                if bbox is not None:
                    # Define the bounding box in red
                    x1, y1, x2, y2 = bbox
                    # The bounding box was defined as if the top left corner was the origin, so we need to adjust it to the bottom left corner
                    y1 = screen_height - y1
                    y2 = screen_height - y2

                    ax_main.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color='red', linewidth=1.5, zorder=3)

            #----- Plot scanpath and gaze if samples provided -----#
            if plot_samples:
                starting_time = phase_samples.select(pl.col('tSample')).to_numpy()[0]
                tSamples_from_start = (phase_samples.select(pl.col('tSample')).to_numpy() - starting_time) / 1000
                # Left eye
                try:
                    phase_samples_lx = phase_samples.select(pl.col('LX')).to_numpy()
                    phase_samples_ly = phase_samples.select(pl.col('LY')).to_numpy()
                    ax_main.plot(phase_samples_lx, phase_samples_ly, '--', color='C0', zorder=1)
                    ax_gaze.plot(tSamples_from_start, phase_samples_lx, label='Left X')
                    ax_gaze.plot(tSamples_from_start, phase_samples_ly, label='Left Y')
                except:
                    pass
                # Right eye
                try:
                    phase_samples_rx = phase_samples.select(pl.col('RX')).to_numpy()
                    phase_samples_ry = phase_samples.select(pl.col('RY')).to_numpy()
                    ax_main.plot(phase_samples_rx, phase_samples_ry, '--', color='black', zorder=1)
                    ax_gaze.plot(tSamples_from_start, phase_samples_rx, label='Right X')
                    ax_gaze.plot(tSamples_from_start, phase_samples_ry, label='Right Y')
                except:
                    pass
                try:
                    phase_samples_x = phase_samples.select(pl.col('X')).to_numpy()
                    phase_samples_y = phase_samples.select(pl.col('Y')).to_numpy()
                    ax_main.plot(phase_samples_x, phase_samples_y, '--', color='black', zorder=1)
                    ax_gaze.plot(tSamples_from_start, phase_samples_x, label='X')
                    ax_gaze.plot(tSamples_from_start, phase_samples_y, label='Y')
                except:
                    pass
                plot_min, plot_max = ax_gaze.get_ylim()
                # Plot fixations as color span in gaze axes
                for fix_idx,(_, fixation) in enumerate(phase_fixations.iter_rows(named=True)):
                    color = cmap(norm(fix_idx + 1))
                    
                    ax_gaze.axvspan(ymin=0, ymax=1, xmin=(fixation['tStart'] - starting_time), xmax=(fixation['tStart'] - starting_time + fixation['duration']), color=color, alpha=0.4, label='fix')
                
                # Plor saccades as vlines in gaze axes
                if plot_saccades:
                    for _, saccade in phase_saccades.iter_rows(named=True):
                        ax_gaze.vlines(x=(saccade['tStart']- starting_time), ymin=plot_min, ymax=plot_max, colors='red', linestyles='--', label='sac', linewidth=0.8)

                # Legend
                handles, labels = plt.gca().get_legend_handles_labels()
                by_label = dict(zip(labels, handles))
                plt.legend(by_label.values(), by_label.keys(),  loc='center left', bbox_to_anchor=(1, 0.5))
                ax_gaze.set_ylabel('Gaze')
                ax_gaze.set_xlabel('Time [ms]')
            plt.tight_layout()  
            if folder_path:
                file_path = folder_path / (scanpath_file_name + f'_{phase}.png')
                fig.savefig(file_path)
            if display:
                plt.show()
            plt.close()


    def fix_duration(self,fixations:pl.DataFrame,axs=None):
        
        ax = axs
        if ax is None:
            fig, ax = plt.subplots()

        ax.hist(fixations.select(pl.col('duration')).to_numpy(), bins=100, edgecolor='black', linewidth=1.2, density=True)
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
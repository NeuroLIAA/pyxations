import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mplcolors
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from .bids_formatting import EYE_MOVEMENT_DETECTION_DICT
from os import path, makedirs,listdir


class Visualization():
    def __init__(self, derivatives_folder_path,events_detection_algorithm):
        self.derivatives_folder_path = derivatives_folder_path
        if events_detection_algorithm not in EYE_MOVEMENT_DETECTION_DICT and events_detection_algorithm != 'eyelink':
            raise ValueError(f"Detection algorithm {events_detection_algorithm} not found.")
        self.events_detection_folder = events_detection_algorithm+'_events'

    def get_data_and_plot_scanpaths(self,session_folder_path:str):
        header = pd.read_hdf(path.join(session_folder_path,'header.hdf5'))
        screen_size = header['line'].iloc[-1].split()
        screen_height = int(screen_size[-1])
        screen_width = int(screen_size[-2])
        samples = pd.read_hdf(path.join(session_folder_path,'samples.hdf5'))
        fixations = pd.read_hdf(path.join(session_folder_path,self.events_detection_folder,'fix.hdf5'))
        saccades = pd.read_hdf(path.join(session_folder_path,self.events_detection_folder,'sacc.hdf5'))
        unique_trials = fixations['trial_number'].unique()
        unique_trials = unique_trials[unique_trials != -1]
        folder_path = path.join(session_folder_path,self.events_detection_folder,'plots')
        makedirs(folder_path, exist_ok=True)
        for trial in unique_trials:
            self.scanpath(fixations,screen_height,screen_width,folder_path,trial_index=trial,saccades=saccades,samples=samples)
        return fixations[['duration']],saccades[['ampDeg','vPeak','deg','dir']]
    
    def process_session(self, session_info):
        subject, session = session_info
        return self.get_data_and_plot_scanpaths(
            path.join(self.derivatives_folder_path, subject, session)
        )

    def global_plots(self,max_workers:int=8):
        bids_folders = [folder for folder in listdir(self.derivatives_folder_path) if folder.startswith("sub-")]
        fixations = []
        saccades = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = executor.map(
                self.process_session,
                [
                    (subject, session)
                    for subject in bids_folders
                    for session in listdir(path.join(self.derivatives_folder_path, subject))
                    if session.startswith("ses-")
                ]
            )

            for result in results:
                fixations.append(result[0])
                saccades.append(result[1])
        self.plot_multipanel(path.join(self.derivatives_folder_path,self.events_detection_folder,"plots"),pd.concat(fixations),pd.concat(saccades))


    def scanpath(self,fixations:pd.DataFrame,screen_height:int, screen_width:int,folder_path:str,trial_index:int=None,trial_label:str=None,
                 tmin:int=None, tmax:int=None, img_path:str=None,saccades:pd.DataFrame=None,samples:pd.DataFrame=None):
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
        trial_index : int, optional
            Index of the trial.
        trial_label : str, optional
            Label of the trial.
        tmin : int, optional
            The minimum time for filtering the data.
        tmax : int, optional
            The maximum time for filtering the data.
        img_path : str, optional
            Path to an image file to be used as a background.
        saccades : pd.DataFrame, optional
            DataFrame containing saccade data with the following columns:
            'tStart', 'tEnd', 'ampDeg', 'vPeak', 'xStart', 'xEnd', 'yStart', 'yEnd'.
        samples : pd.DataFrame, optional
            DataFrame containing gaze samples data with the following columns:
            'tSample', 'LX', 'LY', 'RX', 'RY'.


        Either trial_index or trial_label or tmix and tmax must be provided.

        """
        plot_saccades = not saccades is None
        plot_samples = not samples is None

        scanpath_file_name = 'scanpath' + f'_{trial_index}'*(trial_index is not None) + f'_{trial_label}'*(trial_label is not None) + f'_{tmin}_{tmax}'*(tmin is not None and tmax is not None) + '.png'
        file_path = path.join(folder_path, scanpath_file_name)

        #----- Filter saccades, fixations and samples to defined time interval -----#
        if tmax is not None and tmin is not None:
            filtered_fixations = fixations[(fixations['tStart'] >= tmin) & (fixations['tStart'] <= tmax)]
            if plot_saccades:
                filtered_saccades = saccades[(saccades['tStart'] >= tmin) & (saccades['tStart'] <= tmax)]
            if plot_samples:
                filtered_samples = samples[(samples['tSample'] >= tmin) & (samples['tSample'] <= tmax)]
        
        #----- Filter saccades, fixations and samples to defined trial -----#
        if trial_index is not None:
            filtered_fixations = fixations[fixations['trial_number'] == trial_index]
            if plot_saccades:
                filtered_saccades = saccades[saccades['trial_number'] == trial_index]
            if plot_samples:
                filtered_samples = samples[samples['trial_number'] == trial_index]

        if trial_label is not None:
            filtered_fixations = fixations[fixations['trial_label'] == trial_label]
            if plot_saccades:
                filtered_saccades = saccades[saccades['trial_label'] == trial_label]
            if plot_samples:
                filtered_samples = samples[samples['trial_label'] == trial_label]

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
        sizes = filtered_fixations['duration']
        
        # Define rainwbow cmap for fixations
        cmap = plt.cm.rainbow
        
        # Define the bins and normalize
        fix_num = list(range(1,len(filtered_fixations)+1))
        bounds = np.linspace(1, fix_num[-1] + 1, fix_num[-1] + 1)
        norm = mplcolors.BoundaryNorm(bounds, cmap.N)

        
        # Plot
        ax_main.scatter(filtered_fixations['xAvg'], filtered_fixations['yAvg'], c=fix_num, s=sizes, cmap=cmap, norm=norm, alpha=0.5, zorder=2)

        # Colorbar
        PCM = ax_main.get_children()[0]  # When the fixations dots for color mappable were ploted (first)
        cb = plt.colorbar(PCM, ax=ax_main, ticks=[fix_num[0] + 1/2, fix_num[int(len(fix_num)/2)]+1/2, fix_num[-1]+1/2], fraction=0.046, pad=0.04)
        cb.ax.set_yticklabels([fix_num[0], fix_num[int(len(fix_num)/2)], fix_num[-1]])
        cb.set_label('# of fixation')


        #----- Plot image if provided -----#
        if img_path is not None:
            # Load search image
            img = mpimg.imread(img_path)
            
            # Get image size
            img_res_x = img.shape[1]
            img_res_y = img.shape[0]
            
            # Define box in axes to plot image
            image_box_extent = ((screen_width - img_res_x) / 2, img_res_x + (screen_width - img_res_x) / 2, (screen_height - img_res_y) / 2, img_res_y + (screen_height - img_res_y) / 2) 
            
            # Plot
            ax_main.imshow(img, extent=image_box_extent, zorder=0)

        #----- Plot scanpath and gaze if samples provided -----#
        if plot_samples:
            starting_time = filtered_samples['tSample'].iloc[0]
            tSamples_from_start = (filtered_samples['tSample'] - starting_time)/filtered_samples['Rate_recorded']
            # Left eye
            try:
                ax_main.plot(filtered_samples['LX'], filtered_samples['LY'], '--', color='C0', zorder=1)
                ax_gaze.plot(tSamples_from_start, filtered_samples['LX'], label='Left X')
                ax_gaze.plot(tSamples_from_start, filtered_samples['LY'], label='Left Y')
            except:
                pass
            # Right eye
            try:
                ax_main.plot(filtered_samples['X'], filtered_samples['Y'], '--', color='black', zorder=1)
                ax_gaze.plot(tSamples_from_start, filtered_samples['X'], label='Right X')
                ax_gaze.plot(tSamples_from_start, filtered_samples['RY'], label='Right Y')
            except:
                pass
            try:
                ax_main.plot(filtered_samples['X'], filtered_samples['Y'], '--', color='black', zorder=1)
                ax_gaze.plot(tSamples_from_start, filtered_samples['X'], label='X')
                ax_gaze.plot(tSamples_from_start, filtered_samples['Y'], label='Y')
            except:
                pass
            plot_min, plot_max = ax_gaze.get_ylim()
            # Plot fixations as color span in gaze axes
            for fix_idx, fixation in filtered_fixations.iterrows():
                color = cmap(norm(fix_idx + 1))
                ax_gaze.axvspan(ymin=0, ymax=1, xmin=(fixation['tStart'] - starting_time)/fixation['Rate_recorded'], xmax=(fixation['tStart'] - starting_time + fixation['duration'])/fixation['Rate_recorded'], color=color, alpha=0.4, label='fix')
            
            # Plor saccades as vlines in gaze axes
            if plot_saccades:
                for _, saccade in filtered_saccades.iterrows():
                    ax_gaze.vlines(x=(saccade['tStart']- starting_time)/saccade['Rate_recorded'], ymin=plot_min, ymax=plot_max, colors='red', linestyles='--', label='sac')

            # Legend
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(),  loc='center left', bbox_to_anchor=(1, 0.5))
            ax_gaze.set_ylabel('Gaze')
            ax_gaze.set_xlabel('Time [s]')
        plt.tight_layout()  
        fig.savefig(file_path)
        plt.close()


    def fix_duration(self,fixations:pd.DataFrame,axs=None):
        
        ax = axs
        if ax is None:
            fig, ax = plt.subplots()

        ax.hist(fixations['duration'], bins=100, edgecolor='black', linewidth=1.2, density=True)
        ax.set_title('Fixation duration')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Density')


    def sacc_amplitude(self,saccades:pd.DataFrame,axs=None):

        ax = axs
        if ax is None:
            fig, ax = plt.subplots()

        saccades_amp = saccades['ampDeg']
        ax.hist(saccades_amp, bins=100, range=(0, 20), edgecolor='black', linewidth=1.2, density=True)
        ax.set_title('Saccades amplitude')
        ax.set_xlabel('Amplitude (deg)')
        ax.set_ylabel('Density')


    def sacc_direction(self,saccades:pd.DataFrame,axs=None,figs=None):

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
        saccades_rad = saccades['deg'] * np.pi / 180 

        n_bins = 24
        ang_hist, bin_edges = np.histogram(saccades_rad, bins=24, density=True)
        bin_centers = [np.mean((bin_edges[i], bin_edges[i+1])) for i in range(len(bin_edges) - 1)]

        bars = ax.bar(bin_centers, ang_hist, width=2*np.pi/n_bins, bottom=0.0, alpha=0.4, edgecolor='black')
        ax.set_title('Saccades direction')
        ax.set_yticklabels([])

        for r, bar in zip(ang_hist, bars):
            bar.set_facecolor(plt.cm.Blues(r / np.max(ang_hist)))


    def sacc_main_sequence(self,saccades:pd.DataFrame,axs=None, hline=None):

        ax = axs
        if ax is None:
            fig, ax = plt.subplots()

        saccades_peack_vel = saccades['vPeak']
        saccades_amp = saccades['ampDeg']

        ax.plot(saccades_amp, saccades_peack_vel, '.', alpha=0.1, markersize=2)
        ax.set_xlim(0.01)
        if hline:
            ax.hlines(y=hline, xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], colors='grey', linestyles='--', label=hline)
            ax.legend()
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title('Main sequence')
        ax.set_xlabel('Amplitude (deg)')
        ax.set_ylabel('Peak velocity (deg)')
        ax.grid()


    def plot_multipanel(self,folder_path:str,fixations:pd.DataFrame,saccades:pd.DataFrame):
        plt.rcParams.update({'font.size': 12})
        fig, axs = plt.subplots(2, 2, figsize=(12, 7))
        
        self.fix_duration(fixations,axs=axs[0, 0])
        self.sacc_main_sequence(saccades,axs=axs[1, 1])
        self.sacc_direction(saccades,axs=axs[1, 0],figs=fig)
        self.sacc_amplitude(saccades,axs=axs[0, 1])

        fig.tight_layout()
        makedirs(folder_path, exist_ok=True)
        plt.savefig(path.join(folder_path,'multipanel.png'))
        plt.close()
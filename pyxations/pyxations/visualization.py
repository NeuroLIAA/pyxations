import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mplcolors
import numpy as np
import pandas as pd
from os import path

class Visualization():
    def __init__(self, session_folder_path):
        self.session_folder_path = session_folder_path

    def scanpath(self,fixations:pd.DataFrame,tmin:float, tmax:float, img_path:str=None, saccades:pd.DataFrame=None, samples:pd.DataFrame=None,
             screen_res_x:int=1920, screen_res_y:int=1080):
        """
        Plots the scanpath, including fixations, saccades, and optionally an image background and gaze samples.

        Parameters
        ----------
        fixations : pd.DataFrame
            DataFrame containing fixation data with the following columns:
        tmin : float
            The minimum time for filtering the data.
        tmax : float
            The maximum time for filtering the data.
        img_path : str, optional
            Path to an image file to be used as a background (default is None).
        saccades : pd.DataFrame, optional
            DataFrame containing saccade data with the following columns:
            'tStart', 'tEnd', 'ampDeg', 'vPeak', 'xStart', 'xEnd', 'yStart', 'yEnd'.
        samples : pd.DataFrame, optional
            DataFrame containing gaze samples data with the following columns:
            'tSample', 'LX', 'LY', 'RX', 'RY'.
        screen_res_x : int, optional
            Horizontal resolution of the screen in pixels (default is 1920).
        screen_res_y : int, optional
            Vertical resolution of the screen in pixels (default is 1080).

        Returns
        -------
        fig : matplotlib.figure.Figure
            The figure object containing the plot.
        fixations : pd.DataFrame
            The filtered fixations DataFrame.
        saccades : pd.DataFrame
            The filtered saccades DataFrame.
        samples : pd.DataFrame
            The filtered samples DataFrame.
        """



        #----- Filter saccades, fixations and samples to defined time interval -----#
        fixations = fixations.loc[(fixations['tStart'] >= tmin) & (fixations['tStart'] <= tmax)].reset_index(drop=True)
        if type(saccades) == pd.DataFrame:
            saccades = saccades.loc[(saccades['tStart'] >= tmin) & (saccades['tStart'] <= tmax)]
        if type(samples) == pd.DataFrame:
            samples = samples.loc[(samples['tSample'] >= tmin) & (samples['tSample'] <= tmax)]


        #----- Define figure and axes -----#
        if type(samples) == pd.DataFrame:
            fig, axs = plt.subplots(nrows=2, ncols=1, height_ratios=(4, 1),figsize=(10, 6))
            ax_main = axs[0]
            ax_gaze = axs[1]
        else:
            fig, ax_main = plt.subplots(figsize=(10, 6))
        
        fig.suptitle(f'Scanpath tmin:{tmin} - tmax:{tmax}')
        ax_main.set_xlim(0, screen_res_x)
        ax_main.set_ylim(0, screen_res_y)


        #----- Plot fixations as dots if any in time interval -----#
        # Colormap: Get fixation durations for scatter circle size
        sizes = fixations['duration']
        
        # Define rainwbow cmap for fixations
        cmap = plt.cm.rainbow
        
        # Define the bins and normalize
        fix_num = fixations.index + 1
        bounds = np.linspace(1, fix_num[-1] + 1, fix_num[-1] + 1)
        norm = mplcolors.BoundaryNorm(bounds, cmap.N)
        
        # Plot
        ax_main.scatter(fixations['xAvg'], fixations['yAvg'], c=fix_num, s=sizes, cmap=cmap, norm=norm, alpha=0.5, zorder=2)

        # Colorbar
        PCM = ax_main.get_children()[0]  # When the fixations dots for color mappable were ploted (first)
        cb = plt.colorbar(PCM, ax=ax_main, ticks=[fix_num[0] + 1/2, fix_num[int(len(fix_num)/2)]+1/2, fix_num[-1]+1/2], fraction=0.046, pad=0.04)
        cb.ax.set_yticklabels([fix_num[0], fix_num[int(len(fix_num)/2)], fix_num[-1]])
        cb.set_label('# of fixation')


        #----- Plot image if provided -----#
        if img_path != None:
            # Load search image
            img = mpimg.imread(img_path)
            
            # Get image size
            img_res_x = img.shape[1]
            img_res_y = img.shape[0]
            
            # Define box in axes to plot image
            image_box_extent = ((screen_res_x - img_res_x) / 2, img_res_x + (screen_res_x - img_res_x) / 2, (screen_res_y - img_res_y) / 2, img_res_y + (screen_res_y - img_res_y) / 2) 
            
            # Plot
            ax_main.imshow(img, extent=image_box_extent, zorder=0)

        #----- Plot scanpath and gaze if samples provided -----#
        if type(samples) == pd.DataFrame:

            # Scanpath
            ax_main.plot(samples['LX'], samples['LY'], '--', color='C0', zorder=1)
            ax_main.plot(samples['RX'], samples['RY'], '--', color='black', zorder=1)
            
            # Gaze
            ax_gaze.plot(samples['tSample'], samples['LX'], label='Left X')
            ax_gaze.plot(samples['tSample'], samples['LY'], label='Left Y')
            ax_gaze.plot(samples['tSample'], samples['RX'], label='Right X')
            ax_gaze.plot(samples['tSample'], samples['RY'], label='Right Y')

            plot_min, plot_max = ax_gaze.get_ylim()

            # Plot fixations as color span in gaze axes
            for fix_idx, fixation in fixations.iterrows():
                color = cmap(norm(fix_idx + 1))
                ax_gaze.axvspan(ymin=0, ymax=1, xmin=fixation['tStart'], xmax=fixation['tStart'] + fixation['duration'], color=color, alpha=0.4, label='fix')
            
            # Plor saccades as vlines in gaze axes
            if type(saccades) == pd.DataFrame:
                for _, saccade in saccades.iterrows():
                    ax_gaze.vlines(x=saccade['tStart'], ymin=plot_min, ymax=plot_max, colors='red', linestyles='--', label='sac')

            # Legend
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys(),  loc='center left', bbox_to_anchor=(1, 0.5))
            ax_gaze.set_ylabel('Gaze')
            ax_gaze.set_xlabel('Time [s]')
        # Save figure as "scanpath.png"
        plt.tight_layout()
        fig.savefig(path.join(self.session_folder_path, 'scanpath.png'))

        


    def duration(self,fixations:pd.DataFrame,axs=None):
        ax = axs
        print('Plotting fixation duration histogram')

        if ax is None:
            fig, ax = plt.subplots()

        ax.hist(fixations['duration'], bins=100, edgecolor='black', linewidth=1.2, density=True)
        ax.set_title('Fixation duration')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Density')


    def amplitude(self,saccades:pd.DataFrame,axs=None):
        print('Plotting saccades amplitude histogram')
        ax = axs
        if ax is None:
            fig, ax = plt.subplots()

        saccades_amp = saccades['ampDeg']
        ax.hist(saccades_amp, bins=100, range=(0, 20), edgecolor='black', linewidth=1.2, density=True)
        ax.set_title('Saccades amplitude')
        ax.set_xlabel('Amplitude (deg)')
        ax.set_ylabel('Density')


    def direction(self,saccades:pd.DataFrame,axs=None,figs=None):
        print('Plotting saccades direction histogram')
        ax = axs
        if ax is None:
            fig = plt.figure()
            ax = plt.subplot(polar=True)
        else:
            ax.set_axis_off()
            ax = figs.add_subplot(2, 2, 3, projection='polar')
        if 'deg' not in saccades.columns or 'dir' not in saccades.columns:
            raise ValueError('Compute saccades direction first by using saccades_direction function from the PostProcessing module.')
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


    def main_sequence(self,saccades:pd.DataFrame,axs=None, hline=None):
        print('Plotting main sequence')
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


    def plot_multipanel(self,fixations:pd.DataFrame,saccades:pd.DataFrame):
        plt.rcParams.update({'font.size': 12})
        fig, axs = plt.subplots(2, 2, figsize=(12, 7))
        

        self.duration(fixations,axs=axs[0, 0])
        self.main_sequence(saccades,axs=axs[1, 1])
        self.direction(saccades,axs=axs[1, 0],figs=fig)
        self.amplitude(saccades,axs=axs[0, 1])

        fig.tight_layout()
        plt.savefig(path.join(self.session_folder_path, 'multipanel.png'))


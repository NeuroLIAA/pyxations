import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from . import saccades_direction
from os import path

class Visualization():
    def __init__(self, session_folder_path):
        self.session_folder_path = session_folder_path

    def load_fixations(self):
        path_to_fixations = path.join(self.session_folder_path, 'fix.hdf5')
        fixations = pd.read_hdf(path_or_buf=path_to_fixations)
        return fixations

    def load_saccades(self):
        path_to_saccades = path.join(self.session_folder_path, 'sacc.hdf5')
        saccades = pd.read_hdf(path_or_buf=path_to_saccades)
        return saccades

    @staticmethod
    def duration(axs=None):
        def decorator(func):
            def wrapper():
                fixations = func()
                ax = axs
                print('Plotting fixation duration histogram')

                if ax is None:
                    fig, ax = plt.subplots()

                ax.hist(fixations['duration'], bins=100, edgecolor='black', linewidth=1.2, density=True)
                ax.set_title('Fixation duration')
                ax.set_xlabel('Time (ms)')
                ax.set_ylabel('Density')

                return fixations
            return wrapper
        return decorator

    @staticmethod
    def amplitude(axs=None):
        def decorator(func):
            def wrapper():
                saccades = func()

                print('Plotting saccades amplitude histogram')
                ax = axs
                if ax is None:
                    fig, ax = plt.subplots()

                saccades_amp = saccades['ampDeg']
                ax.hist(saccades_amp, bins=100, range=(0, 20), edgecolor='black', linewidth=1.2, density=True)
                ax.set_title('Saccades amplitude')
                ax.set_xlabel('Amplitude (deg)')
                ax.set_ylabel('Density')

                return saccades
            return wrapper
        return decorator

    @staticmethod
    def direction(axs=None,figs=None):
        def decorator(func):
            def wrapper():
                saccades = func()

                print('Plotting saccades direction histogram')
                ax = axs
                if ax is None:
                    fig = plt.figure()
                    ax = plt.subplot(polar=True)
                else:
                    ax.set_axis_off()
                    ax = figs.add_subplot(2, 2, 3, projection='polar')


                # Add degrees and direction columns to saccades df
                saccades = saccades_direction(saccades=saccades)

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

                return saccades
            return wrapper
        return decorator

    @staticmethod
    def main_sequence(axs=None, hline=None):
        def decorator(func):
            def wrapper():
                saccades = func()

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

                return saccades
            return wrapper
        return decorator

    def plot_multipanel(self):
        plt.rcParams.update({'font.size': 12})
        fig, axs = plt.subplots(2, 2, figsize=(12, 7))
        
        # Load fixations and apply the duration decorator
        decorated_load_fixations = self.duration(axs=axs[0, 0])(self.load_fixations)
        decorated_load_fixations()

        # Apply main sequence decorator
        main_sequence = self.main_sequence(axs=axs[1, 1])(self.load_saccades)
        print(self.load_saccades())

        # Apply direction decorator
        direction = self.direction(axs=axs[1, 0],figs=fig)(main_sequence)

        # Apply amplitude decorator
        amplitude = self.amplitude(axs=axs[0, 1])(direction)
        amplitude()

        fig.tight_layout()
        plt.savefig(path.join(self.session_folder_path, 'multipanel.png'))


'''
Created on Nov 22, 2024

@author: placiana
'''
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.colors as mplcolors
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.animation as animation


class SampleVisualization():
    
    def __init__(self, samples_df,screen_width=1366, screen_height=768):
        self.samples = samples_df
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        
    def plot(self, display=True, scanpath_file_name='scanpath', in_percent=False):
        df = self.samples
        folder_path = Path('.')

        
        fig, axs = plt.subplots(nrows=2, ncols=1, height_ratios=(4, 1),figsize=(10, 6))
        ax_main = axs[0]
        ax_gaze = axs[1]

        ax_main.set_xlim(0, self.screen_width)
        ax_main.set_ylim(0, self.screen_height)
        
        if in_percent:
            x = df['X'] * self.screen_width
            y = df['Y'] * self.screen_height
        else:
            x = df['X']
            y = df['Y']

        ax_main.plot(x, y, '--', color='C0', zorder=1)
        ax_gaze.plot(df['tSample'], x, label='X')
        ax_gaze.plot(df['tSample'], y, label='Y')        
        
        # Legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(),  loc='center left', bbox_to_anchor=(1, 0.5))
        ax_gaze.set_ylabel('Gaze')
        ax_gaze.set_xlabel('Time [ms]')
        plt.tight_layout()  
        if folder_path:
            file_path = folder_path / f'{scanpath_file_name}.png'
            fig.savefig(file_path)
        if display:
            plt.show()
        plt.close()
        
    def animate(self, display=True, in_percent=False, out_file='output.gif'):
        df = self.samples
        
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.screen_width)
        ax.set_ylim(0, self.screen_height)
        
        x = df['X']
        y = df['Y']
        if in_percent:
            x = df['X'] * self.screen_width
            y = df['Y'] * self.screen_height
        
        scat = ax.scatter(x[0], y[0], c="b", s=5, label='a')
        line2 = ax.plot(x[0], y[0], label=f'b')[0]
        #ax.set(xlim=[0, 3], ylim=[-4, 10], xlabel='Time [s]', ylabel='Z [m]')
        ax.legend()
        
        
        def update(frame):
            # for each frame, update the data stored on each artist.
            xa = x[:frame]
            ya = y[:frame]
            # update the scatter plot:
            data = np.stack([xa, ya]).T
            scat.set_offsets(data)
            # update the line plot:
            line2.set_xdata(x[:frame])
            line2.set_ydata(y[:frame])
            return (scat, line2)
        
        
        ani = animation.FuncAnimation(fig=fig, func=update, frames=400, interval=1)
        if display:
            plt.show()
        
        ani.save(filename=out_file, writer="pillow")
        
                
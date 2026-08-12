# Visualization

Plotting utilities for detected eye movements and for raw gaze samples.

Figures are written under the derivative dataset's `figures/` directory, in a
subdirectory named after the detection algorithm, so results from different
detectors do not overwrite each other. That directory is listed in the
dataset's `.bidsignore`, so plotting never invalidates the dataset.

## visualization

Scanpaths, fixation-duration and saccade-amplitude distributions, saccade
direction, the main sequence, the multipanel summary, and animated gaze.
Animations require the optional `video` extra, installed with
`pip install 'pyxations[video]'`.

::: pyxations.visualization.visualization

## samples

Plotting and animating sample-level gaze directly, without requiring detected
events.

::: pyxations.visualization.samples

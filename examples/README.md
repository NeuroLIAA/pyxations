# Source examples

This directory contains source recordings used by the documentation and
integration tests. Most are small, de-identified fixtures. Full original
recordings, such as `eyelink_target_absent`, should be reviewed for
de-identification and redistribution before release. Each data-bearing child
directory contains only the vendor recording and behavioral files supplied to
`dataset_to_bids`.

At conversion time this exact directory tree is copied to the dataset's
archival `sourcedata/`. Its recording, tracker-event, and behavioral content is
also normalized into the raw BIDS `beh/` files used by all later processing.

The raw BIDS datasets, BIDS Derivatives datasets, converted intermediary
files, and figures are generated in temporary or user-selected output
directories. They are intentionally not committed.

- `eyelink_visual_search`: reduced EyeLink ASC recording and two behavioral
  trials.
- `eyelink_target_absent`: full EyeLink EDF recording from a target-absent
  visual-search experiment. It requires SR Research's `edf2asc` utility and is
  kept separate from the reduced ASC fixture because they are different
  recordings.
- `webgazer_antisaccade`: jsPsych/WebGazer recording with embedded behavior.
- `tobii_sceneviewing`: reduced Tobii text export.
- `gazepoint_sart`: reduced GazePoint recording and behavioral events.
- `driving_animation`: background video used by the continuous-task plotting
  notebook; its BIDS dataset and derivatives must be generated separately.

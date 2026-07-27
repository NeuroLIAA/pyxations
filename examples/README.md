# Source examples

This directory contains small, de-identified source recordings used by both
the documentation and the integration tests. Each child directory contains
only the vendor recording and behavioral files supplied to
`dataset_to_bids`.

At conversion time this exact directory tree is copied to the dataset's
archival `sourcedata/`. Its recording, tracker-event, and behavioral content is
also normalized into the raw BIDS `beh/` files used by all later processing.

The raw BIDS datasets, BIDS Derivatives datasets, converted intermediary
files, and figures are generated in temporary or user-selected output
directories. They are intentionally not committed.

- `eyelink_visual_search`: reduced EyeLink ASC recording and two behavioral
  trials.
- `webgazer_antisaccade`: jsPsych/WebGazer recording with embedded behavior.
- `tobii_sceneviewing`: reduced Tobii text export.
- `gazepoint_sart`: reduced GazePoint recording and behavioral events.
- `driving_animation`: background video used by the continuous-task plotting
  notebook; its BIDS dataset and derivatives must be generated separately.

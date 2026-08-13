# Conversion to BIDS

Reading EyeLink, Tobii, GazePoint and webcam recordings, normalizing their
behavioral tables, and writing the raw BIDS dataset. The original vendor files
are preserved verbatim under the dataset's archival `sourcedata/`.

## bids_formatting

The dataset-level entry points: `dataset_to_bids` converts a folder of
recordings, and `compute_derivatives_for_dataset` computes derivatives from the
resulting raw dataset.

::: pyxations.bids_formatting

## bids

The raw BIDS reader and writer, plus the wrapper around the official BIDS
Validator used to check that generated datasets are valid.

::: pyxations.bids

## behavior

Normalizing behavioral CSV or TSV tables into BIDS `events.tsv`, with
source-independent column mapping.

::: pyxations.behavior

## psychopy

Parsing standard PsychoPy `New trial` logs. This does not require PsychoPy to
be installed, and PsychoPy-local timestamps are retained without assuming they
are synchronized to the eye-tracker clock.

::: pyxations.psychopy

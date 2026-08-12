# Preprocessing

Per-recording parsing and trial segmentation, applied to the normalized raw
BIDS dataset. The archived source files are not required at this point.

Segmentation supports explicit start and end timestamps, event-based message
markers, or fixed-duration trials, all with overlap controls and
regular-expression message matching. The operations you configure and their
parameters are logged to machine-readable JSON recipes and provenance sidecars,
so the transformations behind each derivative dataset stay explicit and
repeatable.

## pre_processing

::: pyxations.pre_processing

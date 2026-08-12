# Event detection

Turning processed gaze samples into fixations, saccades and blinks.

Pyxations ships the Engbert–Kliegl implementation and a REMoDNaV adapter, and
can also reuse the events reported by EyeLink's own parser. Selecting a
detector does not change the canonical BIDS storage layer, so results from
different algorithms are stored side by side and can be compared directly.

## eye_movement_detection

The abstract base class. Implement it to add support for another algorithm.

::: pyxations.methods.eyemovement.eye_movement_detection

## Engbert–Kliegl

Velocity-threshold detection following Engbert and Mergenthaler.

::: pyxations.methods.eyemovement.engbert

## REMoDNaV

Adapter for REMoDNaV. Requires the optional `remodnav` extra, installed with
`pip install 'pyxations[remodnav]'`.

::: pyxations.methods.eyemovement.remodnav_detector

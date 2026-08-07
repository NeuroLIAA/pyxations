# Analysis

The experiment hierarchy used to load derivatives and work with them.

The four levels are `Experiment -> Subject -> Session -> Trial`. Table
accessors such as `fixations()`, `saccades()`, `blinks()`, `pupil_samples()`
and `samples()` exist at every level and return Polars DataFrames with the
identifiers of that level attached, so the same call works whether you are
looking at one trial or at the whole dataset.

Quality filters are also available at every level: dropping short fixations,
merging nearby ones, removing trials with too many invalid samples, and
excluding uncalibrated or poorly calibrated trials.

For the search-task specialization of this hierarchy, see
[Visual search](visual_search.md).

## generic

::: pyxations.analysis.generic

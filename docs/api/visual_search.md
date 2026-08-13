# Visual search

A specialization of the [analysis hierarchy](analysis.md) for visual and hybrid
search paradigms.

Each level has a search-aware counterpart: `VisualSearchExperiment`,
`VisualSearchSubject`, `VisualSearchSession` and `VisualSearchTrial`. They add
the notions a search task needs, namely a memorization phase followed by a
search phase, a target that may be present or absent, a memory set whose size
varies, and per-stimulus grouping. Behavioral columns are read from the BIDS
`events.tsv` written during conversion.

This is also the worked example of how to extend the hierarchy for a specific
paradigm without changing the canonical BIDS storage layer.

## visual_search

::: pyxations.analysis.visual_search

# Data storage

The canonical storage layer shared by the raw and derivative datasets.

Derivative samples and eye-movement annotations are stored as compressed BIDS
`TSV.GZ` files with JSON sidecars, following the general BIDS Derivatives
conventions. BIDS does not yet define a domain-specific derivative schema for
detected eye movements, so the additional columns and the processing provenance
are documented in those sidecars.

## tables

The in-memory table container and the BIDS TSV read and write helpers. The
whole tabular pipeline uses Polars.

::: pyxations.tables

## export

The derivative reader and writer, including the initialization of a linked
derivative dataset.

::: pyxations.export.bids

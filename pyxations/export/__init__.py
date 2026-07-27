"""Derivative table exporters."""

from pyxations.export.feather import FeatherExport


BIDS_EXPORT = "bids"
FEATHER_EXPORT = "feather"
EXPORT_METHODS = [BIDS_EXPORT, FEATHER_EXPORT]


def get_exporter(exporter_label: str):
    """Return the configured derivative exporter.

    Parameters
    ----------
    exporter_label
        Export format identifier.

    Raises
    ------
    ValueError
        If an unsupported export format is requested.
    """
    if exporter_label == BIDS_EXPORT:
        from pyxations.export.bids import BIDSDerivativeExport

        return BIDSDerivativeExport()
    if exporter_label == FEATHER_EXPORT:
        return FeatherExport()
    raise ValueError(
        f"Unsupported export format {exporter_label!r}. "
        f"Supported formats: {EXPORT_METHODS}."
    )

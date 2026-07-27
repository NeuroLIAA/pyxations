import pytest

from pyxations.export import (
    BIDS_EXPORT,
    EXPORT_METHODS,
    FEATHER_EXPORT,
    FeatherExport,
    get_exporter,
)
from pyxations.export.bids import BIDSDerivativeExport


def test_bids_is_the_canonical_export_and_feather_remains_available():
    assert EXPORT_METHODS == [BIDS_EXPORT, FEATHER_EXPORT]
    assert isinstance(get_exporter(BIDS_EXPORT), BIDSDerivativeExport)
    assert isinstance(get_exporter(FEATHER_EXPORT), FeatherExport)


def test_removed_hdf_format_is_rejected_clearly():
    with pytest.raises(ValueError, match="Unsupported export format"):
        get_exporter("hdf5")

from pyxations.export.hdf import HDFExport
from pyxations.export.feather import FeatherExport
from pyxations.export.bids import BIDSDerivativeExport


HDF5_EXPORT = 'hdf5'
FEATHER_EXPORT = 'feather'
BIDS_EXPORT = 'bids'

EXPORT_METHODS = [BIDS_EXPORT, HDF5_EXPORT, FEATHER_EXPORT]

def get_exporter(exporter_label):
    if exporter_label == BIDS_EXPORT:
        return BIDSDerivativeExport()
    elif exporter_label == HDF5_EXPORT:
        return HDFExport()
    elif exporter_label == FEATHER_EXPORT:
        return FeatherExport()
    raise Exception(f'export_method should be one of these values: {EXPORT_METHODS}')

from pyxations.export.hdf import HDFExport
from pyxations.export.feather import FeatherExport


HDF5_EXPORT = 'hdf5'
FEATHER_EXPORT = 'feather'

EXPORT_METHODS = [HDF5_EXPORT, FEATHER_EXPORT]

def get_exporter(exporter_label):
    if exporter_label == HDF5_EXPORT:
        return HDFExport()
    elif exporter_label == FEATHER_EXPORT:
        return FeatherExport()
    raise Exception(f'export_method should be one of these values: {EXPORT_METHODS}')
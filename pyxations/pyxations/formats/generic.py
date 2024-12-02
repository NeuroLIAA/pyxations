'''
Created on 2 dic 2024

@author: placiana
'''
from pyxations.export import get_exporter


class BidsParse(object):
    def __init__(self, export_method)->None:
        self.export_method = get_exporter(export_method)
        object.__init__(self)


    def save_dataframe(self, df, path, data_name, key):
        self.export_method.save(df, path, data_name, key)


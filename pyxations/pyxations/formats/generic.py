'''
Created on 2 dic 2024

@author: placiana
'''
from pyxations.export import get_exporter
import pandas as pd


class BidsParse(object):
    def __init__(self, session_folder_path, export_method)->None:
        self.session_folder_path = session_folder_path
        self.export_method = get_exporter(export_method)
        object.__init__(self)


    def save_dataframe(self, df, path, data_name, key):
        self.export_method.save(df, path, data_name, key)


    def store_dataframes(self, dfSamples, dfCalib=pd.DataFrame(), dfFix=pd.DataFrame(), dfSacc=pd.DataFrame(), 
                         dfHeader=pd.DataFrame(),dfBlink=pd.DataFrame(), dfMsg=pd.DataFrame()):
                # Save DataFrames to disk in one go to minimize memory usage during processing
        detection_algorithm = self.detection_algorithm

        self.save_dataframe(dfSamples, self.session_folder_path, 'samples', key='samples')
        
        if not dfCalib.empty:
            self.save_dataframe(dfCalib, self.session_folder_path, 'calib', key='calib')
        if not dfHeader.empty:
            self.save_dataframe(dfHeader, self.session_folder_path, 'header', key='header')
        if not dfMsg.empty:
            self.save_dataframe(dfMsg, self.session_folder_path, 'msg', key='msg')

        
        (self.session_folder_path / f'{detection_algorithm}_events').mkdir(parents=True, exist_ok=True)
        #if not dfFix.empty:
        self.save_dataframe(dfFix, (self.session_folder_path / f'{detection_algorithm}_events'), 'fix', key='fix')
        #if not dfSacc.empty:
        self.save_dataframe(dfSacc, (self.session_folder_path / f'{detection_algorithm}_events'), 'sacc', key='sacc')
        #if not dfBlink.empty:
        self.save_dataframe(dfBlink, (self.session_folder_path / f'{detection_algorithm}_events'), 'blink', key='blink')        


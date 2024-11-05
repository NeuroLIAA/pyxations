'''
Created on 5 nov 2024

@author: placiana
'''
from pyxations.eye_movement_detection import EyeMovementDetection


class EngbertDetection(EyeMovementDetection):
    '''
    Python implementation for
    https://github.com/olafdimigen/eye-eeg/blob/master/detecteyemovements.m
    
    '''

    def __init__(self, session_folder_path, samples):
        self.session_folder_path = session_folder_path
        self.out_folder = (session_folder_path / 'engbert_detection')
        self.samples = samples

    def detect_eye_movements(self, vfac:float, mindur:int, degperpixel:float=0.1, 
                             smooth:bool=False, globalthreshold:bool=False, clusterdist:int=1,
                             clustermode:int=1 ):
        '''
        
        :param vfac: velocity factor ("lambda") to determine
%                  the velocity threshold for saccade detection
        :param mindur:  minimum saccade duration (in samples)
        :param degperpixel: visual angle of one screen pixel
%                  if this value is left empty [], saccade characteristics
%                  are reported in the original data metric (pixel?)
%                  instead of in degrees of visual angle
        :param smooth: if set to 1, the raw data is smoothed over a
%                  5-sample window to suppress noise
%                  noise. Recommended for high native ET sampling rates.
        :param globalthreshold: Use the same thresholds for all epochs?
%                  0: Adaptive velocity thresholds are computed
%                  individually for each data epoch.
%                  1: Adaptive velocity thresholds are first computed for
%                  each epoch, but then the mean thresholds are applied to
%                  each epochs (i.e. same detection parameters are used for
%                  all epochs). Setting is irrelevant if the input data is
%                  still continuous (= only one data epoch).
        :param clusterdist: value in sampling points that defines the
%                  minimum allowed fixation duration between two saccades.
%                  If the off- and onsets of two temp. adjacent sacc. are
%                  closer together than 'clusterdist' samples, these
%                  saccades are regarded as a "cluster" and treated
%                  according to the 'clustermode' setting (see below).
%                  clusterdist is irrelevant if clustermode == 1.
        :param clustermode: [1,2,3,4]. Integer between 1 and 4.
%                  1: keep all saccades, do nothing
%                  2: keep only first saccade of each cluster
%                  3: keep only largest sacc. of each cluster
%                  4: combine all movements into one (longer) saccade
%                     this new saccade is defined as the movement that
%                     occurs between the onset of the 1st saccade in the
%                     cluster and the offset of the last sacc. in cluster
%                     WARNING: CLUSTERMODE 4 is experimental and untested!
        '''
        
        pass
    
    

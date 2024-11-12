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


import numpy as np

def detect_eye_movements(EEG, left_eye_xy, right_eye_xy, vfac, mindur, degperpixel, smooth, globalthresh, clusterdist, clustermode, plotfig, writesac, writefix):
    allsac = []
    allfix = []

    # Check which eye data is available
    ldata = False
    if len(left_eye_xy) == 2:
        ldata = True
    
    rdata = False
    if len(right_eye_xy) == 2:
        rdata = True
    
    if len(left_eye_xy) == 1 or len(right_eye_xy) == 1:
        raise ValueError("For each recorded eye, horizontal (X) and vertical (Y) gaze channel must be specified.")

    nsample = EEG['data'].shape[1]
    nepochs = EEG['data'].shape[2]
    badepochs = np.zeros(nepochs, dtype=int)
    nbadsmp = 0

    # Preallocate storage for saccade detection thresholds
    l_msdx = np.full(nepochs, np.nan)
    l_msdy = np.full(nepochs, np.nan)
    r_msdx = np.full(nepochs, np.nan)
    r_msdy = np.full(nepochs, np.nan)

    # Warn message if back-to-back saccades are detected
    clusterwarning = False

    # Smoothing level based on smooth flag
    # smoothlevel 0: no smoothing, simple diff()
    # smoothlevel 1: 3-point window
    # smoothlevel 2: 5-point window
    smoothlevel = 2 if smooth else 0    


    # Initialize "badvector" with zeros, size based on EEG data dimensions
    badvector = np.zeros((1, EEG['data'].shape[1] * EEG['data'].shape[2]))
    
    # Find indices of events in EEG.event where type is 'bad_ET'
    ix_badETevent = [i for i, event in enumerate(EEG['event']) if event['type'] == 'bad_ET']
    if ix_badETevent:
        print("\nFound \"bad_ET\" events in EEG.events...")
        
        # Collect latency and duration for 'bad_ET' events
        bad_lat = np.array([EEG['event'][i]['latency'] for i in ix_badETevent])
        bad_dur = np.array([EEG['event'][i]['duration'] for i in ix_badETevent])
        bad_ET = np.column_stack((bad_lat, bad_dur))
        
        # Calculate end indices for each "bad_ET" interval
        bad_ET = np.hstack((bad_ET, (bad_ET[:, 0] + bad_ET[:, 1] - 1).reshape(-1, 1)))
        
        # Mark bad samples in "badvector"
        for j in range(bad_ET.shape[0]):
            start_idx = int(bad_ET[j, 0]) - 1  # Convert to 0-based index
            end_idx = int(bad_ET[j, 2]) - 1
            badvector[0, start_idx:end_idx + 1] = 1
    
    # Reshape "badvector" to 3D if data is already epoched
    badvector = badvector.reshape((1, EEG['data'].shape[1], EEG['data'].shape[2]))
    
    
    print("\nComputing adaptive velocity thresholds...")

    if np.any(badvector):
        print(f"\n-- Found {len(ix_badETevent)} \"bad_ET\" events marking bad eye-tracking intervals in EEG.event.")
        percent_bad = (np.sum(badvector) / badvector.size) * 100
        print(f"\n-- These intervals ({percent_bad:.2f}% of data) will be ignored when computing velocity thresholds.")
    
    # Loop over each epoch to compute thresholds
    for e in range(nepochs):
        # Index of good ET samples for current epoch
        ix_goodET = ~badvector[0, :, e].astype(bool)
        
        # Left eye data processing
        if ldata:
            l = EEG['data'][[left_eye_xy[0], left_eye_xy[1]], ix_goodET, e].T
            vl = vecvel(l, EEG['srate'], smoothlevel)  # Assuming vecvel function exists or needs implementation
            l_msdx[e], l_msdy[e] = velthresh(vl)       # Assuming velthresh function exists or needs implementation
        
        # Right eye data processing
        if rdata:
            r = EEG['data'][[right_eye_xy[0], right_eye_xy[1]], ix_goodET, e].T
            vr = vecvel(r, EEG['srate'], smoothlevel)
            r_msdx[e], r_msdy[e] = velthresh(vr)

    for e in range(nepochs):
    
        # Initialize saccades for this epoch
        sac = []
    
        # Saccades of the left eye
        if ldata:
            l = EEG['data'][[left_eye_xy[0], left_eye_xy[1]], :, e].T  # Do not exclude bad ET samples
            # Check for bad/missing samples
            badsmp = np.sum(l <= 0)
            if badsmp > 0:
                badepochs[e] = 1
                nbadsmp += badsmp
            
            # Calculate eye velocities for the left eye
            vl = vecvel(l, EEG['srate'], smoothlevel)
            
            # Detect monocular saccades for the left eye
            if globalthresh:
                # Use precomputed velocity thresholds (mean across all epochs)
                sacL = microsacc_plugin(l, vl, vfac, mindur, np.mean(l_msdx), np.mean(l_msdy))
            else:
                # Use velocity thresholds from this epoch only
                sacL = microsacc_plugin(l, vl, vfac, mindur, l_msdx[e], l_msdy[e])
        
        # Saccades of the right eye
        if rdata:
            r = EEG['data'][[right_eye_xy[0], right_eye_xy[1]], :, e].T
            # Check for bad/missing samples
            badsmp = np.sum(r <= 0)
            if badsmp > 0:
                badepochs[e] = 1
                nbadsmp += badsmp
            
            # Calculate eye velocities for the right eye
            vr = vecvel(r, EEG['srate'], smoothlevel)
            
            # Detect monocular saccades for the right eye
            if globalthresh:
                sacR = microsacc_plugin(r, vr, vfac, mindur, np.mean(r_msdx), np.mean(r_msdy))
            else:
                sacR = microsacc_plugin(r, vr, vfac, mindur, r_msdx[e], r_msdy[e])
        
        # Binocular saccades
        if ldata and rdata:
            sac, _, _ = binsacc(sacL, sacR)
            sac = saccpar(sac)  # Average saccade characteristics of both eyes
            sac = mergesacc(sac, (l + r) / 2, clusterdist, clustermode)  # Merge nearby saccades
        elif ldata:
            sac = sacL
            sac = saccpar([sac, sac])
            sac = mergesacc(sac, l, clusterdist, clustermode)
        elif rdata:
            sac = sacR
            sac = saccpar([sac, sac])
            sac = mergesacc(sac, r, clusterdist, clustermode)


        if sac.size > 0:
            
            # Define saccade duration as the difference between offset and onset samples
            sac[:, 2] = sac[:, 1] - sac[:, 0] + 1
            
            # Report saccade velocity, distance, and amplitude in visual angles
            sac[:, [4, 5, 7]] *= degperpixel
            
            # Convert saccade angles to degrees from radians
            sac[:, [6, 8]] *= (180 / np.pi)
            
            # Add the index of the corresponding data epoch
            sac[:, 9] = e
            
            # Store screen location for start and end of saccade
            if ldata and rdata:
                gazexy = (l + r) / 2  # Binocular recordings: average across eyes
            elif ldata:
                gazexy = l
            elif rdata:
                gazexy = r
            
            # Get positions immediately before saccade onset and after saccade offset
            startsmp = sac[:, 0].astype(int) - 1
            endsmp = sac[:, 1].astype(int) + 1
            
            # Fix out-of-bound indices
            startsmp[startsmp < 0] = 0
            endsmp[endsmp >= gazexy.shape[0]] = gazexy.shape[0] - 1
            
            # Add starting and ending gaze coordinates to saccade metrics
            sac[:, 10] = gazexy[startsmp, 0]  # x-coordinate at start
            sac[:, 11] = gazexy[startsmp, 1]  # y-coordinate at start
            sac[:, 12] = gazexy[endsmp, 0]    # x-coordinate at end
            sac[:, 13] = gazexy[endsmp, 1]    # y-coordinate at end

        # Remove saccades starting/ending during "bad_ET" intervals
        badETsmp = np.where(badvector[:, :, e].flatten())[0]
        if sac.size > 0:
            ix_fakesac = np.where(np.isin(sac[:, 0], badETsmp) | np.isin(sac[:, 1], badETsmp))[0]
            sac = np.delete(sac, ix_fakesac, axis=0)
        
        # Initialize fixations
        nsac = sac.shape[0]
        fix = []
        
        if nsac > 0:
            # Process each fixation
            for f in range(nsac - 1):
                fixation = [sac[f, 1] + 1, sac[f + 1, 0] - 1]
        
                # Handle back-to-back saccades with zero-duration fixations
                if fixation[0] > fixation[1]:
                    fixation[0] = fixation[1]
                    clusterwarning = True
        
                # Add saccade-related properties to fixation
                fixation.extend([None] * 7)  # Placeholder for properties (11-17)
                fixation[10] = e
                fixation[11:17] = sac[f, [4, 5, 6, 10, 11, 12, 13]]
        
                fix.append(fixation)
        
            fix = np.array(fix)
        
            # Add first fixation if epoch doesn't begin with a saccade
            if sac[0, 0] > 0:
                startfix = [0, sac[0, 0] - 1] + [None] * 7 + [e] + [np.nan] * 7
                fix = np.vstack([startfix, fix])
        
            # Add last fixation if epoch doesn't end with a saccade
            if sac[-1, 1] < nsample:
                lastfix = [sac[-1, 1] + 1, nsample - 1] + [None] * 7 + [e]
                lastfix[11:17] = sac[-1, [4, 5, 6, 10, 11, 12, 13]]
                fix = np.vstack([fix, lastfix])
        
            # Update fixation properties (duration, positions)
            for f in range(fix.shape[0]):
                # Fixation duration
                fix[f, 2] = fix[f, 1] - fix[f, 0] + 1
                
                # Mean fixation position (left eye)
                if ldata:
                    fix[f, 3] = np.mean(l[fix[f, 0]:fix[f, 1] + 1, 0])
                    fix[f, 4] = np.mean(l[fix[f, 0]:fix[f, 1] + 1, 1])
                else:
                    fix[f, 3:5] = np.nan, np.nan
                
                # Mean fixation position (right eye)
                if rdata:
                    fix[f, 5] = np.mean(r[fix[f, 0]:fix[f, 1] + 1, 0])
                    fix[f, 6] = np.mean(r[fix[f, 0]:fix[f, 1] + 1, 1])
                else:
                    fix[f, 5:7] = np.nan, np.nan
                
                # Binocular fixation position
                fix[f, 7] = np.nanmean([fix[f, 3], fix[f, 5]])
                fix[f, 8] = np.nanmean([fix[f, 4], fix[f, 6]])
        
            # Adjust saccade and fixation latencies in epoched data
            offset = (e - 1) * nsample
            sac[:, [0, 1]] += offset
            if fix.size > 0:
                fix[:, [0, 1]] += offset


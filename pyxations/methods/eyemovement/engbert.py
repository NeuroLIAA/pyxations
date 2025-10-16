'''
Created on 5 nov 2024

@author: placiana
'''
from pyxations.methods.eyemovement.eye_movement_detection import EyeMovementDetection
import numpy as np
import pandas as pd


def _smooth_1d(x, smoothlevel):
    if smoothlevel == 0 or x.size < 3:
        return x
    if smoothlevel == 1:  # 3-point [1,1,1]/3
        k = np.array([1., 1., 1.]) / 3.0
    elif smoothlevel == 2:  # 5-point [1,2,3,2,1]/9
        k = np.array([1., 2., 3., 2., 1.]) / 9.0
    else:
        k = np.array([1.])
    pad = (len(k) - 1) // 2
    xp = np.pad(x, (pad, pad), mode='edge')
    return np.convolve(xp, k, mode='valid')

def vecvel(gaze_xy, fs, smoothlevel=1):
    """
    gaze_xy: (N,2) positions in pixels (NaN for missing)
    returns velocities (N,2) in px/s using central differences (+ optional smoothing of positions)
    """
    x = _smooth_1d(gaze_xy[:,0].astype(float), smoothlevel)
    y = _smooth_1d(gaze_xy[:,1].astype(float), smoothlevel)

    vx = np.empty_like(x); vy = np.empty_like(y)
    vx[1:-1] = (x[2:] - x[:-2]) * (fs / 2.0)
    vy[1:-1] = (y[2:] - y[:-2]) * (fs / 2.0)
    vx[0] = (x[1] - x[0]) * fs
    vy[0] = (y[1] - y[0]) * fs
    vx[-1] = (x[-1] - x[-2]) * fs
    vy[-1] = (y[-1] - y[-2]) * fs
    vx[~np.isfinite(vx)] = np.nan
    vy[~np.isfinite(vy)] = np.nan
    return np.column_stack([vx, vy])

def _robust_std(v):
    med = np.nanmedian(v)
    med2 = np.nanmedian(v**2)
    s2 = med2 - med**2
    return np.sqrt(max(s2, 1e-12))

def velthresh(vxy):
    return _robust_std(vxy[:,0]), _robust_std(vxy[:,1])

def _find_runs(mask):
    """Return (start_idx, end_idx) inclusive for True-runs in a boolean array."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return np.empty((0,2), dtype=int)
    d = np.diff(mask.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends   = np.where(d == -1)[0]
    if mask[0]:
        starts = np.r_[0, starts]
    if mask[-1]:
        ends = np.r_[ends, mask.size-1]
    return np.column_stack([starts, ends]).astype(int)

def microsacc_plugin(pos_xy, vel_xy, vfac, mindur_samples, sdx, sdy):
    """
    Return saccades array with columns:
    0 onset, 1 offset, 2 dur, 3 avgvel(px/s), 4 vpeak(px/s), 5 dist(px),
    6 theta(rad), 7 amp(px), 8 dir(rad), 9 epoch (filled outside),
    10 x0, 11 y0, 12 x1, 13 y1
    """
    vx, vy = vel_xy[:,0], vel_xy[:,1]
    with np.errstate(invalid='ignore'):
        crit = (vx / sdx)**2 + (vy / sdy)**2
    suprath = crit > (vfac**2)
    runs = _find_runs(suprath)

    sac = []
    for s0, s1 in runs:
        if (s1 - s0 + 1) < mindur_samples:
            continue
        seg_v = np.hypot(vx[s0:s1+1], vy[s0:s1+1])
        vpeak = np.nanmax(seg_v)
        avgv  = np.nanmean(seg_v)

        x0, y0 = pos_xy[s0,0], pos_xy[s0,1]
        x1, y1 = pos_xy[s1,0], pos_xy[s1,1]
        amp = np.hypot(x1 - x0, y1 - y0)
        theta = np.arctan2(y1 - y0, x1 - x0)  # main vector angle

        seg = pos_xy[s0:s1+1]
        dist = np.nansum(np.hypot(np.diff(seg[:,0]), np.diff(seg[:,1])))

        sac.append([s0, s1, (s1 - s0 + 1), avgv, vpeak, dist, theta, amp, theta, np.nan, x0, y0, x1, y1])
    return np.array(sac, dtype=float)

# =========================
# Chunking & column helpers
# =========================

def _available_eye_columns(df):
    """Return dict for left/right or generic eye presence."""
    cols = set(df.columns)
    has_L = {'LX','LY'}.issubset(cols)
    has_R = {'RX','RY'}.issubset(cols)
    has_generic = {'X','Y'}.issubset(cols)
    return dict(has_L=has_L, has_R=has_R, has_generic=has_generic)

def _compute_px2deg(screen_size_cm, screen_distance_cm, screen_width_px):
    # degrees per pixel
    return np.degrees(np.arctan2(0.5*screen_size_cm, screen_distance_cm)) / (0.5*screen_width_px)

def _split_into_chunks(df, fallback_fs=None):
    """
    Create chunk labels where sampling rate is stable and time gaps are reasonable.
    If 'Rate_recorded' missing, use fallback_fs as constant.
    """
    df = df.copy()
    if 'Rate_recorded' in df.columns and df['Rate_recorded'].notna().any():
        fs_series = df['Rate_recorded'].ffill().bfill()
    else:
        if not fallback_fs:
            raise ValueError("Rate_recorded not present and no fallback sample_rate provided.")
        fs_series = pd.Series(np.full(len(df), float(fallback_fs)), index=df.index)

    # New chunk when fs changes or time gap too large relative to previous fs
    t = df['tSample'].values.astype(float)  # assumed ms
    fs = fs_series.values.astype(float)

    chunk = np.zeros(len(df), dtype=int)
    for i in range(1, len(df)):
        fs_changed = not np.isclose(fs[i], fs[i-1], rtol=0, atol=1e-6)
        expected_dt = 1000.0 / fs[i-1] if np.isfinite(fs[i-1]) and fs[i-1] > 0 else np.inf
        observed_dt = t[i] - t[i-1]
        # tolerate some jitter; start new chunk if huge jump (>1.5x expected)
        big_gap = observed_dt > (1.5 * expected_dt)
        if fs_changed or big_gap:
            chunk[i] = 1
    chunk_ids = np.cumsum(chunk)
    return chunk_ids, fs_series





class EngbertDetection(EyeMovementDetection):
    '''
    Python implementation for
    https://github.com/olafdimigen/eye-eeg/blob/master/detecteyemovements.m
    
    '''

    def __init__(self, session_folder_path, samples):
        self.session_folder_path = session_folder_path
        self.out_folder = (session_folder_path / 'engbert_events')
        self.samples = samples

    def detect_eye_movements(
            self,
            vfac: float = 5.0,
            mindur_ms: float = 6.0,
            smoothlevel: int = 1,
            globalthresh: bool = True,
            # deg/px: either give degperpixel OR let it be computed from screen params
            degperpixel: float | None = None,
            screen_size_cm: float = 38.0,
            screen_width_px: int = 1920,
            screen_distance_cm: float = 60.0,
            sample_rate_fallback: float | None = None,
        ):
            """
            Returns (fixations_df, saccades_df) with times in **ms**.
            Columns:
            Saccades: ['tStart','tEnd','duration','xStart','yStart','xEnd','yEnd','ampDeg','vPeak','distDeg','thetaDeg','eye','Calib_index','Eyes_recorded','Rate_recorded','chunk']
            Fixations: ['tStart','tEnd','duration','xAvg','yAvg','pupilAvg','eye','Calib_index','Eyes_recorded','Rate_recorded','chunk']
            """
            df = self.samples.copy()
            if degperpixel is None:
                degperpixel = _compute_px2deg(screen_size_cm, screen_distance_cm, screen_width_px)

            chunk_ids, fs_series = _split_into_chunks(df, fallback_fs=sample_rate_fallback)
            df['_chunk'] = chunk_ids
            df['_fs'] = fs_series.values

            sac_rows = []
            fix_rows = []

            eye_cols = _available_eye_columns(df)

            # Process each chunk & each available eye
            for chunk_id, g in df.groupby('_chunk', sort=True):
                fs = float(g['_fs'].iloc[0])
                t0_ms = float(g['tSample'].iloc[0])
                calib = g['Calib_index'].iloc[0] if 'Calib_index' in g else np.nan
                eyes_rec = g['Eyes_recorded'].iloc[0] if 'Eyes_recorded' in g else np.nan

                # choose which eye streams exist in this chunk
                streams = []
                if eye_cols['has_L'] and g[['LX','LY']].notna().any().any():
                    streams.append(('L', 'LX', 'LY', 'LPupil' if 'LPupil' in g else None))
                if eye_cols['has_R'] and g[['RX','RY']].notna().any().any():
                    streams.append(('R', 'RX', 'RY', 'RPupil' if 'RPupil' in g else None))
                if (not streams) and eye_cols['has_generic'] and g[['X','Y']].notna().any().any():
                    streams.append(('U', 'X', 'Y', 'Pupil' if 'Pupil' in g else None))  # U = unknown/unspecified eye

                if not streams:
                    continue  # nothing to do in this chunk

                # Prepare global thresholds (if requested)
                global_sigmas = {}
                if globalthresh:
                    for eye_label, cx, cy, _ in streams:
                        xy = g[[cx, cy]].to_numpy(dtype=float)
                        vel = vecvel(xy, fs, smoothlevel=smoothlevel)
                        sdx, sdy = velthresh(vel)
                        global_sigmas[eye_label] = (sdx, sdy)

                # Minimum duration in samples
                mindur_samples = max(1, int(round(mindur_ms * fs / 1000.0)))

                for eye_label, cx, cy, cp in streams:
                    xy = g[[cx, cy]].to_numpy(dtype=float)
                    # mark bad rows where both x & y are missing
                    valid = np.isfinite(xy).all(axis=1)
                    # If everything is invalid, skip
                    if not np.any(valid):
                        continue

                    vel = vecvel(xy, fs, smoothlevel=smoothlevel)

                    if globalthresh:
                        sdx, sdy = global_sigmas[eye_label]
                    else:
                        sdx, sdy = velthresh(vel)

                    # handle degenerate (all-NaN) sigmas
                    if not np.isfinite(sdx) or sdx == 0:
                        sdx = np.nanmedian(np.abs(vel[:,0])) or 1.0
                    if not np.isfinite(sdy) or sdy == 0:
                        sdy = np.nanmedian(np.abs(vel[:,1])) or 1.0

                    sac = microsacc_plugin(xy, vel, vfac=vfac, mindur_samples=mindur_samples, sdx=sdx, sdy=sdy)
                    if sac.size == 0:
                        # still may have one big fixation (whole chunk)
                        # synthesize a single fixation over all finite samples
                        idx = np.where(np.isfinite(xy[:,0]) & np.isfinite(xy[:,1]))[0]
                        if idx.size:
                            s_idx = int(idx[0]); e_idx = int(idx[-1])
                            # ms times aligned to t0_ms
                            tStart = t0_ms + (s_idx / fs) * 1000.0
                            tEnd   = t0_ms + (e_idx / fs) * 1000.0
                            dur    = (e_idx - s_idx + 1) / fs * 1000.0
                            xAvg = float(np.nanmean(xy[s_idx:e_idx+1,0]))
                            yAvg = float(np.nanmean(xy[s_idx:e_idx+1,1]))
                            pupilAvg = float(np.nanmean(g[cp].values[s_idx:e_idx+1])) if cp in g else np.nan
                            fix_rows.append([tStart, tEnd, dur, xAvg, yAvg, pupilAvg, eye_label, calib, eyes_rec, fs, chunk_id])
                        continue

                    # Fill epoch column (not used here) and convert to ms/deg
                    # sac columns: [0 onset, 1 offset, 2 dur(samples), 3 avgV(px/s), 4 vPeak(px/s), 5 dist(px), 6 theta(rad), 7 amp(px), 8 dir(rad), 9 epoch, 10 x0, 11 y0, 12 x1, 13 y1]
                    onset_idx = sac[:,0].astype(int)
                    offset_idx = sac[:,1].astype(int)

                    tStart = t0_ms + (onset_idx / fs) * 1000.0
                    tEnd   = t0_ms + (offset_idx / fs) * 1000.0
                    dur_ms = (sac[:,2] / fs) * 1000.0

                    vPeak_deg = sac[:,4] * degperpixel            # px/s → deg/s
                    dist_deg  = sac[:,5] * degperpixel            # px → deg
                    amp_deg   = sac[:,7] * degperpixel            # px → deg
                    theta_deg = sac[:,6] * (180.0 / np.pi)

                    x0, y0, x1, y1 = sac[:,10], sac[:,11], sac[:,12], sac[:,13]

                    # Build saccade rows
                    for i in range(sac.shape[0]):
                        sac_rows.append([
                            float(tStart[i]), float(tEnd[i]), float(dur_ms[i]),
                            float(x0[i]), float(y0[i]), float(x1[i]), float(y1[i]),
                            float(amp_deg[i]), float(vPeak_deg[i]), float(dist_deg[i]), float(theta_deg[i]),
                            eye_label, calib, eyes_rec, fs, chunk_id
                        ])

                    # Build inter-saccadic fixations inside the chunk
                    # Sort events by onset
                    order = np.argsort(onset_idx)
                    onset_sorted = onset_idx[order]
                    offset_sorted = offset_idx[order]

                    # Leading fixation
                    if onset_sorted[0] > 0:
                        s_idx = 0
                        e_idx = onset_sorted[0] - 1
                        tS = t0_ms + (s_idx / fs) * 1000.0
                        tE = t0_ms + (e_idx / fs) * 1000.0
                        dur = (e_idx - s_idx + 1) / fs * 1000.0
                        xAvg = float(np.nanmean(xy[s_idx:e_idx+1,0]))
                        yAvg = float(np.nanmean(xy[s_idx:e_idx+1,1]))
                        pupilAvg = float(np.nanmean(g[cp].values[s_idx:e_idx+1])) if cp in g else np.nan
                        fix_rows.append([tS, tE, dur, xAvg, yAvg, pupilAvg, eye_label, calib, eyes_rec, fs, chunk_id])

                    # Middle fixations
                    for k in range(len(onset_sorted) - 1):
                        s_idx = offset_sorted[k] + 1
                        e_idx = onset_sorted[k+1] - 1
                        if e_idx < s_idx:
                            continue
                        tS = t0_ms + (s_idx / fs) * 1000.0
                        tE = t0_ms + (e_idx / fs) * 1000.0
                        dur = (e_idx - s_idx + 1) / fs * 1000.0
                        xAvg = float(np.nanmean(xy[s_idx:e_idx+1,0]))
                        yAvg = float(np.nanmean(xy[s_idx:e_idx+1,1]))
                        pupilAvg = float(np.nanmean(g[cp].values[s_idx:e_idx+1])) if cp in g else np.nan
                        fix_rows.append([tS, tE, dur, xAvg, yAvg, pupilAvg, eye_label, calib, eyes_rec, fs, chunk_id])

                    # Trailing fixation
                    last_off = int(offset_sorted[-1])
                    if last_off < (len(g)-1):
                        s_idx = last_off + 1
                        e_idx = len(g) - 1
                        tS = t0_ms + (s_idx / fs) * 1000.0
                        tE = t0_ms + (e_idx / fs) * 1000.0
                        dur = (e_idx - s_idx + 1) / fs * 1000.0
                        xAvg = float(np.nanmean(xy[s_idx:e_idx+1,0]))
                        yAvg = float(np.nanmean(xy[s_idx:e_idx+1,1]))
                        pupilAvg = float(np.nanmean(g[cp].values[s_idx:e_idx+1])) if cp in g else np.nan
                        fix_rows.append([tS, tE, dur, xAvg, yAvg, pupilAvg, eye_label, calib, eyes_rec, fs, chunk_id])

            # Assemble DataFrames
            saccades = pd.DataFrame(
                sac_rows,
                columns=['tStart','tEnd','duration','xStart','yStart','xEnd','yEnd','ampDeg','vPeak','distDeg','thetaDeg','eye','Calib_index','Eyes_recorded','Rate_recorded','chunk']
            ).sort_values('tEnd', ignore_index=True)

            fixations = pd.DataFrame(
                fix_rows,
                columns=['tStart','tEnd','duration','xAvg','yAvg','pupilAvg','eye','Calib_index','Eyes_recorded','Rate_recorded','chunk']
            ).sort_values('tEnd', ignore_index=True)

            return fixations, saccades
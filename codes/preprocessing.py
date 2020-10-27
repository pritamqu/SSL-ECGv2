import numpy as np
from biosppy.signals import ecg as ecg_func
from biosppy.signals import bvp as ppg_func
from biosppy.signals import tools as tools

# signal= np.loadtxt('D:\\datasets\\Biosignals\\final_WESAD\\256\\ecg\\S2.txt')

compare_ecg_segments = ecg_func.compare_segmentation 

def filter_ecg(signal, sampling_rate):
    
    signal = np.array(signal)
    order = int(0.3 * sampling_rate)
    filtered, _, _ = tools.filter_signal(signal=signal,
                                  ftype='FIR',
                                  band='bandpass',
                                  order=order,
                                  frequency=[3, 45],
                                  sampling_rate=sampling_rate)
    
    return filtered

def filter_ppg(signal, sampling_rate):
    
    signal = np.array(signal)
    sampling_rate = float(sampling_rate)
    filtered, _, _ = tools.filter_signal(signal=signal,
                                  ftype='butter',
                                  band='bandpass',
                                  order=4,
                                  frequency=[1, 8],
                                  sampling_rate=sampling_rate)

    return filtered

def get_Rpeaks_ECG(filtered, sampling_rate):
    
    # segment
    rpeaks, = ecg_func.hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)

    # correct R-peak locations
    rpeaks, = ecg_func.correct_rpeaks(signal=filtered,
                             rpeaks=rpeaks,
                             sampling_rate=sampling_rate,
                             tol=0.05)
    rr_intervals = np.diff(rpeaks)

    return rpeaks, rr_intervals

def get_Rpeaks_PPG(filtered, sampling_rate):
    
    # segment
    rpeaks, = ppg_func.find_onsets(signal=filtered, sampling_rate=sampling_rate)
    rr_intervals = np.diff(rpeaks)
   
    return rpeaks, rr_intervals

def heartbeats_ecg(filtered, sampling_rate):
    
    # segment
    rpeaks, = ecg_func.hamilton_segmenter(signal=filtered, sampling_rate=sampling_rate)

    # correct R-peak locations
    rpeaks, = ecg_func.correct_rpeaks(signal=filtered,
                             rpeaks=rpeaks,
                             sampling_rate=sampling_rate,
                             tol=0.05)


    # compute heart rate
    hr_idx, hr = tools.get_heart_rate(beats=rpeaks,
                                   sampling_rate=sampling_rate,
                                   smooth=True,
                                   size=3)
    
    return hr_idx, hr


def heartbeats_ppg(filtered, sampling_rate):
    
    onsets, = ppg_func.find_onsets(signal=filtered, sampling_rate=sampling_rate)

    # compute heart rate
    hr_idx, hr = tools.get_heart_rate(beats=onsets,
                                   sampling_rate=sampling_rate,
                                   smooth=True,
                                   size=3)
    
    
    return hr_idx, hr
    
    
# def ecg_beats_per_min(ecg_signal, sampling_rate=128, window=6, sliding=2):
    
#     # duration, sliding is in seconds --> while calculating HR
    
#     if sliding==None:
#         sliding=window

#     rpeaks, _ = get_Rpeaks_ECG(ecg_signal, sampling_rate)
#     final_bpm = []
#     for k in range(0, len(ecg_signal)//(sampling_rate*sliding)):
#         start = k*sampling_rate*sliding
#         end = start + sampling_rate*window
#         bpm = len(np.where(np.logical_and(rpeaks>=start, rpeaks<=end))[0]) * (60//window)
#         final_bpm.append(bpm)    
        
#     return final_bpm
    
# def ppg_beats_per_min(ppg_signal, sampling_rate=128, window=6, sliding=2):
    
#     # duration, sliding is in seconds --> while calculating HR
    
#     if sliding==None:
#         sliding=window

#     rpeaks, _ = get_Rpeaks_PPG(ppg_signal, sampling_rate)
#     final_bpm = []
#     for k in range(0, len(ppg_signal)//(sampling_rate*sliding)):
#         start = k*sampling_rate*sliding
#         end = start + sampling_rate*window
#         bpm = len(np.where(np.logical_and(rpeaks>=start, rpeaks<=end))[0]) * (60//window)
#         final_bpm.append(bpm)    
        
#     return final_bpm
    
    
def ecg_bpm(ecg_signal, sampling_rate=128, window=6, sliding=2):
    
    # duration, sliding is in seconds --> while calculating HR
    
    if sliding==None:
        sliding=window

    final_bpm = []
    for k in range(0, len(ecg_signal)//(sampling_rate*sliding)):
        start = k*sampling_rate*sliding
        end = start + sampling_rate*window
        sample = ecg_signal[start:end]
        hr_idx, hr = heartbeats_ecg(sample, sampling_rate)
        bpm = np.mean(hr)
        final_bpm.append(bpm)    
        
    return final_bpm    
    
    
    
def ppg_bpm(ppg_signal, sampling_rate=128, window=6, sliding=2):
    
    # duration, sliding is in seconds --> while calculating HR
    
    if sliding==None:
        sliding=window

    final_bpm = []
    for k in range(0, len(ppg_signal)//(sampling_rate*sliding)):
        start = k*sampling_rate*sliding
        end = start + sampling_rate*window
        sample = ppg_signal[start:end]
        hr_idx, hr = heartbeats_ppg(sample, sampling_rate)
        bpm = np.mean(hr)
        final_bpm.append(bpm)    
        
    return final_bpm    
        
    
def error_bpm_ecg_ppg(ecg_signal, ppg_signal, sampling_rate=128, window=6, sliding=2):
    
    ebpm = ecg_bpm(ecg_signal, sampling_rate=sampling_rate, window=window, sliding=sliding)
    pbpm = ppg_bpm(ppg_signal, sampling_rate=sampling_rate, window=window, sliding=sliding)
       
    error = np.subtract(np.asarray(ebpm), np.asarray(pbpm))
    error_mean = np.mean(error)
    error_sd = np.std(error)
    
    return error_mean, error_sd
    
    
    
    

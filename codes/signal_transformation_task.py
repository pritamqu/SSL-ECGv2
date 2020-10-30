import os
import numpy as np
import math
import matplotlib.pyplot as plt
import cv2

## get filename during run time
dirname, filename = os.path.split(os.path.abspath(__file__))
print("running: {}".format(filename) )
## change directory to the current directory where code is saved

   
def add_noise_with_SNR(signal, noise_amount):
    """ 
    adding noise
    noise amount limit 0.005 - 0.05 """
    
    target_snr_db = noise_amount #20
    # Calculate signal power and convert to dB 
    x_watts = signal ** 2
#    x_db = 10 * np.log10(x_watts)
    
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), len(x_watts))
    noised_signal = signal + noise_volts

    return noised_signal 

def scaled(signal, factor):
    """"
    scale the signal
    scaling factor limit 0.2 - 2 """
    scaled_signal = signal * factor
    return scaled_signal
 

def negate(signal):
    """ 
    negate the signal """
    negated_signal = signal * (-1)
    return negated_signal

    
def hor_filp(signal):
    """ 
    flipped horizontally """
    hor_flipped = np.flip(signal)
    return hor_flipped

## permuted

def permute(signal, pieces):
    """ 
    signal permutation
    number of pieces limit 2-20 """
    pieces       = int(np.ceil(np.shape(signal)[0]/(np.shape(signal)[0]//pieces)).tolist())
    piece_length = int(np.shape(signal)[0]//pieces)
    
    sequence = list(range(0,pieces))
    np.random.shuffle(sequence)
    
    permuted_signal = np.reshape(signal[:(np.shape(signal)[0]//pieces*pieces)], (pieces, piece_length)).tolist() + [signal[(np.shape(signal)[0]//pieces*pieces):]]
    permuted_signal = np.asarray(permuted_signal)[sequence]
    permuted_signal = np.hstack(permuted_signal)
     
        
    return permuted_signal

    
## timewarping
    
def time_warp(signal, sampling_freq, pieces, stretch_factor, squeeze_factor):
    """ 
    signal time warping: stretch and squeeze some part of the signal
    bellow is the best limit of the parameters
    slices should be factor of time_length
    stretch_factor = 1.2
    squeeze_factor = 1.2
    pieces = 6, 2, 
    sampling_freq = 256 """
    total_time = np.shape(signal)[0]//sampling_freq
    segment_time = total_time/pieces
    sequence = list(range(0,pieces))
    stretch = np.random.choice(sequence, math.ceil(len(sequence)/2), replace = False)
    squeeze = list(set(sequence).difference(set(stretch)))
    initialize = True
    for i in sequence:
        orig_signal = signal[int(i*np.floor(segment_time*sampling_freq)):int((i+1)*np.floor(segment_time*sampling_freq))]
        orig_signal = orig_signal.reshape(np.shape(orig_signal)[0],1)
        if i in stretch:
            output_shape = int(np.ceil(np.shape(orig_signal)[0]*stretch_factor))
            new_signal = cv2.resize(orig_signal, (1, output_shape), interpolation=cv2.INTER_LINEAR)
            if initialize == True:
                time_warped = new_signal
                initialize = False
            else:
                time_warped = np.vstack((time_warped, new_signal))
        elif i in squeeze:
            output_shape = int(np.ceil(np.shape(orig_signal)[0]*squeeze_factor))
            new_signal = cv2.resize(orig_signal, (1, output_shape), interpolation=cv2.INTER_LINEAR)
            if initialize == True:
                time_warped = new_signal
                initialize = False
            else:
                time_warped = np.vstack((time_warped, new_signal))
    return time_warped
   


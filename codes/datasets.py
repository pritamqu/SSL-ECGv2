# -*- coding: utf-8 -*-
"""
Created on Wed Apr 29 22:13:28 2020

@author: pritam
"""
import os
## get filename during run time and set pwd manually
dirname, filename = os.path.split(os.path.abspath(__file__))
print("running: {}".format(filename) )

from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
# import amigos_dataset
# import dreamer_dataset
# import swell_dataset
# import wesad_dataset
# import martins_dataset
# import sqi_ecg 
# import warnings

# def save_dataset(overlap_pct=1, 
#                  window_size_sec=10, 
#                  fs = 256,
#                  data_path = os.path.join(Path(dirname).parent, 'data')):

#     amigos_data = amigos_dataset.extract_amigos_dataset(overlap_pct, window_size_sec, fs, data_path) 
#     dreamer_data = dreamer_dataset.extract_dreamer_dataset(overlap_pct, window_size_sec, fs, data_path)
#     swell_data = swell_dataset.extract_swell_dataset(overlap_pct, window_size_sec, fs, data_path)
#     wesad_data = wesad_dataset.extract_wesad_dataset(overlap_pct, window_size_sec, fs, data_path)
#     martin_data = martins_dataset.extract_martins_dataset(overlap_pct, window_size_sec, fs, data_path)
    
#     return amigos_data, dreamer_data, swell_data, wesad_data, martin_data

def load_data(path):
    dataset = np.load(path, allow_pickle=True)     
    return dataset 

   
    
def train_test_split(data_folder, test_pct):
    
    swell_data              = load_data(os.path.join(data_folder, 'swell.npy'))
    dreamer_data            = load_data(os.path.join(data_folder, 'dreamer.npy'))   
    amigos_data             = load_data(os.path.join(data_folder, 'amigos.npy'))
    wesad_data              = load_data(os.path.join(data_folder, 'wesad.npy'))
    
    amigos_train_index, amigos_test_index = get_train_test_index(amigos_data, test_pct)
    dreamer_train_index, dreamer_test_index = get_train_test_index(dreamer_data, test_pct)
    swell_train_index, swell_test_index = get_train_test_index(swell_data, test_pct)
    wesad_train_index, wesad_test_index = get_train_test_index(wesad_data, test_pct)
    
    train_data = np.vstack((amigos_data[amigos_train_index], dreamer_data[dreamer_train_index], swell_data[swell_train_index], wesad_data[wesad_train_index]))
    test_data = np.vstack((amigos_data[amigos_test_index], dreamer_data[dreamer_test_index], swell_data[swell_test_index], wesad_data[wesad_test_index]))
    
    return train_data, test_data
    
def train_test_split_martin_loso(data_folder, test_pct, overlap_pct, type_m_or_f):
    
    def _get_train_test_index(data, test_pct):
    
        train_subs, test_subs = _train_test_subjects(data, test_pct)
    
        test_index = np.zeros((1,1))
        for k in test_subs:
            index = np.where(data[:,0] == k)
            if np.all(test_index ==0):
                test_index = index[0]
            else:
                test_index = np.hstack((test_index, index[0]))
                
        train_index = np.setdiff1d(np.arange(len(data)), test_index)
                
        return train_index, test_index
    
    def _train_test_subjects(data, test_pct):
        np.random.seed(12223)
        person = np.unique(data[:, 0])
        no_test_subs = int(np.round(len(person) * test_pct))
        test_subs = np.random.choice(person, size= no_test_subs, replace=False)   
        train_subs = np.setdiff1d(person, test_subs)
        
        return train_subs, test_subs


    martins_data              = load_data(os.path.join(data_folder, 'martins_'+ type_m_or_f + '_' + str(overlap_pct)+'.npy'))
    martin_train_index, martin_test_index = _get_train_test_index(martins_data, test_pct)
    train_data = martins_data[martin_train_index]
    test_data = martins_data[martin_test_index]
    
    return train_data, test_data

def train_test_split_martin_kfold(data_folder, kfold, total_fold, overlap_pct, type_m_or_f='mecg'):
    
    def _get_train_test_index(data, kfold):
        np.random.seed(999999)
        person = np.unique(data[:, 0])
    
        test_index = np.zeros((1,1))
        for k in person:
            index = np.where(data[:,0] == k)[0]
            np.random.shuffle(index)
            l = len(index)
            start = kfold*int(l//total_fold)
            end = start + int(l//total_fold)
            
            if np.all(test_index ==0):
                test_index = index[start:end]
            else:
                test_index = np.hstack((test_index, index[start:end]))
                
        train_index = np.setdiff1d(np.arange(len(data)), test_index)
                
        return train_index, test_index
    
    martins_data              = load_data(os.path.join(data_folder, 'martins_'+ type_m_or_f + '_' + str(overlap_pct)+'.npy'))
    martin_train_index, martin_test_index = _get_train_test_index(martins_data, kfold)
    train_data = martins_data[martin_train_index]
    test_data = martins_data[martin_test_index]
    
    return train_data, test_data

# def train_test_split_martin_kfold(data_folder, total_fold, overlap_pct, type_m_or_f='mecg'):
    
#     data              = load_data(os.path.join(data_folder, 'martins_'+ type_m_or_f + '_' + str(overlap_pct)+'.npy'))
#     person = np.unique(data[:, 0])
#     final_test_index = []
#     final_train_index = []
#     for pid in person:
#         test_index = []
#         train_index = []
#         index = np.where(data[:,0] == pid)[0]
#         kf = KFold(n_splits=total_fold, random_state=9999, shuffle=False)
#         for tr, te in kf.split(index):
#             train_index.append(tr)
#             test_index.append(te)
            
#         final_train_index.append(train_index)
#         final_test_index.append(test_index)
    
    
#     return final_train_index, final_test_index
    
def train_test_subjects(dataset, test_pct):
    np.random.seed(12223)
    person = np.unique(dataset[:, 1])
    no_test_subs = int(np.round(len(person) * test_pct))
    test_subs = np.random.choice(person, size= no_test_subs, replace=False)   
    train_subs = np.setdiff1d(person, test_subs)
    
    return train_subs, test_subs

def get_train_test_index(data, test_pct):
    
    train_subs, test_subs = train_test_subjects(data, test_pct)

    test_index = np.zeros((1,1))
    for k in test_subs:
        index = np.where(data[:,1] == k)
        if np.all(test_index ==0):
            test_index = index[0]
        else:
            test_index = np.hstack((test_index, index[0]))
            
    train_index = np.setdiff1d(np.arange(len(data)), test_index)
            
    return train_index, test_index

def train_test_index_martin(data, test_pct):
    
    def _train_test_subjects(dataset, test_pct):
        np.random.seed(12223)
        person = np.unique(dataset[:, 0])
        no_test_subs = int(np.round(len(person) * test_pct))
        test_subs = np.random.choice(person, size= no_test_subs, replace=False)   
        train_subs = np.setdiff1d(person, test_subs)
        
        return train_subs, test_subs
    
    train_subs, test_subs = _train_test_subjects(data, test_pct)

    test_index = np.zeros((1,1))
    for k in test_subs:
        index = np.where(data[:,0] == k)
        if np.all(test_index ==0):
            test_index = index[0]
        else:
            test_index = np.hstack((test_index, index[0]))
            
    train_index = np.setdiff1d(np.arange(len(data)), test_index)
            
    return train_index, test_index
                                   

# def save_sqi_ecg(data_folder):
    
#     # warnings.filterwarnings("ignore")
    
#     swell_data              = load_data(os.path.join(data_folder, 'swell.npy'))
#     dreamer_data            = load_data(os.path.join(data_folder, 'dreamer.npy'))   
#     amigos_data             = load_data(os.path.join(data_folder, 'amigos.npy'))
#     wesad_data              = load_data(os.path.join(data_folder, 'wesad.npy'))
#     martins_data            = load_data(os.path.join(data_folder, 'martins.npy'))
    
#     swell_sqi, swell_data_sqi = sqi_ecg.SQI(swell_data)
#     dreamer_sqi, dreamer_data_sqi = sqi_ecg.SQI(dreamer_data)
#     amigos_sqi, amigos_data_sqi = sqi_ecg.SQI(amigos_data)
#     wesad_sqi, wesad_data_sqi = sqi_ecg.SQI(wesad_data)
#     martins_sqi, martins_data_sqi = sqi_ecg.SQI(martins_data)
    
#     np.save(os.path.join(data_folder, 'train_test_sqi', 'swell.npy'), swell_data_sqi)
#     np.save(os.path.join(data_folder, 'train_test_sqi', 'dreamer.npy'), dreamer_data_sqi)
#     np.save(os.path.join(data_folder, 'train_test_sqi', 'amigos.npy'), amigos_data_sqi)
#     np.save(os.path.join(data_folder, 'train_test_sqi', 'wesad.npy'), wesad_data_sqi)
#     np.save(os.path.join(data_folder, 'train_test_sqi', 'martins.npy'), martins_data_sqi)
    
    
# def save_sqi_ecg(data_folder, dataset):
    
#     # warnings.filterwarnings("ignore")
    
#     data              = load_data(os.path.join(data_folder, dataset))
#     sqi, data_sqi = sqi_ecg.SQI(data)
    
#     np.save(os.path.join(data_folder, 'train_test_sqi', dataset), data_sqi)

   
# amigos_dataset = amigos_dataset.extract_amigos_dataset()
# dreamer_dataset = dreamer_dataset.extract_dreamer_dataset()
# wesad_dataset = wesad_dataset.extract_wesad_dataset()
# swell_dataset = swell_dataset.extract_swell_dataset()

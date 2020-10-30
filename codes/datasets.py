import os
## get filename during run time and set pwd manually
dirname, filename = os.path.split(os.path.abspath(__file__))
print("running: {}".format(filename) )

from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold

def load_data(path):
    dataset = np.load(path, allow_pickle=True)     
    return dataset 

def train_test_split_felicity_kfold(data_folder, kfold, total_fold, overlap_pct, type_m_or_f='mecg'):
    
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
    
    felicitys_data              = load_data(os.path.join(data_folder, 'felicitys_'+ type_m_or_f + '_' + str(overlap_pct)+'.npy'))
    felicity_train_index, felicity_test_index = _get_train_test_index(felicitys_data, kfold)
    train_data = felicitys_data[felicity_train_index]
    test_data = felicitys_data[felicity_test_index]
    
    return train_data, test_data

    
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

def train_test_index_felicity(data, test_pct):
    
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
                                   

import os
## get filename during run time and set pwd manually
dirname, filename = os.path.split(os.path.abspath(__file__))
print("running: {}".format(filename) )


from scipy.io import loadmat
import numpy as np
from pathlib import Path
from tqdm import tqdm
import cv2
import pandas as pd
from scipy import signal as scipy_signal
import matplotlib.pyplot as plt
# import neurokit as nk
import utils
import preprocessing
import sklearn.preprocessing as skp


def import_sqi(path = Path('D:\\datasets\\Biosignals\\raw_downloads\\University_of_Washington_ECG_Martin\\sqi NEW SAVER algo\\sqi'),
               new_path = Path('D:\\datasets\\Biosignals\\final_ECG_Martin'),
               type_m_or_f = 'mecg'
               ):
    
    if type_m_or_f == 'mecg':
        new_path = new_path/ 'msqi'
        utils.makedirs(new_path)
        path = os.path.join(path, 'msqi')
        files, _ = utils.import_filenames(path)
    elif type_m_or_f == 'fecg':
        new_path = new_path/ 'fsqi'
        utils.makedirs(new_path)
        path = os.path.join(path, 'fsqi')
        files, _ = utils.import_filenames(path)
        
    for name in tqdm(files):
        if name.endswith('.mat'):
            filename        = os.path.join(path, name)
            sqi             = loadmat(filename)
            if type_m_or_f == 'mecg':
                sqi             = sqi['sqi_m1']
            elif type_m_or_f == 'fecg':
                sqi             = sqi['sqi_f1']
            sqi             = sqi[:, 1][0]
            np.savetxt(os.path.join(new_path, name[:-4] + '.txt'), sqi)

               


def fetch_save_txt(original_path = Path('D:\\datasets\\Biosignals\\raw_downloads\\University_of_Washington_ECG_Martin'),
                    new_path = Path('D:\\datasets\\Biosignals\\final_ECG_Martin'),
                    old_sampling_freq = 1000,
                    new_sampling_freq = 256,
                    type_m_or_f = 'mecg'):
    
    """ transfer mat files to txt file for better accessiblity; downsampled at 256 Hz """
    if type_m_or_f == 'mecg':
        new_path = new_path/ 'mECG_256'
        utils.makedirs(new_path)
        original_path        = os.path.join(original_path, 'mECG')        
        files, _             = utils.import_filenames(original_path)

    elif type_m_or_f == 'fecg':
        new_path = new_path/ 'fECG_256'
        utils.makedirs(new_path)
        original_path        = os.path.join(original_path, 'fECG')
        files, _             = utils.import_filenames(original_path)

    for name in tqdm(files):
        if name.endswith('.mat'):
            filename        = os.path.join(original_path, name)
            data            = loadmat(filename)
            if type_m_or_f == 'mecg':
                data            = data['mECG']
            elif type_m_or_f == 'fecg':
                data            = data['fECG']    
            ch1             = data[1, :]
            
            ch1             = utils.downsample(ch1, old_sampling_freq, new_sampling_freq).reshape(-1, )
            ch1             = preprocessing.filter_ecg(ch1, new_sampling_freq)
            
            np.savetxt(os.path.join(new_path, name[10:-4] + '.txt'), ch1)

    





def martins_score(path= 'D:\\datasets\\Biosignals\\final_ECG_Martin\\score.xlsx'):
        
    labels = pd.read_excel(path)
    labels.reset_index(drop = True)
    labels['id'] = labels['PATIENT CODE'].map(lambda x: int(x[3:]))
    labels = labels.rename(columns={"GROUP": "stress"})
    labels = labels.rename(columns={"SCORE PSS": "pss"})
    labels = labels.rename(columns={"SCORE PDQ": "pdq"})
    labels = labels.rename(columns={"FSI (bPRSA)": "fsi"})
    labels = labels.rename(columns={"CORTISOL (pg/mg of hair)": "cortisol"})

    martins_labels_mecg = labels.drop(['AChE/BChE_mat', 'AChE/BChE_fet', 'total activity_fet', 'AChE_fet', 'BChE_fet', 'total activity_mat', 'PATIENT CODE'], axis=1)
    martins_labels_mecg = martins_labels_mecg.dropna()
    martins_labels_mecg = martins_labels_mecg[['id', 'stress', 'pss', 'pdq', 'fsi', 'AChE_mat', 'BChE_mat', 'cortisol']]

    martins_labels_fecg = labels.drop(['AChE/BChE_mat', 'AChE/BChE_fet', 'total activity_fet', 'AChE_mat', 'BChE_mat', 'total activity_mat', 'PATIENT CODE'], axis=1)
    martins_labels_fecg = martins_labels_fecg.dropna()
    martins_labels_fecg = martins_labels_fecg[['id', 'stress', 'pss', 'pdq', 'fsi', 'AChE_fet', 'BChE_fet', 'cortisol']]


    martins_labels_mecg.to_csv(path[:-5] + '_mecg.csv')
    martins_labels_fecg.to_csv(path[:-5] + '_fecg.csv')

    return martins_labels_mecg, martins_labels_fecg


def extract_martins_dataset(path = Path('D:\\datasets\\Biosignals\\final_ECG_Martin'),
                         fs = 256,
                         window_size_sec=10,
                         overlap_pct = 10,
                         data_path = os.path.join(Path(dirname).parent, 'data'),
                         type_m_or_f='mecg'):
    
    def _normalize(x):
        """ 
        perform z-score normalization of a signal """
        temp = np.sort(x)
        x_std = np.std(temp[np.int(0.025*temp.shape[0]) : np.int(0.975*temp.shape[0])])
        x_mean = np.mean(temp)
        x_scaled = (x-x_mean)/x_std
        return x_scaled
        
    
    if type_m_or_f == 'mecg':
        martins_label = pd.read_csv(os.path.join(path, 'score_mecg.csv')).to_numpy()
        martins_label = martins_label[:, 1:]
        sqi_path = os.path.join(path, 'msqi')  
        sqi_files, _             = utils.import_filenames(sqi_path)
        path = path/ 'mECG_256'    
        filename, _             = utils.import_filenames(path)
        
    elif type_m_or_f == 'fecg':
        martins_label = pd.read_csv(os.path.join(path, 'score_fecg.csv')).to_numpy()
        martins_label = martins_label[:, 1:]
        sqi_path = os.path.join(path, 'fsqi')        
        sqi_files, _             = utils.import_filenames(sqi_path)
        path = path/ 'fECG_256'
        filename, _             = utils.import_filenames(path)

    
    window_size = fs * window_size_sec
    dataset   = np.zeros((1, window_size+8), dtype = int)
    for k in tqdm(range(len(filename))):
        file        = os.path.join(path, filename[k])
        sqi_file         = os.path.join(sqi_path, sqi_files[k])

        data            = np.loadtxt(file)
        sqi             = np.loadtxt(sqi_file)
        data            = _normalize(data)
        # data            = skp.minmax_scale(data, (-1,1), axis=1)

        segmented_data = utils.make_window(data, fs, overlap_pct, window_size_sec)
        segmented_sqi  = utils.make_window(sqi, 1, overlap_pct, window_size_sec)
        segmented_sqi  = np.mean(segmented_sqi, axis=1)
        segmented_sqi = segmented_sqi[:len(segmented_data)]
        segmented_data = segmented_data[np.where(segmented_sqi>=0.5)[0]]
        identity = int(filename[k][5:-4])
        if identity in martins_label[:,0]:

            stress = martins_label[np.where(martins_label[:,0]==identity), 1][0][0]
            pss = martins_label[np.where(martins_label[:,0]==identity), 2][0][0]
            pdq = martins_label[np.where(martins_label[:,0]==identity), 3][0][0]
            fsi = martins_label[np.where(martins_label[:,0]==identity), 4][0][0]
            AChE = martins_label[np.where(martins_label[:,0]==identity), 5][0][0]
            BChE = martins_label[np.where(martins_label[:,0]==identity), 6][0][0]
            cortisol = martins_label[np.where(martins_label[:,0]==identity), 7][0][0]
            
            ids = np.ones((segmented_data.shape[0], 1))*  identity
            stress = np.ones((segmented_data.shape[0], 1))* stress
            pss = np.ones((segmented_data.shape[0], 1))* pss
            pdq = np.ones((segmented_data.shape[0], 1))* pdq
            fsi = np.ones((segmented_data.shape[0], 1))* fsi
            AChE = np.ones((segmented_data.shape[0], 1))* AChE
            BChE = np.ones((segmented_data.shape[0], 1))* BChE
            cortisol = np.ones((segmented_data.shape[0], 1))* cortisol

            segmented_data = np.hstack((ids, stress, pss, pdq, fsi, AChE, BChE, cortisol, segmented_data)) 
        
            if np.all(dataset==0):
                dataset = segmented_data
            else:
                dataset = np.vstack((dataset, segmented_data))

    np.save((data_path + str('\\martins_'+ type_m_or_f + '_' + str(overlap_pct)+'.npy')), dataset)
            
    return dataset


# fetch_save_txt()
# martins_dataset = extract_martins_dataset()

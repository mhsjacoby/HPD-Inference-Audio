"""
audio_confidence.py
Authors: Sin Yong Tan and Maggie Jacoby
Edited: 2020-10-06 

Input: Processed audio in .npz files (organized by hour/by day)
Output: 'complete (full day: 8640) occupancy decision for each day, may have nans

To run: python3 audio_confidence.py -path /Volumes/TOSHIBA-18/H6-black/ 
    optional parameters: 	-hub (eg 'BS2'). if not specified will do for all hubs
                            -save_location.  if not specifed same as read
                            -start_index (number, eg 1). corresponds to how many files to skip
"""

import os
import sys
import time
import pickle
import numpy as np
import pandas as pd

from glob import glob
from datetime import datetime
from joblib import load
from platform import python_version
from struct import calcsize

from tensorflow.keras.models import model_from_json
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings("ignore")

from gen_argparse import *
from my_functions import *

# from load_data import load_data



# class Audio_Pred(object):
#     def __init__(self, clf_path):
#         self.clf = load(clf_path)

#     def occ_pred(self, audio):
#         return self.clf.predict(audio)


def create_timeframe(start_date, end_date=None, freq="10S"):
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d') if end_date else start_date + pd.Timedelta(days=1)
    clf = Audio_Pred(f'/Users/maggie/Documents/Github/HPD-Inference_and_Processing/Audio/Inference-Audio/trained_RF(3.7.6-64).joblib')
    timeframe = pd.date_range(start_date, end_date, freq=freq).strftime('%Y-%m-%d %H:%M:%S')[:-1]
    timeframe = pd.to_datetime(timeframe)
    
    return timeframe



def load_data(npz):
    hour = os.path.basename(npz).split('_')[1]
    full_hr = np.load(npz)   
    times = full_hr.files
    time_keys, audio_data = [], []
    for time in times:
        if len(full_hr[time]) > 0:
            time_keys.append(time)
            audio_data.append(full_hr[time])

    if len(audio_data) > 0:
        input_data = np.stack(audio_data)
        input_data = input_data.transpose(0,2,1)

        print('returning data')
        print(np.shape(input_data))
        return input_data, time_keys
    else:
        print('returning nothing')
        return [], []


def main(date_folder_path):
    date = os.path.basename(date_folder_path)

    # clf = Audio_Pred(f'/Users/maggie/Documents/Github/HPD-Inference_and_Processing/Audio/Inference-Audio/trained_RF(3.7.6-64).joblib')


    predictions, day_times = [], []
    hour_npzs = glob(os.path.join(date_folder_path, '*_ds.npz'))

    

    for npz in hour_npzs:

        input_data, times = load_data(npz)
        hour = os.path.basename(npz).split('_')[1]
        print('hour:', hour)

        num_filters = input_data.shape[1]

        # Flatten for scaling and reshape to 3D
        ori_input_shape = input_data.shape
        input_data = input_data.reshape((len(input_data), -1))
        input_data = scaler.transform(input_data)
        input_data = input_data.reshape((len(input_data), ori_input_shape[1], ori_input_shape[2], 1))

        class_prob = model.predict(input_data)
        predictions = class_prob[:,1]
        # predictions = [probability[1] for probability in class_prob]
        print(len(predictions), len(times))
        
        for time, pred in zip(predictions, times):
            print(time, pred)

        sys.exit()


        # 
        # audio_data = np.load(npz)
        # times = audio_data.files

        # preds = []

        # for time in times:
        #     DCTs = audio_data[time]
        #     if len(DCTs) != 0:
        #         pred = np.max(clf.occ_pred(DCTs)) # WavFile-wise OR-gate
        #         preds.append(pred)
        #     else:
        #         preds.append(np.nan)
        # predictions += preds
        # day_times += times

    timestamp = [f'{date} {time}' for time in day_times]

    data = pd.DataFrame(data=predictions, index=timestamp, columns=['occupied'])
    
    data.index = pd.to_datetime(data.index) 				# turn into datatime index
    data = data[~data.index.duplicated(keep='first')] 		# remove duplicate values

    timeframe = create_timeframe(date)						# create timestamp index for full day

    data = data.reindex(timeframe, fill_value=np.nan) 		# use new index
    data.index = data.index + pd.Timedelta(seconds=10)		# shift indexes up in time by 10 seconds
    data.index.name = 'timestamp'

    data.to_csv(os.path.join(save_root_path,f'{date}.csv'))



if __name__ == '__main__':  
    print(f'List of Hubs: {hubs}')

    model_path = os.getcwd()
    model = model_from_json(open(os.path.join(model_path, 'Audio_CNN/model-94_96/CNN_model.json')).read())
    model.load_weights(os.path.join(model_path, 'Audio_CNN/model-94_96/CNN_weights.h5'))
    model.summary()
    scaler = pickle.load(open(os.path.join(model_path, 'Audio_CNN/model-94_96/scaler.pkl'), 'rb'))

    for hub in hubs:
        start = time.time()
        print(f'Reading processed audio data from hub: {hub}')
        read_root_path = os.path.join(path, hub, 'processed_audio', 'audio_downsampled','20*')

        dates = sorted(glob(read_root_path))
        dates = [x for x in dates if os.path.basename(x) >= start_date]
        save_root_path = make_storage_directory(os.path.join(save_root,'Inference_DB', hub, 'audio_inf'))

        for date_folder_path in dates:
            print(f"Loading date folder: {os.path.basename(date_folder_path)} ...")
            main(date_folder_path)

        end = time.time()
        total_time = (end-start)/3600
        print(f'Total time taken to process hub {hub} in home {H_num}: {total_time:.02} hours')
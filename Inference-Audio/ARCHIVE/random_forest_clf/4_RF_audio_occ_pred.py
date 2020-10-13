"""
4_RF_audio_occ_pred.py
Authors: Sin Yong Tan and Maggie Jacoby
Edited: 2020-09-01 shift prediction by 10 seconds fill missing with nan and remove duplicates

Input: Processed audio in .npz files (organized by hour/by day)
Output: 'complete (full day: 8640) occupancy decision for each day, may have nans

To run: python3 4_RF_audio_occ_pred.py -path /Volumes/TOSHIBA-18/H6-black/ 
	optional parameters: 	-hub (eg 'BS2'). if not specified will do for all hubs
							-save_location.  if not specifed same as read
							-start_index (number, eg 1). corresponds to how many files to skip
"""

import numpy as np
import pandas as pd
from joblib import load
from platform import python_version
from struct import calcsize
import argparse
import glob
import time
import os
import sys
from datetime import datetime

import warnings
warnings.filterwarnings("ignore")

from gen_argparse import *
from my_functions import *


class Audio_Pred(object):
	def __init__(self, clf_path):
		self.clf = load(clf_path)

	def occ_pred(self, audio):
		return self.clf.predict(audio)


def create_timeframe(start_date, end_date=None, freq="10S"):
    
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    if end_date == None:
        end_date = start_date + pd.Timedelta(days=1)
    else:
        end_date = datetime.strptime(end_date, '%Y-%m-%d')

	# clf = Audio_Pred("trained_RF(%s-%s).joblib"%(python_version(),calcsize("P")*8)) # Call the classifier matching python version and bit
	# clf = Audio_Pred(f'/Users/maggie/Documents/Github/HPD-Inference_and_Processing/Inference-Audio/trained_RF({python_version}-{calcsize("P")*8}).joblib')
	clf = Audio_Pred(f'/Users/maggie/Documents/Github/HPD-Inference_and_Processing/Audio/Inference-Audio/trained_RF(3.7.6-64).joblib')
    timeframe = pd.date_range(start_date, end_date, freq=freq).strftime('%Y-%m-%d %H:%M:%S')[:-1]
    timeframe = pd.to_datetime(timeframe)
    
    return timeframe


def main(date_folder_path):
	date = os.path.basename(date_folder_path)

	clf = Audio_Pred(f'/Users/maggie/Documents/Github/HPD-Inference_and_Processing/Audio/Inference-Audio/trained_RF(3.7.6-64).joblib')

	predictions, day_times = [], []
	hour_npzs = glob.glob(os.path.join(date_folder_path, '*_ps.npz'))

	for npz in hour_npzs:
		hour = os.path.basename(npz).split('_')[1]
		audio_data = np.load(npz)
		times = audio_data.files

		preds = []

		for time in times:
			DCTs = audio_data[time]
			if len(DCTs) != 0:
				pred = np.max(clf.occ_pred(DCTs)) # WavFile-wise OR-gate
				preds.append(pred)
			else:
				preds.append(np.nan)
		predictions += preds
		day_times += times

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
	parser = argparse.ArgumentParser()
	parser.add_argument('-path','--path', default='AA', type=str, help='path of stored data')
	parser.add_argument('-hub', '--hub', default='', type=str, help='if only one hub... ')
	parser.add_argument('-save_location', '--save', default='', type=str, help='location to store files (if different from path')
	parser.add_argument('-start_index','--start_date_index', default=0, type=int, help='Processing START Date index')

	args = parser.parse_args()

	path = args.path
	save_path = args.save if len(args.save) > 0 else path
	home_system = os.path.basename(path.strip('/'))
	H = home_system.split('-')
	H_num, color = H[0], H[1][0].upper()
	hubs = [args.hub] if len(args.hub) > 0 else sorted(mylistdir(path, bit=f'{color}S', end=False))
	print(f'List of Hubs: {hubs}')

	# start_date_index = args.start_date_index

	for hub in hubs:
		start = time.time()
		print(f'Reading processed audio data from hub: {hub}')

		read_root_path = os.path.join(path, hub, 'processed_audio', 'audio_dct','20*')
		dates = sorted(glob.glob(f'{read_root_path}'))[start_date_index:]
		save_root_path = make_storage_directory(os.path.join(save_path,'Inference_DB', hub, 'audio_inf'))
		print("save_root_path: ", save_root_path)

		for date_folder_path in dates:
			print(f"Loading date folder: {os.path.basename(date_folder_path)} ...")
			main(date_folder_path)

		end = time.time()
		total_time = (end-start)/3600
		print(f'Total time taken to process hub {hub} in home {H_num}: {total_time:.02} hours')
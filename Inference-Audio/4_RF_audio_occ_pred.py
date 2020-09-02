"""
4_RF_audio_occ_pred.py
Authors: Sin Yong Tan and Maggie Jacoby

Takes processed audio in .npz files (organized by hour/by day) and outputs an occupancy decision

"""



import numpy as np
from joblib import load
from platform import python_version
from struct import calcsize
import argparse
import glob
import time


import os

import warnings
warnings.filterwarnings("ignore")

'''
Need to shift the prediction csv time +1 during final OR-Gate
'''

class Audio_Pred(object):
	def __init__(self, clf_path):
		self.clf = load(clf_path)

	def occ_pred(self, audio):
		return self.clf.predict(audio)


def mylistdir(directory, bit='', end=True):
    filelist = os.listdir(directory)
    if end:
        return [x for x in filelist if x.endswith(f'{bit}') and not x.endswith('.DS_Store') and not x.startswith('Icon')]
    else:
         return [x for x in filelist if x.startswith(f'{bit}') and not x.endswith('.DS_Store') and not x.startswith('Icon')]
        
def make_storage_directory(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    return target_dir


def main():

	# clf = Audio_Pred("trained_RF(%s-%s).joblib"%(python_version(),calcsize("P")*8)) # Call the classifier matching python version and bit
	# clf = Audio_Pred(f'/Users/maggie/Documents/Github/HPD-Inference_and_Processing/Inference-Audio/trained_RF({python_version}-{calcsize("P")*8}).joblib')
	clf = Audio_Pred(f'/Users/maggie/Documents/Github/HPD-Inference_and_Processing/Audio/Inference-Audio/trained_RF(3.7.6-64).joblib')



	# save_folder = "Inference_DB/H1-green/GS1/audio"

	# npz looping

	predictions = []
	day_times = []

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
	print(f'{len(predictions)} files.')

	save_data = np.vstack((timestamp,predictions)).T # (N rows x 2 columns)

	# # np.savetxt(os.path.join(save_folder,f"{date}.csv"), save_data, delimiter=',',fmt='%s',header="timestamp,occupied",comments='')
	
	np.savetxt(os.path.join(save_root_path,f'{date}.csv'), save_data, delimiter=',',fmt='%s',header="timestamp,occupied",comments='')



if __name__ == '__main__':  
	parser = argparse.ArgumentParser()
	parser.add_argument('-path','--path', default='AA', type=str, help='path of stored data')
	parser.add_argument('-hub', '--hub', default='', type=str, help='if only one hub... ')
	parser.add_argument('-save_location', '--save', default='', type=str, help='location to store files (if different from path')

	parser.add_argument('-start_index','--start_date_index', default=0, type=int, help='Processing START Date index')

	args = parser.parse_args()
	path = args.path
	save_path = args.save if len(args.save) > 0 else path
	home_system = path.strip('/').split('/')[-1]
	H = home_system.split('-')
	H_num, color = H[0], H[1][0].upper()
	hubs = [args.hub] if len(args.hub) > 0 else sorted(mylistdir(path, bit=f'{color}S', end=False))
	print(f'List of Hubs: {hubs}')

	start_date_index = args.start_date_index

	

	for hub in hubs:
		start = time.time()
		print(f'Reading processed audio data from hub: {hub}')

		read_root_path = os.path.join(path,hub,'processed_audio','audio_dct','*')
		dates = sorted(glob.glob(read_root_path))[start_date_index:]
		print('Dates: ', [os.path.basename(d) for d in dates])

		save_root_path = make_storage_directory(os.path.join(save_path,'Inference_DB', hub, 'audio_inf'))
		print("save_root_path: ", save_root_path)

		for date_folder_path in dates:
			date = os.path.basename(date_folder_path)
			if not date.startswith('20'):
				print(f'passing folder: {date}')
				continue

			print(f"Loading date folder: {date} ...")

			main()

		end = time.time()
		total_time = (end-start)/3600
		print(f'Total time taken to process hub {hub} in home {H_num}: {total_time:.02} hours')







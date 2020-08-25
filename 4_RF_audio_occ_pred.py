import numpy as np
from joblib import load
from platform import python_version
from struct import calcsize

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



def main():

	clf = Audio_Pred("trained_RF(%s-%s).joblib"%(python_version(),calcsize("P")*8)) # Call the classifier matching python version and bit

	# save_folder = "Inference_DB/H1-green/GS1/audio"

	# npz looping

	# Loading data
	data_path = "2019-10-18_0800_BS2_H6_ps.npz"
	X_test = np.load(data_path)
	date = data_path.split("_")[0] # 2019-10-18

	times = X_test.files # 360 counts

	preds = []

	for time in times:
		DCTs = X_test[time]
		if len(DCTs) != 0: # not empty
			pred = np.max(clf.occ_pred(DCTs)) # WavFile-wise OR-gate
			preds.append(pred)
		else:
			preds.append(np.nan)



	timestamp = [date +" "+ time for time in times] # "2019-10-18" + " " + "00:00:00"
	save_data = np.vstack((timestamp,preds)).T # (N rows x 2 columns)

	# np.savetxt(os.path.join(save_folder,f"{date}.csv"), save_data, delimiter=',',fmt='%s',header="timestamp,occupied",comments='')
	np.savetxt(f"{date}.csv", save_data, delimiter=',',fmt='%s',header="timestamp,occupied",comments='')



if __name__ == '__main__':  
	main()
import numpy as np
import pandas as pd
import random
import os

path = os.getcwd()

def load_data(dataset, hubs=None):
	num_filters = None

	# ==== minimal labeled dataset -- amplitude ====
	if dataset == 1:
		noisy_data = load_min_labeled_data("noisy", hubs)
		quiet_data = load_min_labeled_data("quiet", hubs)


	# ==== H1 BS1 old labeled data -- amplitude ====
	elif dataset == 2:
		noisy_data = load_H1BS1_labeled_data("noisy")
		quiet_data = load_H1BS1_labeled_data("quiet")


	# ==== H1 BS1 old labeled 20 sec audio data -- MATLAB processed ====
	elif dataset == 3:
		noisy_data = load_matlab_processed_data("noisy") # (??, ??, 17)
		quiet_data = load_matlab_processed_data("quiet")
		# num_filters = quiet_data.shape[2]


	# ==== H1 BS1 old labeled 10 sec data -- Python processed ====
	elif dataset == 4:
		noisy_data = load_python_processed_H1BS1("noisy") # (187, 16, 1000)
		quiet_data = load_python_processed_H1BS1("quiet")

		# Trim data
		quiet_data = quiet_data[:len(noisy_data)] # In this scenarios there are more quiet data than noisy 
		# num_filters = quiet_data.shape[2]


	# ==== minimal labeled dataset -- Python processed ====
	elif dataset == 5:
		noisy_data = load_min_labeled_processed_data("noisy", hubs)
		quiet_data = load_min_labeled_processed_data("quiet", hubs)

	elif dataset == 6:
		return load_H1red(hubs)


	if (dataset == 3) or (dataset == 4) or (dataset == 5):
		num_filters = quiet_data.shape[1]


	print("==== Data dimension ====")
	print(f"Noisy data: {np.shape(noisy_data)}")
	print(f"Quiet data: {np.shape(quiet_data)}\n")

	return quiet_data, noisy_data, num_filters





def load_min_labeled_data(data_type, hubs):
	all_data = []
	for hub in hubs:
		data = np.load("minimal_labeled_sets/"+hub+"-"+data_type+".npy")
		all_data.extend(data)
	return np.array(all_data)

def load_H1BS1_labeled_data(data_type):
	'''
	Load 20 sec audio data
	separate them into 10 seconds (assuming the noisy is noisy across 20 seconds...)
	'''
	data_20 = np.load("H1BS1/"+data_type+".npy")
	all_data = []
	for nd20 in data_20:
		all_data.append(nd20[:80000])
		all_data.append(nd20[80000:])
	return np.array(all_data)

def load_matlab_processed_data(data_type):
	# Each wav file is 3137 rows, this csv contain 99 wav file data
	if data_type == "noisy":
		data_20 = pd.read_csv("H1BS1/RaspPi_all_Occupied.csv",header=None)
	elif data_type == "quiet":
		data_20 = pd.read_csv("H1BS1/RaspPi_all_Unoccupied.csv",header=None)
	start_index_20 = np.arange(0,len(data_20),3137)
	data_10 = []
	for i in start_index_20:
		data_10.append(data_20[i:i+1568].values)
		data_10.append(data_20[i+1568:i+2*1568].values)
	return np.array(data_10)

def load_python_processed_H1BS1(data_type):
	if data_type == "noisy":
		data_10 = np.load("H1BS1/Occupied.npy")
	elif data_type == "quiet":
		data_10 = np.load("H1BS1/Unoccupied.npy")
	data_10 = data_10.transpose(0,2,1) # Transpose (1000 x 16) to (16 x 1000)
	return data_10


def load_min_labeled_processed_data(data_type, hubs):
	all_data = []
	for hub in hubs:
		data = np.load("minimal_labeled_sets/processed/"+hub+"-"+data_type+".npy")
		all_data.extend(data)
	all_data = np.array(all_data)
	all_data = all_data.transpose(0,2,1) # Transpose (1000 x 16) to (16 x 1000)
	return all_data


def load_H1red(hubs, test_ratio=0.3):
	all_noisy_train = []
	all_noisy_test  = []
	all_quiet_train = []
	all_quiet_test  = []


	for hub in hubs:
		# hub = hubs[0]
		noisy_data = np.load(os.path.join(path,"H1-red/processed/"+hub+"-noise.npy"))
		quiet_data = np.load(os.path.join(path,"H1-red/processed/"+hub+"-quiet.npy"))

		# Transpose (1000 x 16) to (16 x 1000)
		noisy_data = noisy_data.transpose(0,2,1)
		quiet_data = quiet_data.transpose(0,2,1)

		# Split to train and test -- This ensures that train and test will have same proportion of data from each hub
		noisy_train, noisy_test = train_test_split(noisy_data, test_ratio)
		quiet_train, quiet_test = train_test_split(quiet_data, test_ratio)

		# Collect all
		all_noisy_train.extend(noisy_train)
		all_noisy_test.extend(noisy_test)
		all_quiet_train.extend(quiet_train)
		all_quiet_test.extend(quiet_test)


	# Generate labels
	y_noisy_train = np.ones((len(all_noisy_train),))
	y_noisy_test  = np.ones((len(all_noisy_test),))
	y_quiet_train = np.zeros((len(all_quiet_train),))
	y_quiet_test  = np.zeros((len(all_quiet_test),))


	# Stack all train and test
	x_train = np.vstack((all_noisy_train,all_quiet_train))
	x_test  = np.vstack((all_noisy_test,all_quiet_test))
	y_train = np.hstack((y_noisy_train,y_quiet_train))
	y_test  = np.hstack((y_noisy_test,y_quiet_test))


	# Shuffle the x and y content IN THE SAME WAY
	x_train, y_train = shuffle_data(x_train, y_train)
	x_test, y_test   = shuffle_data(x_test, y_test)

	print("==== Data dimension ====")
	print(f"x_train: {np.shape(x_train)}")
	print(f"y_train: {np.shape(y_train)}")
	print(f"x_test: {np.shape(x_test)}")
	print(f"y_test: {np.shape(y_test)}\n")

	return x_train, y_train, x_test, y_test





def train_test_split(data, test_ratio):
	test_index = random.sample(range(0, len(data)), int(test_ratio*len(data)))
	test_data  = data[test_index]
	train_data = np.delete(data, test_index, axis=0)
	return train_data, test_data


def shuffle_data(x,y):
	shuffle_idx = np.arange(x.shape[0])
	np.random.shuffle(shuffle_idx)
	shuffled_x = x[shuffle_idx]
	shuffled_y = y[shuffle_idx]
	return shuffled_x, shuffled_y








# a = [[[1,2,3],[4,5,6]],[[7,8,9],[10,11,12]]]
# a = np.array(a)
# ori_shape = a.shape
# print(ori_shape)
# print(a)

# a = a.reshape((len(a),-1))
# new_shape = a.shape
# print(new_shape)
# print(a)

# a = a.reshape(ori_shape)
# reshaped = a.shape
# print(reshaped)
# print(a)




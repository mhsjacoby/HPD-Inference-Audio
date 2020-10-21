"""
test_audio_classifier.py

Authors: Maggie Jacoby and Sin Yong Tan
Edited: 2020-10-20

"""

import sys
import os
import warnings
import argparse
from glob import glob
from copy import deepcopy
warnings.filterwarnings("ignore")

from tensorflow.keras.models import model_from_json
from tensorflow.keras.utils import to_categorical
# from load_data import load_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

import pickle



def DESTROY_transient_effect(data):
    data[:,:10,:] = data[:,10:20,:] # data shape = (N,1000,16)
    return data


def load_data(data_path, scaling='filter'):

    npy_loc = os.path.join(data_path, 'train_test')

    x_train = np.load(npy_loc+"/X_train.npy")
    y_train = np.load(npy_loc+"/Y_train.npy")
    x_test  = np.load(npy_loc+"/X_test.npy")
    y_test  = np.load(npy_loc+"/Y_test.npy")

    # Shuffle the x and y content IN THE SAME WAY
    x_train, y_train = shuffle_data(x_train, y_train)
    x_test, y_test   = shuffle_data(x_test, y_test)

    print("==== Data dimension ====")
    print(f'x_train: {np.shape(x_train)}, y_train: {np.shape(y_train)}')
    print(f'x_test: {np.shape(x_test)}, y_test: {np.shape(y_test)}\n')

    # ==== Scaling Input ====
    if scaling == "filter":

        ori_train_shape, ori_test_shape = x_train.shape, x_test.shape

        x_train = norm_by_filter(x_train) # (2500, 16, 1000)
        x_test  = norm_by_filter(x_test)

        x_train = x_train.reshape((len(x_train), ori_train_shape[1], ori_train_shape[2], 1))
        x_test = x_test.reshape((len(x_test), ori_test_shape[1], ori_test_shape[2], 1))


    # One-hot Encode Output
    nb_classes = len(np.unique(y_test))
    y_train = to_categorical(y_train, nb_classes)
    y_test = to_categorical(y_test, nb_classes)

    return x_train, y_train, x_test, y_test

def norm_by_filter(data):
    # Assume data dimension: (N，16，1000)
    N, num_filter, filter_data_len = data.shape
    assert filter_data_len > num_filter, "Assume data dimension: (N_samples, num_filter, filter_data_len)"
    scaled_data = deepcopy(data)
    for N in range(len(data)):
        for filter_ in range(len(data[N])): # (0, 0, 1000)
            scaled_data[N][filter_] = (scaled_data[N][filter_] - np.min(scaled_data[N][filter_]))/(np.max(scaled_data[N][filter_]) - np.min(scaled_data[N][filter_]))
    return scaled_data


def shuffle_data(x,y):
    shuffle_idx = np.arange(x.shape[0])
    np.random.shuffle(shuffle_idx)
    shuffled_x  = x[shuffle_idx]
    shuffled_y  = y[shuffle_idx]
    return shuffled_x, shuffled_y


def eval_model(model, X_test, Y_test, verbose=0):

    Y_pred = (model.predict(X_test)>0.5).astype("int32")
    y_pred = np.argmax(Y_pred, axis=1)
    y_test = np.argmax(Y_test, axis=1)
    print('Accuracy Score:')
    print(accuracy_score(y_test, y_pred))

    if verbose >= 1:
        print('Confusion Matrix:')
        print(confusion_matrix(y_test, y_pred))
    if verbose >= 2:
        print('Classification Report:')
        target_names = ['Quiet', 'Noisy']
        print(classification_report(y_test, y_pred, target_names=target_names))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Description")

    parser.add_argument('-system', '--system', type=str)
    parser.add_argument('-self', '--self', type=bool, default=True)
    args = parser.parse_args()

    home_system = args.system
    H_num, color = home_system.split('-')
    self_test = args.self

    current_path = os.getcwd()

    model_path = os.path.join(current_path, 'Audio_CNN', 'model-94_96', f'CNN_model.json')
    model = model_from_json(open(model_path).read())
    weight_path = os.path.join(current_path, 'Audio_CNN', 'model-94_96', f'CNN_weights_{home_system}.h5')
    model.load_weights(weight_path)
    model.summary()


    data_path = os.path.join(current_path, 'Audio_CNN', 'CNN_testing_code', home_system)
    x_train, y_train, x_test, y_test = load_data(data_path)

    if self_test:
        eval_model(model, x_test, y_test, verbose=3)

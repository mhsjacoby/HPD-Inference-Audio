'''
Load and test trained models
'''

import warnings
warnings.filterwarnings("ignore")
import sys
import os
from tensorflow.keras.models import model_from_json
from load_data import load_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from tensorflow.keras.utils import to_categorical


path = os.getcwd()
model = model_from_json(open(os.path.join(path, '94_96/94_96/CNN_model.json')).read())
model.load_weights(os.path.join(path,'94_96/94_96/CNN_weights.h5'))
model.summary()



hubs = ["RS1","RS2","RS3","RS4","RS5"]
X_train, y_train, X_test, y_test = load_data(dataset=6, hubs=hubs)
num_filters = X_test.shape[1]

# Flatten for scaling later
ori_X_shape = X_train.shape
X_train = X_train.reshape((len(X_train),-1))
X_test = X_test.reshape((len(X_test),-1))

# Scaling input
scaler_x_cross = MinMaxScaler(feature_range=(0, 1))
np.save(os.path.join(path, '94_96/94_96/scaler_x_cross.json'))
X_train_cross = scaler_x_cross.fit_transform(X_train)
X_test_cross = scaler_x_cross.transform(X_test)

# X_data = np.vstack((X_train_cross,X_test_cross ))
# scaler_x_cross = scaler_x_cross.fit(X_data)

# Reshape back to 3D
X_train_cross = X_train_cross.reshape((len(X_train_cross),ori_X_shape[1],ori_X_shape[2],1)) # len(X_train_cross)== number of samples
X_test_cross = X_test_cross.reshape((len(X_test_cross),ori_X_shape[1],ori_X_shape[2],1))

# One-hot Encode Output
nb_classes = len(np.unique(y_train))
Y_train_cross = to_categorical(y_train, nb_classes)
Y_test_cross = to_categorical(y_test, nb_classes)


# Performance on train data
Y_train_prob = model.predict(X_train_cross) # returns probability of each class (index 0: quiet, 1: noisy)
Y_train_pred = (model.predict(X_train_cross)>0.5).astype("int32")
y_train_pred = np.argmax(Y_train_pred, axis=1)
Y_train_cross = np.argmax(Y_train_cross, axis=1)
print('Accuracy Score:')
print(accuracy_score(Y_train_cross, y_train_pred))

# Performance on test data
Y_test_prob = model.predict(X_test_cross) # returns probability of each class (index 0: quiet, 1: noisy)
Y_test_pred = (model.predict(X_test_cross)>0.5).astype("int32")
y_test_pred = np.argmax(Y_test_pred, axis=1)
Y_test_cross = np.argmax(Y_test_cross, axis=1)
print('Accuracy Score:')
print(accuracy_score(Y_test_cross, y_test_pred))


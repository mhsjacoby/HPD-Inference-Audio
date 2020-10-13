'''
Load and test trained models
'''
# import os
# os.chdir(r"C:\Users\Sin Yong Tan\Desktop\to_maggie\Audio_CNN")
import warnings
warnings.filterwarnings("ignore")

from tensorflow.keras.models import model_from_json
from load_data import load_data
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np

import pickle


model = model_from_json(open('94_96/CNN_model.json').read())
model.load_weights('94_96/CNN_weights.h5')
model.summary()


hubs = ["RS1","RS2","RS3","RS4","RS5"]
X_train, y_train, X_test, y_test = load_data(dataset=6, hubs=hubs)
input_data = np.vstack((X_train,X_test))
ground_truth = np.hstack((y_train,y_test))
del X_train, y_train, X_test, y_test
num_filters = input_data.shape[1]

# Flatten for scaling later
ori_input_shape = input_data.shape
input_data = input_data.reshape((len(input_data),-1))

# load the scaler
scaler = pickle.load(open('94_96/scaler.pkl', 'rb'))
# Scaling input
input_data = scaler.transform(input_data)


# Reshape back to 3D
input_data = input_data.reshape((len(input_data),ori_input_shape[1],ori_input_shape[2],1))


# Performance on all data
class_prob = model.predict(input_data) # returns probability of each class (index 0: quiet, 1: noisy)
class_pred = (class_prob>0.5).astype("int32") # one-hot encoded predictions
# Y_train_pred = (model.predict(X_train_cross)>0.5).astype("int32") # one-hot encoded predictions
class_pred = np.argmax(class_pred, axis=1) # binary predictions (undo the one-hot encoding)

print('Accuracy Score:')
print(accuracy_score(ground_truth, class_pred))

print('Confusion Matrix:')
print(confusion_matrix(ground_truth, class_pred))

print('Classification Report:')
target_names = ['Quiet', 'Noisy']
print(classification_report(ground_truth, class_pred, target_names=target_names))

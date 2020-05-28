import numpy as np
import time

import tensorflow.keras.backend as K
from tensorflow.keras import metrics
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.utils.np_utils import to_categorical

from data_prep.data_splitter import *

split_functions = [splitDataSet1, splitDataSet2, splitDataSet3]
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
model = None
divisions = 2


def create_spectrograms(dataset):
	spectrograms = np.empty((int(len(dataset) * divisions), 20, int(1290 / divisions)))
	labels = []
	for i in range(len(dataset)):
		label = genres.index(os.path.basename(dataset[i]).split('.')[-4])
		data_points = np.hsplit(np.load(dataset[i]), divisions)
		for j in range(divisions):
			spectrograms[i] = data_points[j]
			labels.append(label)
	return spectrograms, np.array(labels)


def load_dataset(split_index, normalized=True):
	folder = 'spectr-data-normalized' if normalized else 'spectr-data'
	split = split_functions[split_index](folder, 5)
	return [create_spectrograms(split[i]) for i in range(3)]


def test_network(testing_set, testing_labels, network_path):
	global model
	print('----TESTING----')
	train_time = time.time_ns()
	history = model.evaluate(testing_set, testing_labels)
	print((time.time_ns() - train_time) / 1000000)
	print(history[0])
	print(history[1])


def train_network(training_set, training_labels, save_path='network.keras'):
	global model
	print('----TRAINING----')
	model = Sequential([
		Flatten(input_shape=training_set[0].shape),
		Dense(256, activation='relu'),
		Dense(64, activation='relu'),
		Dense(10, activation='softmax'),
	])
	model_metrics = ['accuracy']
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	print(model.summary())

	train_time = time.time_ns()
	history = model.fit(training_set, training_labels, epochs=2, batch_size=32, validation_split=0)
	print((time.time_ns() - train_time) / 1000000)
	print(str(history.history['loss'])[1:-1].replace(',', ''))
	print(str(history.history['accuracy'])[1:-1].replace(',', ''))

	model.save(save_path)

def f1_score(y_true, y_pred):
	precision = precision_m(y_true, y_pred)
	recall = recall_m(y_true, y_pred)
	return 2*((precision*recall) / (precision + recall + K.epsilon()))


def fbeta_score(y_true, y_pred, threshold_shift=0):
	beta = 2
	y_pred = K.clip(y_pred, 0, 1)
	y_pred_bin = K.round(y_pred + threshold_shift)

	tp = K.sum(K.round(y_true * y_pred_bin)) + K.epsilon()
	fp = K.sum(K.round(K.clip(y_pred_bin - y_true, 0, 1)))
	fn = K.sum(K.round(K.clip(y_true - y_pred, 0, 1)))

	precision = tp / (tp + fp)
	recall = tp / (tp + fn)

	beta_squared = beta ** 2
	return (beta_squared + 1) * (precision * recall) / (beta_squared * precision + recall)


def recall_m(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
	recall = true_positives / (possible_positives + K.epsilon())
	return recall


def precision_m(y_true, y_pred):
	true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
	predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
	precision = true_positives / (predicted_positives + K.epsilon())
	return precision

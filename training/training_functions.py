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
divisions = 1


def create_spectrograms(dataset, fma_set):
	spectrograms = np.empty((int(len(dataset) * divisions), 20, int(1290 / divisions)))
	labels = []
	for i in range(len(dataset)):
		genre_name = os.path.basename(dataset[i]).split('.')[-4]
		label = int(genre_name) if fma_set else genres.index(genre_name)
		data_points = np.hsplit(np.load(dataset[i]), divisions)
		for j in range(divisions):
			spectrograms[i] = data_points[j]
			labels.append(label)
	classes_count = 8 if fma_set else len(genres)
	return spectrograms, to_categorical(np.array(labels), num_classes=classes_count)


def load_dataset(split_index, fma_set=True, normalized=True):
	if fma_set:
		normalized = False
	folder = 'data-normalized' if normalized else 'data'
	if fma_set:
		folder = 'fma-' + folder
	split = split_functions[split_index]('spectr-' + folder, 5)
	return [create_spectrograms(split[i], fma_set) for i in range(3)]


def test_network(testing_set, testing_labels, network_path):
	global model
	print('----TESTING----')
	train_time = time.time_ns()
	history = model.evaluate(testing_set, testing_labels)
	print((time.time_ns() - train_time) / 1000000)
	print(history[0])
	print(history[1])


def train_network(training_set, training_labels, fma_set=True, save_path='network.keras'):
	global model
	print('----TRAINING----')
	classes_count = 8 if fma_set else len(genres)
	model = Sequential([
		Flatten(input_shape=training_set[0].shape),
		Dense(256, activation='relu'),
		Dense(64, activation='relu'),
		Dense(classes_count, activation='softmax'),
	])
	model_metrics = [metrics.CategoricalAccuracy(), metrics.Recall(), metrics.AUC(),
	                 metrics.SensitivityAtSpecificity(.8), metrics.SpecificityAtSensitivity(.8), f1_score, fbeta_score]
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=model_metrics)
	print(model.summary())

	train_time = time.time_ns()
	history = model.fit(training_set, training_labels, epochs=5, batch_size=32, validation_split=0)
	print((time.time_ns() - train_time) / 1000000)
	print(str(history.history['loss'])[1:-1].replace(',', ''))
	print(str(history.history['categorical_accuracy'])[1:-1].replace(',', ''))

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

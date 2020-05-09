import numpy as np
import time
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten
from tensorflow.python.keras.models import load_model

from data_prep.data_splitter import *


split_functions = [splitDataSet1, splitDataSet2, splitDataSet3]
genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']


def create_spectrograms(dataset):
	spectrograms = np.empty((len(dataset), 20, 1290))
	labels = []
	for i in range(len(dataset)):
		spectrograms[i] = np.load(dataset[i])
		labels.append(genres.index(os.path.basename(dataset[i]).split('.')[-4]))
	return spectrograms, np.array(labels)


def load_dataset(split_index, normalized=True):
	folder = 'spectr-data-normalized' if normalized else 'spectr-data'
	split = split_functions[split_index](folder, 5)
	return [create_spectrograms(split[i]) for i in range(3)]


def test_network(testing_set, testing_labels, network_path):
	print('----TESTING----')
	model = load_model(network_path)
	train_time = time.time_ns()
	history = model.evaluate(testing_set, testing_labels)
	print((time.time_ns() - train_time) / 1000000)
	print(history[0])
	print(history[1])


def train_network(training_set, training_labels, save_path='network.keras'):
	print('----TRAINING----')
	model = Sequential([
		Flatten(input_shape=training_set[0].shape),
		Dense(256, activation='relu'),
		Dense(64, activation='relu'),
		Dense(10, activation='softmax'),
	])
	model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
	print(model.summary())

	train_time = time.time_ns()
	history = model.fit(training_set, training_labels, epochs=15, batch_size=64, validation_split=0)
	print((time.time_ns() - train_time) / 1000000)
	print(str(history.history['loss'])[1:-1].replace(',', ''))
	print(str(history.history['accuracy'])[1:-1].replace(',', ''))
	model.save(save_path)

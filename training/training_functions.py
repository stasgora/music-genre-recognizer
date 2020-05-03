import librosa
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
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
	return spectrograms, labels


def load_dataset(split_index, normalized=True):
	folder = 'spectr-data-normalized' if normalized else 'spectr-data'
	split = split_functions[split_index](folder, 5)
	return [create_spectrograms(split[i]) for i in range(3)]


def test_network(testing_set, testing_labels, network_path):
	model = load_model(network_path)
	model.evaluate(testing_set, testing_labels)


def train_network(training_set, training_labels, save_path='network.keras'):
	model = Sequential([
		Dense(256, input_dim=training_set[0].shape[0], activation='relu'),
		Dense(64, activation='relu'),
		Dense(10, activation='softmax'),
	])
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	print(model.summary())

	model.fit(training_set, training_labels, epochs=10, batch_size=64, validation_split=0)
	model.save(save_path)

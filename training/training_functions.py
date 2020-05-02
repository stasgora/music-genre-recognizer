import librosa
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import load_model
from data_prep.data_splitter import *


split_functions = [splitDataSet1, splitDataSet2, splitDataSet3]


def create_spectrograms(dataset):
	spectrograms = []
	labels = []
	size = (0, 0)
	for file in dataset:
		y, sr = librosa.load(file)
		mfcc = librosa.feature.mfcc(y=y, sr=sr)
		if size == (0, 0) or mfcc.shape < size:
			size = mfcc.shape
		spectrograms.append(mfcc)
		labels.append(os.path.basename(file).split('.')[-3])
	t = [[x[0:size[0]][0:size[1]] for x in spectrograms], labels]
	return t


def load_dataset(split_index, normalized=True):
	folder = 'data-normalized' if normalized else 'data'
	split = split_functions[split_index](folder, 5)
	return [create_spectrograms(split[i]) for i in range(3)]


def test_network(testing_set, testing_labels, network_path):
	model = load_model(network_path)
	model.evaluate(testing_set, testing_labels)


def train_network(training_set, training_labels, save_path='network.keras'):
	model = Sequential([
		Dense(256, input_shape=(training_set), activation='relu'),
		Dense(64, activation='relu'),
		Dense(10, activation='softmax'),
	])
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	print(model.summary())

	model.fit(training_set, training_labels, epochs=10, batch_size=64, validation_split=0)
	model.save(save_path)

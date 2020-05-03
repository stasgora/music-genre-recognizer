import os

import librosa
import numpy as np

normalized = True
folder_suffix = '-normalized' if normalized else ''
data_dir = 'data' + folder_suffix
spectr_data_dir = 'spectr-data' + folder_suffix

os.mkdir(spectr_data_dir)
with os.scandir(data_dir) as data:
	for entry in data:
		if entry.is_dir():
			genre_dir = os.path.join(data_dir, entry.name)
			spectr_genre_dir = os.path.join(spectr_data_dir, entry.name)
			os.mkdir(spectr_genre_dir)
			with os.scandir(genre_dir) as genre:
				for song in genre:
					y, sr = librosa.load(os.path.join(genre_dir, song.name))
					mfcc = librosa.feature.mfcc(y=y, sr=sr)[:20, :1290]
					mfcc /= np.amax(np.absolute(mfcc))
					np.save(os.path.join(spectr_genre_dir, song.name), mfcc)

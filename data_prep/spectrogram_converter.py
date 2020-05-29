import os
import threading

import librosa
import numpy as np

normalized = False
folder_suffix = '-normalized' if normalized else ''
data_dir = 'fma-data' + folder_suffix
spectr_data_dir = 'spectr-fma-data' + folder_suffix


def process(genre_id):
	output_lock = threading.Lock()
	genre_dir = os.path.join('fma-data', str(genre_id))
	spectr_genre_dir = os.path.join('spectr-fma-data', str(genre_id))
	with os.scandir(genre_dir) as genre:
		for song in genre:
			new_path = os.path.join(spectr_genre_dir, song.name)
			if os.path.isdir(new_path):
				continue
			try:
				y, sr = librosa.load(os.path.join(genre_dir, song.name))
			except Exception:
				with output_lock:
					print(str(song.name) + ' oooo nie!!!!!')
				continue
			mfcc = librosa.feature.mfcc(y=y, sr=sr)[:20, :1290]
			mfcc /= np.amax(np.absolute(mfcc))
			np.save(new_path, mfcc)


threads = []
for i in range(8):
	threads.append(threading.Thread(name='genre_'+str(i), target=process, args=(i,)))
	threads[-1].start()
for i in range(8):
	threads[i].join()


def run_linear():
	os.mkdir(spectr_data_dir)
	with os.scandir(data_dir) as data:
		for entry in data:
			if entry.is_dir():
				print(entry.name)
				genre_dir = os.path.join(data_dir, entry.name)
				spectr_genre_dir = os.path.join(spectr_data_dir, entry.name)
				os.mkdir(spectr_genre_dir)
				with os.scandir(genre_dir) as genre:
					for song in genre:
						new_path = os.path.join(spectr_genre_dir, song.name)
						if os.path.isdir(new_path):
							continue
						try:
							y, sr = librosa.load(os.path.join(genre_dir, song.name))
						except Exception:
							print(str(song.name) + ' oooo nie!!!!!')
							continue
						mfcc = librosa.feature.mfcc(y=y, sr=sr)[:20, :1290]
						mfcc /= np.amax(np.absolute(mfcc))
						np.save(os.path.join(new_path, song.name), mfcc)

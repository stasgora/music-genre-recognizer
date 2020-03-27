import hashlib
import os

data_dir = '../data'
repetition_count = 0


def hash(fname):
	return hashlib.sha1(open(fname, 'rb').read()).hexdigest()


with os.scandir(data_dir) as data:
	for entry in data:
		if entry.is_dir():
			genre_dir = os.path.join(data_dir, entry.name)
			with os.scandir(genre_dir) as genre:
				songs = list(genre)
				for i in range(len(songs)):
					for j in range(i + 1, len(songs)):
						song_1 = songs[i]
						song_2 = songs[j]
						if hash(os.path.join(genre_dir, song_1.name)) == hash(os.path.join(genre_dir, song_2.name)):
							print(song_1.name[:-4] + '-' + song_2.name[:-4])
							repetition_count += 1
	print("TOTAL:", repetition_count)

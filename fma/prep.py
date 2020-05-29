import json
import os
import pickle
import shutil


def sort_files():
	folder = 'fma-data'
	files = '../fma_small'
	with open('fma/title_map', 'rb') as file:
		title_map = pickle.load(file)
	with open('fma/genre_tree_map', 'rb') as file:
		genre_tree_map = pickle.load(file)
	with open('fma/genre_map', 'rb') as file:
		genre_map = pickle.load(file)
	out_genres = []
	with os.scandir(files) as data:
		index = 0
		for entry in data:
			if entry.is_dir():
				index += 1
				print(index)
				with os.scandir(entry) as genre_dir:
					for song in genre_dir:
						song_id = int(song.name.split('.')[0])
						if song_id == 133297:
							continue
						if song_id not in title_map or title_map[song_id] not in genre_tree_map or genre_tree_map[title_map[song_id]] not in genre_map:
							print(str(song_id) + ' oooo nie!!!!!')
							continue
						genre_name = genre_map[genre_tree_map[title_map[song_id]]]
						if genre_name not in out_genres:
							out_genres.append(genre_name)
						genre_id = out_genres.index(genre_name)
						song_path = os.path.join(folder, str(genre_id))
						if not os.path.isdir(song_path):
							os.mkdir(song_path)
						shutil.copyfile(song.path, os.path.join(song_path, str(genre_id) + '.' + str(song_id) + '.mp3'))
	print(out_genres)


def create_genre_tree():
	file = '../fma_metadata/genres.csv'
	with open(file) as input:
		next(input)
		genre_map = {}
		for line in input:
			fields = line.strip().split(',')
			genre_map[int(fields[0])] = int(fields[4])
		with open('fma/genre_tree_map', 'wb') as file:
			pickle.dump(genre_map, file)
		print(genre_map)
		print(len(genre_map))


sort_files()


def create_maps():
	file = '../fma_metadata/raw_tracks-sep.csv'
	with open(file) as input:
		next(input)
		title_map = {}
		genre_map = {}
		index = 0
		for line in input:
			index += 1
			if index % 100 == 0:
				print(index)
			fields = line.strip().split(';')
			if fields[27] == '':
				continue
			try:
				genre = json.loads(fields[27].replace("'", '"'))[0]
			except Exception:
				continue
			title_map[int(fields[0])] = int(genre['genre_id'])
			genre_map[int(genre['genre_id'])] = genre['genre_title']
		with open('fma/title_map', 'wb') as file:
			pickle.dump(title_map, file)
		with open('fma/genre_map', 'wb') as file:
			pickle.dump(genre_map, file)
		print(genre_map)
		print(len(genre_map))

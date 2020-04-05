import os

from pydub import AudioSegment

addedDirString = "Normalized"
slash = "\\"


def normalize(sound, target_dBFS):
	changeIndBFS = target_dBFS - sound.dBFS
	return sound.apply_gain(changeIndBFS)


def prepareNormalization(mainDir, target):  # main directory containing genres, targetdBFS - best: -20.0
	genres = os.listdir(mainDir)

	newMainDir = mainDir + addedDirString
	if not os.path.isdir(newMainDir):
		try:
			os.mkdir(newMainDir)
		except OSError:
			print("Dir creation error in ", newMainDir)
	for genreDir in genres:
		genreSpecifiedNormDir = newMainDir + slash + genreDir + addedDirString
		if not os.path.isdir(genreSpecifiedNormDir):
			try:
				os.mkdir(genreSpecifiedNormDir)
			except OSError:
				print("Dir creation error in ", genreSpecifiedNormDir)
		genreFiles = os.listdir(mainDir + slash + genreDir)
		for file in genreFiles:
			sound = AudioSegment.from_file(mainDir + slash + genreDir + slash + file, "wav")
			normalizedFile = normalize(sound, target)
			try:
				normalizedFile.export(newMainDir + slash + genreDir + addedDirString + slash + file, format="wav")
			except FileExistsError:
				print("Can't save file:", file)
		print("Genre normalization completed: ", genreDir)
	print("Completed normalization.")

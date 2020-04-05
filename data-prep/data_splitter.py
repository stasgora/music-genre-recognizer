import os
import random
import numpy

genresCount = 10
slash = "\\"


def splitDataSet1(mainDir, k):  # main directory containing genres, number of sets
	j = 0
	fileSplit = [[] for i in range(k)]
	for genreDir in os.listdir(mainDir):
		files = os.listdir(mainDir + slash + genreDir)
		i = 0
		while not len(files) == 0:
			chosen = random.choice(files)
			files.remove(chosen)
			fileSplit[i].append(chosen)
			i += 1
			if i == k:
				i = 0
		j += 1
	finalSet = [[] for i in range(3)]
	finalSet[2] = random.choice(fileSplit)
	fileSplit.remove(finalSet[2])
	finalSet[1] = random.choice(fileSplit)
	fileSplit.remove(finalSet[1])
	finalSet[0] = numpy.concatenate(fileSplit).tolist()
	print("Data has been splitted into set 1")
	return finalSet


def splitDataSet2(mainNormDir, k, lowestNumberOfFiles = 86):
	j = 0
	fileSplit = [[] for i in range(k)]
	for genreDir in os.listdir(mainNormDir):
		files = os.listdir(mainNormDir + slash + genreDir)
		i = 0
		while not len(files) == 0:
			chosen = random.choice(files)
			files.remove(chosen)
			fileSplit[i].append(chosen)
			i += 1
			if i == k:
				i = 0
			j += 1
			if j == lowestNumberOfFiles:
				fileSplit[k-1].extend(files)
				break
	finalSet = [[] for i in range(3)]
	finalSet[2] = fileSplit[k-1]
	fileSplit.remove(finalSet[2])
	finalSet[1] = random.choice(fileSplit)
	fileSplit.remove(finalSet[1])
	finalSet[0] = numpy.concatenate(fileSplit).tolist()
	print("Data has been splitted into set 2")
	return finalSet


def splitDataSet3(mainNormDir, k, lowestNumberOfFiles=86):
	j = 0
	fileSplit = [[] for i in range(k)]
	for genreDir in os.listdir(mainNormDir):
		files = os.listdir(mainNormDir + slash + genreDir)
		i = 0
		while not len(files) == 0:
			chosen = random.choice(files)
			files.remove(chosen)
			fileSplit[i].append(chosen)
			i += 1
			if i == k:
				i = 0
			j += 1
			if j == lowestNumberOfFiles:
				fileSplit[k - 1].extend(files)
				break
	finalSet = [[] for i in range(3)]
	finalSet[2] = fileSplit[k - 1]
	fileSplit.remove(finalSet[2])
	finalSet[1] = random.choice(fileSplit)
	finalSet[0] = numpy.concatenate(fileSplit).tolist()
	print("Data has been splitted into set 3")
	return finalSet

import librosa.display
import matplotlib.pyplot as plt
import numpy


def view_spectrogram(file):
	y, sr = librosa.load(file)
	plt.figure(figsize=(12, 8))
	D = librosa.amplitude_to_db(numpy.abs(librosa.stft(y)), ref=numpy.max)
	librosa.display.specshow(D, y_axis='linear')
	plt.colorbar(format='%+2.0f dB')
	plt.title('Linear-frequency power spectrogram')
	plt.show()

def view_rms(file):
	y, sr = librosa.load(file)
	plt.figure(figsize=(12, 8))
	S, phase = librosa.magphase(librosa.stft(y))
	rms = librosa.feature.rms(S=S)
	plt.semilogy(rms.T, label='RMS Energy')
	plt.xticks([])
	plt.xlim([0, rms.shape[-1]])
	plt.legend(loc='best')
	plt.show()
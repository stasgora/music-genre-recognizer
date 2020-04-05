import librosa
import numpy as np

def getSpectrogram(filename):
    y, sr = librosa.load(filename)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    return S_dB
    
print(getSpectrogram(librosa.util.example_audio_file()))

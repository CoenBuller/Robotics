import librosa
import librosa.feature
import numpy as np

def extract_features(audio, rate=44100):
    mfccs = librosa.feature.mfcc(y=audio, sr=rate, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)
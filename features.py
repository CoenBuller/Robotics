import numpy as np
from scipy.fft import rfft, rfftfreq
import librosa
import librosa.feature

RATE = 44100


def extract_features(audio):
    audio = audio.flatten()

    energy = np.sum(audio ** 2)
    zcr = np.sum(np.diff(np.sign(audio)) != 0) / len(audio)
    attack = np.argmax(np.abs(audio)) / len(audio)

    fft_vals = np.abs(rfft(audio))
    frequencies = rfftfreq(len(audio), 1 / RATE)

    peak_freq = frequencies[np.argmax(fft_vals)]
    centroid = np.sum(frequencies * fft_vals) / (np.sum(fft_vals) + 1e-9)
    spread = np.sqrt(np.sum(((frequencies - centroid) ** 2) * fft_vals) / (np.sum(fft_vals) + 1e-9))

    mfccs = librosa.feature.mfcc(y=audio, sr=RATE, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1)

    return np.concatenate([[energy, zcr, attack, peak_freq, centroid, spread], mfcc_means])
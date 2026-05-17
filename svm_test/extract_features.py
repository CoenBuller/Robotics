import librosa
import numpy as np

def extract_features(y, sr=15872):
    features = []

    # MFCCs — capture timbral texture
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=32)
    features.extend(np.mean(mfcc, axis=1))
    features.extend(np.std(mfcc, axis=1))

    # Zero Crossing Rate — high for claps, low for whistle/harmonic
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # Spectral centroid — where the "center of mass" of the spectrum is
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(centroid))

    # Spectral bandwidth — narrow for whistle, wide for clap
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(bandwidth))

    # Spectral rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    features.append(np.mean(rolloff))

    # RMS energy — low for background
    rms = librosa.feature.rms(y=y)
    features.append(np.mean(rms))
    features.append(np.max(rms))

    # Spectral flux — very high for claps (sudden change)
    stft = np.abs(librosa.stft(y))
    flux = np.mean(np.diff(stft, axis=1) ** 2)
    features.append(flux)

    return np.array(features)  # ~42 features total
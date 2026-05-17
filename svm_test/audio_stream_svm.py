import sounddevice as sd
import numpy as np
import joblib
import librosa
from collections import deque
from extract_features import extract_features

model = joblib.load("sound_classifier.pkl")
LABELS = ["clap", "whistle", "harmonic", "background"]

SR = 15_872
WINDOW_SIZE = SR // 2       # 500ms
HOP_SIZE = SR // 4          # 250ms (50% overlap)
CONFIDENCE_THRESHOLD = 0.8
RMS_THRESHOLD = 0.02

# Smooth predictions over last N windows to reduce flicker
recent_preds = deque(maxlen=3)

buffer = np.zeros(WINDOW_SIZE)

def callback(indata, frames, time, status):
    global buffer
    new_audio = indata[:, 0]
    buffer = np.roll(buffer, -len(new_audio))
    buffer[-len(new_audio):] = new_audio

    if np.sqrt(np.mean(buffer**2)) < RMS_THRESHOLD:
        print("background")
        return

    features = extract_features(buffer, SR).reshape(1, -1)
    proba = model.predict_proba(features)[0]

    if proba.max() > CONFIDENCE_THRESHOLD:
        pred = LABELS[np.argmax(proba)]
        recent_preds.append(pred)
        # Majority vote over recent windows
        majority = max(set(recent_preds), key=list(recent_preds).count)
        print(f"{majority:12s}  conf={proba.max():.2f}")
    else:
        print(f"{'uncertain':12s}  conf={proba.max():.2f}")

with sd.InputStream(samplerate=SR, channels=1, blocksize=HOP_SIZE, callback=callback):
    print("Listening... Ctrl+C to stop")
    while True:
        sd.sleep(100)
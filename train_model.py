import os
import librosa
import pickle
from sklearn.ensemble import RandomForestClassifier
from features import extract_features

DATA_PATH = "data/"
classes = sorted(os.listdir(DATA_PATH))
X, y = [], []

for idx, label in enumerate(classes):
    class_path = os.path.join(DATA_PATH, label)
    for file in os.listdir(class_path):
        if file.endswith('.wav'):
            audio, sr = librosa.load(os.path.join(class_path, file), sr=44100)
            feat = extract_features(audio, sr)
            X.append(feat)
            y.append(idx)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

with open("sound_model.pkl", "wb") as f:
    pickle.dump((model, classes), f)
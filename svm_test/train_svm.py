from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from extract_features import extract_features
from pathlib import Path
import joblib
import numpy as np
import librosa as lb
import os


# Build dataset
X, y = [], []
for label, folder in enumerate(["data/clap", "data/harmonica", "data/whistle"]):
    for file in Path(folder).glob("*.wav"):
        audio, sr = lb.load(file, sr=15_872, duration=1)
        audio = lb.util.fix_length(audio, size=int(sr))
        X.append(extract_features(audio, int(sr)))
        y.append(label)


X, y = np.array(X), np.array(y)

# Pipeline: scale → classify
model = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", SVC(kernel="rbf", C=10, gamma="scale", probability=True, class_weight="balanced")
)
])

# Evaluate with stratified k-fold (important with only 100 samples/class)
scores = cross_val_score(model, X, y, cv=StratifiedKFold(n_splits=5), scoring="f1_macro")
print(f"F1: {scores.mean():.3f} ± {scores.std():.3f}")

model.fit(X, y)
joblib.dump(model, "sound_classifier.pkl")
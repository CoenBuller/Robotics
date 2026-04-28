import numpy as np
import os
import pickle
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from audioProcessor import AudioProcessor

ap = AudioProcessor(samplerate=16_000)

SAMPLES = 20
LABELS = {
    'clap': 0,
    'whistle': 1,
    'harmonica': 2,
    'silence': 3
}

X = []
y = []

for label_name, label_num in LABELS.items():
    folder = f'data/{label_name}'
    if not os.path.exists(folder):
        continue

    for i in range(SAMPLES):
        file_path = f'data/{label_name}/{label_name}_{i}.wav'
        if os.path.exists(file_path):
            audio, _ = librosa.load(file_path, sr=16_000)
            fft, mfcc, peak, amp = ap.CalcMFCC(audio)
            X.append(mfcc)
            y.append(label_num)

X = np.array(X)
y = np.array(y)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# Switching back to Random Forest
# n_estimators=300 makes the model more robust than the default 100
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    class_weight='balanced',
    random_state=42
)

model.fit(X_train, y_train)

predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)

print(f"Random Forest Accuracy: {score * 100:.1f}%")
print(classification_report(y_test, predictions, target_names=list(LABELS.keys())))

with open('sound_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
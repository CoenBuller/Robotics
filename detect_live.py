import numpy as np
import pickle
from collections import deque
from features import extract_features
import sound_util

with open("sound_model.pkl", "rb") as f:
    model, classes = pickle.load(f)

prediction_buffer = deque(maxlen=5)
THRESHOLD = 0.85


def audio_callback(indata, _frames, _time, _status):
    audio_data = indata.flatten()
    feat = extract_features(audio_data, 44100).reshape(1, -1)

    probs = model.predict_proba(feat)[0]
    max_prob = np.max(probs)
    prediction_idx = np.argmax(probs)

    if max_prob > THRESHOLD:
        prediction_buffer.append(classes[prediction_idx])
    else:
        prediction_buffer.append("noise_or_silence")

    final_command = max(set(prediction_buffer), key=prediction_buffer.count)
    execute_command(final_command)


def execute_command(cmd):
    if cmd == "whistle":
        print("FORWARDS")
    elif cmd == "harmonica":
        print("HARMONICA_DETECTED")
    elif cmd == "clap":
        print("STOP")
    else:
        print("IDLE")


with sound_util.get_stream(audio_callback):
    while True:
        pass
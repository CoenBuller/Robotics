import sounddevice as sd
import numpy as np
import pickle
from features import extract_features

RATE = 44100
DURATION = 1.0
THRESHOLD = 0.02

with open('sound_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

NAMES = {0: 'CLAP', 1: 'WHISTLE', 2: 'HARMONICA', 3: 'SILENCE'}
ROBOT_STATE = 'stopped'


def safe_command(cmd_id):
    global ROBOT_STATE
    if cmd_id == 0:
        ROBOT_STATE = 'stopped' if ROBOT_STATE == 'moving' else 'moving'
        print(f">>> {ROBOT_STATE.upper()}")
    elif cmd_id == 1:
        ROBOT_STATE = 'moving'
        print(">>> FORWARD")
    elif cmd_id == 2:
        ROBOT_STATE = 'turning'
        print(">>> SPIN")

last_detected = None

while True:
    audio = sd.rec(int(DURATION * RATE), samplerate=RATE, channels=1, dtype='float32')
    sd.wait()

    if np.max(np.abs(audio)) > THRESHOLD:
        feat = scaler.transform([extract_features(audio)])
        probs = model.predict_proba(feat)[0]
        max_prob = np.max(probs)
        current_result = model.classes_[np.argmax(probs)]

        if max_prob > 0.5 and current_result != 3:
            if current_result != last_detected:
                print(f"\n{NAMES[current_result]} ({max_prob * 100:.1f}%)")
                safe_command(current_result)
                last_detected = current_result
        else:
            last_detected = None
            print(f"Listening... {max_prob * 100:.1f}%", end='\r')
    else:
        last_detected = None
        print("Quiet...", end='\r')
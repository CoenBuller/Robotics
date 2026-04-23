import sounddevice as sd
import os
from scipy.io import wavfile

RATE = 44100
DURATION = 1.0

classes = ['clap', 'whistle', 'harmonica', 'silence']
for c in classes:
    os.makedirs(f'data/{c}', exist_ok=True)


def record_one(label, number):
    input(f"[{label.upper()} {number + 1}/20] Press Enter to record...")

    audio = sd.rec(int(DURATION * RATE), samplerate=RATE, channels=1, dtype='float32')
    sd.wait()

    path = f'data/{label}/{label}_{number}.wav'
    wavfile.write(path, RATE, audio)
    print(f"Saved: {path}\n")


for c in classes:
    print(f"--- Starting {c.upper()} ---")
    for i in range(20):
        record_one(c, i)

print("All samples recorded successfully.")
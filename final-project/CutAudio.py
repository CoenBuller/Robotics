import librosa as lb
from scipy.io import wavfile
import os
from tqdm import tqdm

final_folder = os.path.join("data", "silence")
# os.makedirs(final_folder, exist_ok=True)

path = "BG_uni_1.wav"
audio = lb.load(path=path, sr=15_872)[0]



def cutAudio(audio, durations=1, sr=15_872):
    audio_time = len(audio)/sr
    n_frames = int(audio_time/durations)

    for i in tqdm(range(n_frames)):
        lower, upper = i * sr*durations, (i+1) * sr * durations
        y = audio[lower: upper]
        path = f"BG_uni_{i}.wav"
        path = os.path.join(final_folder, path)
        wavfile.write(path, sr, y)

cutAudio(audio)
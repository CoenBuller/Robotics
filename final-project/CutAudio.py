import librosa as lb
from scipy.io import wavfile
import os
from tqdm import tqdm

final_folder = os.path.join("final-project", "data_audio", "Background_Noise_Uni")
os.makedirs(final_folder, exist_ok=True)

path = os.path.join("final-project", "achtergrond_uni_final_project.wav")
audio = lb.load(path=path, sr=16_000)[0]



def cutAudio(audio, durations=1, sr=16_000):
    audio_time = len(audio)/sr
    n_frames = int(audio_time/durations)

    for i in tqdm(range(n_frames)):
        lower, upper = i * sr, (i+1) * sr
        y = audio[lower: upper]
        path = f"BG_uni_{i}.wav"
        path = os.path.join("final-project", "data_audio", "Background_Noise_Uni", path)
        wavfile.write(path, sr, y)

cutAudio(audio)
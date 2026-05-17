import librosa as lb
from scipy.io import wavfile
import os
from tqdm import tqdm

final_folder = os.path.join("final_car_noise")

folder = "car_sounds"

def cutAudio(audio, durations=1, sr=15_872, j=100):
    audio_time = len(audio)/sr
    n_frames = int(audio_time/durations)

    for i in tqdm(range(n_frames)):
        lower, upper = i * sr*durations, (i+1) * sr * durations
        y = audio[lower: upper]
        path = f"CN_{i}_{j}.wav"
        print(final_folder, path)
        path = os.path.join(final_folder, path)
        wavfile.write(path, sr, y)


for k, file in enumerate(os.listdir(folder)):
    audio = lb.load(path=os.path.join(folder, file), sr=15_872)[0]
    cutAudio(audio, j=k)

# import librosa
# import soundfile as sf

# def remove_last_four_seconds(input_path, output_path):
#     # 1. Load the audio file
#     # sr=None preserves the original sampling rate
#     y, sr = librosa.load(input_path, sr=None)
    
#     # 2. Calculate number of samples to remove
#     # (seconds * samples per second)
#     samples_to_remove = 3 * sr
    
#     # 3. Slice the array
#     # If the file is shorter than 4 seconds, this returns an empty array
#     trimmed_y = y[:-samples_to_remove]
    
#     # 4. Save the result
#     sf.write(output_path, trimmed_y, sr)
#     print(f"Trimmed audio saved to {output_path}")

# # Usage
# remove_last_four_seconds(folder, folder)




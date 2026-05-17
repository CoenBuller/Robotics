import librosa as lb
import numpy as np
import matplotlib.pyplot as plt

from AudioAugmentationPipelin import AudioAugmentationPipeline, AugmentConfig
from FinalFolder.PitchExtraction import FastMelSpec
from scipy.io.wavfile import write


# audio_file = "data/harmonica/harmonica_43.wav"
# audio_file = "data/clap/yt_2_2.wav"
audio_file = "data/whistle/whistle_42.wav"
# audio_file = "data/silence/UB_BG_100.wav"
audio, sr = lb.load(path=audio_file, sr=15872, duration=1)

ap = AudioAugmentationPipeline(hop=512, n_fft=512, n_mels=62)
ap2 = FastMelSpec()

mel = ap.process(audio=audio, noise=False, pitch=False, volume=False, spec_aug=False, time_shift=False)
mel2, _, _ = ap2(audio / (np.max(np.abs(audio)) + 1e-9))

print(mel.shape, mel2.shape)

fig, ax = plt.subplots(1, 3)
ax[0].imshow(mel)
ax[1].imshow(mel2)
ax[2].imshow(np.abs(mel-mel2))

plt.show()
# augmented_audio = np.int16(mfcc * 32767)

# write(filename="test.wav", rate=15872, data=augmented_audio)


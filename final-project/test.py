import librosa as lb
import numpy as np
import matplotlib.pyplot as plt

from AudioProcessPipeline import AudioAugmentationPipeline, AugmentConfig
from scipy.io.wavfile import write


# audio_file = "data/harmonica/harmonica_67.wav"
# audio_file = "data/clap/clap_1.wav"
audio_file = "data/whistle/whistle_2.wav"
# audio_file = "data/silence/BG_uni_1000.wav"
audio, sr = lb.load(path=audio_file, sr=15872, duration=1)


cfg = AugmentConfig(
                    noise_prob=0.80,      noise_snr_range=(5, 20),
                    pitch_shift_prob=0.70, pitch_shift_range=(-4, 4),
                    volume_scale_prob=0.70, volume_gain_range=(0.5, 1.5),
                    spec_augment_prob=0.80,
                    n_freq_masks=2, freq_mask_param=6,
                    n_time_masks=2, time_mask_param=6,
                    )
ap = AudioAugmentationPipeline(hop=512, n_mels=13)
augmented_audio = ap.process_audio(audio=audio)
# plt.imshow(mfcc[1:])
# plt.show()
augmented_audio = np.int16(augmented_audio * 32767)

write(filename="test.wav", rate=15872, data=augmented_audio)


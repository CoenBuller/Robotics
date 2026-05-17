import numpy as np
import librosa as lb
import torchaudio.functional as F

sr    = 15_872
n_fft = 512
n_mels = 62

mel_fb = F.melscale_fbanks(n_freqs=n_fft // 2 + 1, f_min=0, f_max=sr/2, n_mels=n_mels, sample_rate=sr).numpy().astype(np.float32)

np.save("mel_fb.npy", mel_fb.T)
print(mel_fb.shape)  # (62, 257)
from __future__ import annotations
import numpy as np


class FastMelSpec:
    """
    Pure-numpy log-mel spectrogram extractor matching torchaudio's:

        T.MelSpectrogram(sample_rate=sr, n_fft=n_fft, hop_length=hop, n_mels=n_mels)
        T.AmplitudeToDB(top_db=90)

    Requires mel_fb.npy — generate once with:
        import torchaudio, numpy as np
        fb = torchaudio.functional.melscale_fbanks(
                n_freqs=512//2+1, f_min=0.0, f_max=15_872/2,
                n_mels=62, sample_rate=15_872, norm=None, mel_scale='htk')
        np.save("mel_fb.npy", fb.T.numpy().astype(np.float32))
    """

    def __init__(
        self,
        mel_fb_path: str  = "final-project/FinalFolder/mel_fb.npy",
        sr: int           = 15_872,
        n_fft: int        = 512,
        hop: int          = 512,
        top_db: float     = 90.0,
    ):
        self.sr     = sr
        self.n_fft  = n_fft
        self.hop    = hop
        self.top_db = top_db

        # Hann window — identical to torch.hann_window used by torchaudio
        self.window = np.hanning(n_fft).astype(np.float32)

        # Mel filterbank (n_mels, n_fft//2+1)
        self.mel_fb = np.load(mel_fb_path).astype(np.float32)

        # Frequency axis for pitch FFT
        self.freqs = np.fft.rfftfreq(n_fft, 1.0 / sr).astype(np.float32)

    def _power_stft(self, x: np.ndarray) -> np.ndarray:
        """Returns power spectrogram |STFT|^2, shape (n_fft//2+1, n_frames)."""
        x = np.pad(x, self.n_fft // 2, mode='reflect')

        n_frames = 1 + (len(x) - self.n_fft) // self.hop
        frames   = np.lib.stride_tricks.as_strided(
            x,
            shape=(n_frames, self.n_fft),
            strides=(x.strides[0] * self.hop, x.strides[0]),
            writeable=False,
        )

        spec  = np.fft.rfft(frames * self.window, axis=-1)
        power = spec.real ** 2 + spec.imag ** 2

        return power.T.astype(np.float32)                    # (n_fft//2+1, n_frames)

    def __call__(self, audio: np.ndarray):
        """
        Returns
        -------
        log_mel   : np.ndarray (n_mels, n_frames) float32
        pitch_hz  : float
        amplitude : float
        """
        x = np.ascontiguousarray(audio, dtype=np.float32).ravel()

        # Normalize to [-1, 1] — must match the augmentation pipeline exactly:
        #   audio /= (np.max(np.abs(audio)) + 1e-9)
        # Without this, int16 input (range ±32768) produces mel values on a
        # completely different scale than what the model was trained on.
        x = x / (np.max(np.abs(x)) + 1e-9)

        power   = self._power_stft(x)
        mel     = self.mel_fb @ power                        # (n_mels, n_frames)

        log_mel = 10.0 * np.log10(np.maximum(mel, 1e-10))
        log_mel = np.maximum(log_mel, log_mel.max() - self.top_db)

        # Pitch — single FFT on the first n_fft samples
        segment   = x[:self.n_fft] if len(x) >= self.n_fft else np.pad(x, (0, self.n_fft - len(x)))
        spec      = np.fft.rfft(segment * self.window)
        pitch_pwr = spec.real ** 2 + spec.imag ** 2

        peak_bin  = int(np.argmax(pitch_pwr))
        pitch_hz  = float(self.freqs[peak_bin])
        amplitude = float(np.sqrt(pitch_pwr[peak_bin]))

        return log_mel.astype(np.float32), pitch_hz, amplitude
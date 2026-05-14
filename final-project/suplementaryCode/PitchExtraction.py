from __future__ import annotations
import numpy as np
import librosa as lb

class FastMFCC:
    def __init__(
        self,
        sr: int = 15872,
        n_fft: int = 2048,
        hop: int = 512,
        n_mels: int = 128,           # librosa's actual default — your trained model uses this
        n_mfcc: int = 62,            # number of MFCC coefficients to keep (was n_mels in AudioProcessor)
        fmin: float = 0.0,
        fmax: float | None = None,
        center: bool = True,         # librosa default; needed for shape parity
        top_db: float = 80.0,        # librosa power_to_db default
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop = hop
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.center = center
        self.top_db = top_db

        # Hann window of length n_fft — applied per-frame before FFT.
        # NB: librosa uses 'hann' by default, which is identical to numpy's hanning.
        self.window = np.hanning(n_fft).astype(np.float32)

        # Mel filterbank — same call librosa.feature.mfcc makes internally.
        # Shape: (n_mels, n_fft//2 + 1)
        self.mel_fb = lb.filters.mel(
            sr=sr,
            n_fft=n_fft,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax if fmax is not None else sr / 2,
            htk=False,
            norm='slaney',
        ).astype(np.float32)

        nonempty = self.mel_fb.sum(axis=1) > 0
        self.mel_fb = self.mel_fb[nonempty]
        self.n_mels = int(nonempty.sum())
        self.dct = self._make_dct_ortho(self.n_mels, self.n_mels)[:self.n_mfcc].astype(np.float32)
        self.freqs = np.fft.rfftfreq(n_fft, 1.0 / sr).astype(np.float32)

    @staticmethod
    def _make_dct_ortho(n_out: int, n_in: int) -> np.ndarray:
        """Orthonormal DCT-II matrix. Equivalent to scipy.fftpack.dct(..., norm='ortho')."""
        n = np.arange(n_in)
        k = np.arange(n_out)[:, None]
        d = np.cos(np.pi * (n + 0.5) * k / n_in) * np.sqrt(2.0 / n_in)
        d[0] *= 1.0 / np.sqrt(2.0)
        return d

    def _frame(self, x: np.ndarray) -> np.ndarray:
        """
        Slice x into overlapping frames of length n_fft, hop=self.hop.
        Mimics librosa's center=True zero-padding (pad_mode='constant') so
        frame count matches lb.feature.mfcc exactly.
        Returns shape (n_frames, n_fft).
        """
        if self.center:
            pad = self.n_fft // 2
            x = np.pad(x, pad, mode='constant')  # librosa default is 'constant', not 'reflect'

        n = len(x)
        n_frames = 1 + (n - self.n_fft) // self.hop
        if n_frames < 1:
            # Pad to at least one frame
            x = np.pad(x, (0, self.n_fft - n))
            n_frames = 1

        # Zero-copy view: stride_tricks avoids materialising a (n_frames, n_fft) buffer
        # until we multiply by the window.
        strides = (x.strides[0] * self.hop, x.strides[0])
        frames = np.lib.stride_tricks.as_strided(
            x, shape=(n_frames, self.n_fft), strides=strides, writeable=False
        )
        return frames

    def __call__(self, audio: np.ndarray):
        """
        Returns:
            mfcc:      (n_mfcc, n_frames)  float32
            pitch_hz:  (n_frames,)         float32 — peak frequency per frame
            amplitude: (n_frames,)         float32 — peak magnitude per frame
        """
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            x = np.ascontiguousarray(audio, dtype=np.float32).ravel()

            # Frame the signal
            frames = self._frame(x)                          # (T, n_fft)
            windowed = frames * self.window                  # broadcast window across T

            # Batched real FFT — one call covers all frames
            spec = np.fft.rfft(windowed, axis=-1)            # (T, n_fft//2 + 1) complex
            power = spec.real * spec.real + spec.imag * spec.imag  # |X|^2, real

            # Pitch readout: argmax over frequency, per frame. Free byproduct.
            peak_bin = np.argmax(power, axis=-1)             # (T,)
            pitch_hz = self.freqs[peak_bin]                  # (T,)
            amplitude = np.sqrt(power[np.arange(len(peak_bin)), peak_bin])

            # Mel projection: (n_mels, F) @ (F, T) -> (n_mels, T)
            power = np.maximum(power, 1e-10).astype(np.float32)
            mel = self.mel_fb @ power.T                      # (n_mels, T)

            # Log power, matching librosa.power_to_db(ref=1.0, amin=1e-10, top_db=80.0):
            #   log_spec = 10 * log10(max(amin, mel))
            #   log_spec = max(log_spec, log_spec.max() - top_db)
            log_mel = 10.0 * np.log10(np.maximum(mel, 1e-10))
            if self.top_db is not None:
                log_mel = np.maximum(log_mel, log_mel.max() - self.top_db)

            # DCT-II ortho
            mfcc = self.dct @ log_mel                        # (n_mfcc, T)

        return mfcc, pitch_hz, amplitude
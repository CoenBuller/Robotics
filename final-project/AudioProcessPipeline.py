import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
 
import librosa as lb
import numpy as np
 
from audioProcessor import AudioProcessor

@dataclass
class AugmentConfig:

    # --- Noise ---
    noise_prob:       float = 0.5
    noise_snr_range:  Tuple[float, float] = (10.0, 30.0)   # dB  (higher = cleaner)
    # Which noise types to sample from. 
    noise_types:      List[str] = field(default_factory=lambda: ['white', 'pink'])
 
 
    # --- Pitch shift ---
    pitch_shift_prob:  float = 0.4
    pitch_shift_range: Tuple[float, float] = (-3.0, 3.0)   # semitones
 
    # --- Volume scaling ---
    volume_scale_prob:  float = 0.5
    volume_gain_range:  Tuple[float, float] = (0.5, 1.5)   # linear gain
 
    # --- SpecAugment ---
    spec_augment_prob:  float = 0.5
    n_freq_masks:       int   = 1
    freq_mask_param:    int   = 3    # max mel bands to zero out per mask
    n_time_masks:       int   = 1
    time_mask_param:    int   = 10   # max time frames to zero out per mask


class AudioAugmentationPipeline:
    """
    Wraps an AudioProcessor and applies stochastic augmentations before
    MFCC extraction.
 
    Waveform augmentations are applied in a fixed order (noise → time stretch
    → pitch shift → volume scale) but each stage fires independently based on
    its configured probability.  SpecAugment is applied after MFCC extraction.
    """
 
    def __init__(
        self,
        processor: AudioProcessor,
        config: Optional[AugmentConfig] = None,
        seed: Optional[int] = None,
    ):
        self.processor = processor
        self.config    = config or AugmentConfig()
 
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

    def _add_noise(
        self, audio: np.ndarray, snr_db: float, noise_type: str
    ) -> np.ndarray:
        
        """Mix noise into *audio* at the requested signal-to-noise ratio."""
        signal_power = np.mean(audio ** 2) + 1e-9
 
        if noise_type == "white":
            noise = np.random.randn(len(audio))
 
        elif noise_type == "pink":
            # Shape white noise with a 1/√f spectrum to approximate pink noise.
            f = np.fft.rfftfreq(len(audio))
            f[0] = 1.0          # avoid divide-by-zero at DC
            spectrum = np.random.randn(len(f)) / np.sqrt(f)
            noise = np.fft.irfft(spectrum, n=len(audio))
 
        else:
            return audio    # unknown type or empty pool — no-op
 
        noise_power = np.mean(noise ** 2) + 1e-9
        target_noise_power = signal_power / (10 ** (snr_db / 10))
        noise *= np.sqrt(target_noise_power / noise_power)
        return audio + noise

 
    def _pitch_shift(self, audio: np.ndarray, n_steps: float) -> np.ndarray:
        """Shift pitch by *n_steps* semitones without affecting duration."""
        return lb.effects.pitch_shift(
            audio.astype(np.float32),
            sr=self.processor.samplerate,
            n_steps=n_steps,
        ).astype(np.float64)
 
    def _volume_scale(self, audio: np.ndarray, gain: float) -> np.ndarray:
        """Multiply amplitude by *gain* and hard-clip to [-1, 1]."""
        return np.clip(audio * gain, -1.0, 1.0)
 
    # Feature-level augmentation (private) 
    def _spec_augment(self, mfcc: np.ndarray) -> np.ndarray:
        """
        SpecAugment — zero out random frequency bands and time frames.
 
        Operates on the MFCC matrix of shape (n_mels, n_frames).
        """
        mfcc = mfcc.copy()
        n_mels, n_frames = mfcc.shape
        cfg = self.config
 
        for _ in range(cfg.n_freq_masks):
            f  = random.randint(0, cfg.freq_mask_param)
            f0 = random.randint(0, max(0, n_mels - f))
            mfcc[f0 : f0 + f, :] = 0.0
 
        for _ in range(cfg.n_time_masks):
            t  = random.randint(0, cfg.time_mask_param)
            t0 = random.randint(0, max(0, n_frames - t))
            mfcc[:, t0 : t0 + t] = 0.0
 
        return mfcc
 
    
    # Public API 
    def process(self, audio: np.ndarray, hop: int = 512, noise=True, pitch=True, volume=True, spec_aug=True) -> np.ndarray:
        """
        Apply stochastic augmentations to a raw waveform and return MFCCs.
 
        Each augmentation fires independently according to its configured
        probability.  Waveform augmentations come first; SpecAugment is
        applied after MFCC extraction.
 
        Parameters
        ----------
        audio : np.ndarray
            Raw 1-D (or flattenable) audio waveform, float32 or float64.
        hop : int
            Hop length forwarded to AudioProcessor.CalcMFCC.
 
        Returns
        -------
        mfcc : np.ndarray, shape (n_mels, n_frames)
        """
        cfg   = self.config
        audio = audio.flatten().astype(np.float64)
 
        # 1 ── Noise
        if random.random() < cfg.noise_prob and noise:
            pool = list(cfg.noise_types)
            audio = self._add_noise(audio, random.uniform(*cfg.noise_snr_range), random.choice(pool))
 
        # 3 ── Pitch shift
        if random.random() < cfg.pitch_shift_prob and pitch:
            audio = self._pitch_shift(audio, random.uniform(*cfg.pitch_shift_range))
 
        # 4 ── Volume scaling
        if random.random() < cfg.volume_scale_prob and volume:
            audio = self._volume_scale(audio, random.uniform(*cfg.volume_gain_range))
 
        # 5 ── Extract MFCCs (via AudioProcessor)
        mfcc = self.processor.CalcMFCC(audio, hop=hop)
 
        # 6 ── SpecAugment
        if random.random() < cfg.spec_augment_prob and spec_aug:
            mfcc = self._spec_augment(mfcc)
 
        return mfcc
 


class AudioProcessor:
    def __init__(self, samplerate=15_872, window_duration=1, chunk_duration=0.25, n_fft=2048, n_mels=13):
        self.samplerate = samplerate
        self.chunk_size = int(samplerate * chunk_duration)  # 4000 samples @ 16kHz
        self.n_fft = n_fft                                  # Power of 2 → fast FFT
        self.window_size = int(samplerate * window_duration)
        self.n_mels = n_mels

        self.window = np.zeros(self.window_size)

        self.pitch = 0.0
        self.amp = 0.0
        self.note = None
        self.octave = None

        # --- Precomputed constants ---
        # Hanning sized to chunk, applied to time-domain signal (before FFT)
        self.hanningWindow = np.hanning(self.window_size)

        # Mel filterbank expects n_fft//2 + 1 bins (rfft output size)
        self.melFilters = lb.filters.mel(sr=samplerate, n_fft=n_fft, n_mels=n_mels)

        # Frequency axis for rfft output
        self.x_fft = np.fft.rfftfreq(n_fft, 1.0 / samplerate)

        self.notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        self.len_notes = len(self.notes)

    
    def update_window(self, frames):
        frames = frames.flatten()
        n = len(frames)
        self.window = np.roll(self.window, -n)  # np.roll is NOT in-place; result must be assigned
        self.window[-n:] = frames

    def freq_to_note(self, freq):

        if freq == 0:
            return self.notes[0], 0
        note_number = 12 * np.log2(freq / 440) + 49  
        note_number = round(note_number)
            
        note = (note_number - 1 ) % self.len_notes
        note = self.notes[note]
        
        octave = (note_number + 8 ) // self.len_notes
        
        return note, octave

    def CalcMFCC(self, soundData: np.ndarray, hop: int = 512) -> np.ndarray:

        sd = soundData.flatten()

        # Avoid spectral leakage by using a hanning window
        windowed = sd * self.hanningWindow     
        fft = np.abs(np.fft.rfft(windowed, n=self.n_fft))

        # Determine the peak frequence for pitch detection
        argmax = np.argmax(fft)
        pitch = self.x_fft[argmax]
        amplitude = fft[argmax]
        self.note, self.octave = self.freq_to_note(pitch)
        self.pitch, self.amp = pitch, amplitude

        # MFCC extraction from the sound data
        mfcc = lb.feature.mfcc(y=sd, sr=self.samplerate, n_mfcc=self.n_mels, hop_length=hop, norm='ortho')

        return mfcc

    def other_function(self):
        # This function can access the latest data whenever it wants
        print(f"Current Pitch: {self.pitch}, Amplitude: {self.amp}")
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
 
    def process_batch(
        self,
        samples: List[Tuple[np.ndarray, int]],
        n_augments: int = 3,
        hop: int = 512,
        noise: bool = True,
        pitch: bool = True,
        volume: bool = True, 
        spec_aug: bool = True
    ) -> Tuple[List[np.ndarray], List[int]]:
        """
        Augment a labelled dataset, generating *n_augments* extra copies per sample.
 
        The unaugmented original is always included so the clean signal is
        represented alongside its augmented variants.
 
        Parameters
        ----------
        samples : list of (audio, label) tuples
            *audio* is a raw waveform array; *label* is an integer class index.
        n_augments : int
            Number of augmented copies to produce per sample (default 3).
        hop : int
            Hop length forwarded to CalcMFCC.
 
        Returns
        -------
        features : list of np.ndarray
            MFCC arrays, length = len(samples) * (1 + n_augments).
        labels : list of int
            Corresponding class labels, same length as *features*.
 
        Example
        -------
        >>> samples = [(audio1, 0), (audio2, 1), (audio3, 0)]
        >>> features, labels = pipeline.process_batch(samples, n_augments=4)
        >>> len(features)   # 3 originals + 3 × 4 augmented = 15
        15
        """
        features: List[np.ndarray] = []
        labels:   List[int]        = []
 
        for audio, label in samples:
            audio = audio.flatten().astype(np.float64)
 
            # Unaugmented original
            features.append(self.processor.CalcMFCC(audio, hop=hop))
            labels.append(label)
 
            # Augmented copies
            for _ in range(n_augments):
                features.append(self.process(audio, hop=hop, noise=noise, volume=volume, spec_aug=spec_aug, pitch=pitch))
                labels.append(label)
 
        return features, labels
 

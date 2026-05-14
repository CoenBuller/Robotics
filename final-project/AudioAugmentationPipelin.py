import random
import torchaudio.transforms as T
import torch
import librosa as lb
import numpy as np

from tqdm import tqdm
from config import AugmentConfig
from typing import Optional

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
        sr: int = 15_872,
        n_mels: int = 62,
        hop: int = 512,
        n_fft: int = 512,
        config: Optional[AugmentConfig] = None,
        seed: Optional[int] = None,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.hop = hop
        self.hanningWindow = np.hanning(sr)

        self.config    = config or AugmentConfig()

        # Instances for calculating log mel spectograms
        self.mel_transform = T.MFCC(
                                    sample_rate=sr,
                                    n_mfcc=n_mels,
                                    log_mels=True,
                                    melkwargs={"hop_length": hop, "n_fft": n_fft, "n_mels": n_mels},
                                    )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=90)
 
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
    
    def _polarity_flip(self, audio: np.ndarray) -> np.ndarray:
         return -audio

 
    def _pitch_shift(self, audio: np.ndarray, n_steps: float) -> np.ndarray:
        """Shift pitch by *n_steps* semitones without affecting duration."""
        return lb.effects.pitch_shift(
            audio.astype(np.float32),
            sr=self.sr,
            n_steps=n_steps,
        ).astype(np.float64)
 
    def _volume_scale(self, audio: np.ndarray, gain: float) -> np.ndarray:
        """Multiply amplitude by *gain* and hard-clip to [-1, 1]."""
        return np.clip(audio * gain, -1.0, 1.0)
    

    def _time_shift(self, audio: np.ndarray, shift: float) -> np.ndarray:
        # shift is a fraction of total length, e.g. (-0.2, 0.2)
        n = int(shift * len(audio))
        return np.roll(audio, n) 
 

    # Feature-level augmentation (private) 
    def _spec_augment(self, result: np.ndarray) -> np.ndarray:
        result = result.copy()
        _, n_mels, n_frames = result.shape
        cfg = self.config
        fill = result.mean()

        # Onset channel is result[1], identical across mel rows — just read row 0
        onset_row = result[1, 0, :]                    # (n_frames,)
        protected = onset_row > 0.5                    # frames with strong transient

        for _ in range(cfg.n_freq_masks):
            f  = random.randint(0, cfg.freq_mask_param)
            f0 = random.randint(0, max(0, n_mels - f))
            result[:, f0 : f0 + f, :] = fill          # freq masking is fine for claps

        for _ in range(cfg.n_time_masks):
            t  = random.randint(0, cfg.time_mask_param)
            t0 = random.randint(0, max(0, n_frames - t))
            # Skip any frame where onset is strong
            maskable = [i for i in range(t0, min(t0 + t, n_frames))
                        if not protected[i]]
            if maskable:
                result[:, :, maskable] = fill

        return result
    
    
    def _onset_strength(self, log_mel: np.ndarray) -> np.ndarray:
        # Positive first-order difference across time axis
        diff = np.diff(log_mel, axis=1)              # (n_mels, n_frames-1)
        onset = np.mean(np.maximum(0, diff), axis=0) # (n_frames-1,)
        
        # Pad to match mel frame count
        onset = np.pad(onset, (1, 0))                # (n_frames,)
        return onset / (onset.max() + 1e-9)
        

    # Public API 
    def process(self, audio: np.ndarray, noise=True, pitch=True, volume=True, spec_aug=True, time_shift=True, polarity_flip=True, extra_noise=False) -> np.ndarray:
        """
        Apply stochastic augmentations to a raw waveform and return MFCCs.
        """
        cfg   = self.config
        audio = audio.flatten().astype(np.float64)
 
        # 1 ── Noise
        if random.random() < cfg.noise_prob and noise:
            pool = list(cfg.noise_types)
            delta = 0
            if extra_noise:
                delta = 5
            audio = self._add_noise(audio, random.uniform(*cfg.noise_snr_range)-delta, random.choice(pool))
 
        # 3 ── Pitch shift
        if random.random() < cfg.pitch_shift_prob and pitch:
            audio = self._pitch_shift(audio, random.uniform(*cfg.pitch_shift_range))

        # 4 ── Time shift
        if random.random() < cfg.time_shift_prob and time_shift:
            audio = self._time_shift(audio, random.uniform(*cfg.time_shift_range))
 
        # 5 ── Volume scaling
        if random.random() < cfg.volume_scale_prob and volume:
            audio = self._volume_scale(audio, random.uniform(*cfg.volume_gain_range))

        if polarity_flip and random.random() < 0.5:
            audio = self._polarity_flip(audio)
 
        # 6 ── Extract MFCCs
        audio_tensor = torch.from_numpy(audio.astype(np.float32))                  # shape: (audio) 
        mfcc= self.mel_transform(audio_tensor)  

        # 2D onset map
        onset_env = self._onset_strength(mfcc)                                     # shape: (n_frames,)
        onset_2d  = np.tile(onset_env, (self.n_mels, 1))                           # (n_mels, n_frames)
        # Concat results
        result = np.stack([mfcc, onset_2d], axis=0)                                # (2, n_mels, n_frames)

        # 7 ── SpecAugment
        if random.random() < cfg.spec_augment_prob and spec_aug:
            result = self._spec_augment(result)
 
        return result
 


class AugmentationScheduler:
    """
    Upgrades augmentation difficulty in stages as epoch accuracy crosses
    predefined thresholds.  Each stage is a full AugmentConfig; once a
    threshold is reached the scheduler replaces the pipeline's config and
    never steps down again (one-way ratchet).

    Thresholds are expressed as fractions (0–1).  Adjust the stage configs
    below to match your data and pipeline's parameter names.
    """

    def __init__(self, pipeline: AudioAugmentationPipeline):
        self.pipeline = pipeline
        self.current_stage = 0

        # ── Curriculum stages ─────────────────────────────────────────────────
        # Stage 0  (< 50 %)  – gentle: low probs, tight ranges
        # Stage 1  (≥ 50 %)  – moderate: higher probs, wider ranges
        # Stage 2  (≥ 70 %)  – hard: aggressive everything
        # Stage 3  (≥ 85 %)  – brutal: maximum pressure, more masks
        self.stages = [
            # threshold, config
            (0.50, AugmentConfig(
                noise_prob=0.5,       noise_snr_range=(15, 30),
                pitch_shift_prob=0.4, pitch_shift_range=(-2, 2),
                time_shift_prob=0.4, time_shift_range=(-0.1, 0.1),
                volume_scale_prob=0.4, volume_gain_range=(0.7, 1.3),
                spec_augment_prob=0.5,
                n_freq_masks=1, freq_mask_param=2,
                n_time_masks=1, time_mask_param=2,
                mixup_prob=0.4, alpha=0.2
            )),

            (0.70, AugmentConfig(
                noise_prob=0.65,      noise_snr_range=(10, 25),
                pitch_shift_prob=0.55, pitch_shift_range=(-3, 3),
                time_shift_prob=0.5, time_shift_range=(-0.15, 0.15),
                volume_scale_prob=0.55, volume_gain_range=(0.6, 1.4),
                spec_augment_prob=0.65,
                n_freq_masks=1, freq_mask_param=4,
                n_time_masks=1, time_mask_param=4,
                mixup_prob=0.4, alpha=0.3

            )),

            (0.85, AugmentConfig(
                noise_prob=0.80,      noise_snr_range=(5, 20),
                pitch_shift_prob=0.70, pitch_shift_range=(-5, 5),
                time_shift_prob=0.6, time_shift_range=(-0.2, 0.2),
                volume_scale_prob=0.70, volume_gain_range=(0.3, 1.5),
                spec_augment_prob=0.80,
                n_freq_masks=2, freq_mask_param=6,
                n_time_masks=2, time_mask_param=6,
                mixup_prob=0.5, alpha=0.4
            )),
        ]
        # ─────────────────────────────────────────────────────────────────────

    def step(self, epoch_acc: float) -> bool:
        """
        Call once per epoch with the overall accuracy (0–1).
        Returns True and logs a message if a new stage was unlocked.
        """
        if self.current_stage >= len(self.stages):
            return False  # Already at maximum difficulty

        threshold, new_cfg = self.stages[self.current_stage]
        if epoch_acc >= threshold:
            self.pipeline.config = new_cfg
            stage_num = self.current_stage + 1
            self.current_stage += 1
            tqdm.write(
                f"\n  ▲ Augmentation unlocked stage {stage_num} "
                f"(acc {epoch_acc * 100:.1f}% ≥ {threshold * 100:.0f}%)\n"
            )
            return True
        return False

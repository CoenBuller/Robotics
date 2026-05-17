import os
import random

import librosa as lb
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from tqdm import tqdm

from config import AugmentConfig
from FinalFolder.PitchExtraction import FastMelSpec
from FinalFolder.SoundClassifier import AudioCNN


#  Audio-level augmentation helpers
def _mix_at_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    """Add *noise* into *signal* scaled to the requested SNR (dB)."""
    sig_pwr   = np.mean(signal ** 2) + 1e-9
    noise_pwr = np.mean(noise  ** 2) + 1e-9
    scale     = np.sqrt(sig_pwr / (noise_pwr * 10 ** (snr_db / 10)))
    return signal + noise * scale


def _pitch_shift(audio: np.ndarray, sr: int, n_steps: float) -> np.ndarray:
    return lb.effects.pitch_shift(
        audio.astype(np.float32), sr=sr, n_steps=n_steps
    ).astype(np.float64)


def _volume_scale(audio: np.ndarray, gain: float) -> np.ndarray:
    return np.clip(audio * gain, -1.0, 1.0)


def _time_shift(audio: np.ndarray, shift: float) -> np.ndarray:
    return np.roll(audio, int(shift * len(audio)))


def _spec_augment(mel: np.ndarray, cfg: AugmentConfig) -> np.ndarray:
    result = mel.copy()
    n_mels, n_frames = result.shape
    fill = result.mean()

    for _ in range(np.random.randint(cfg.n_freq_masks)):
        f  = random.randint(0, n_mels-1)
        result[f, :] = fill          # freq masking is fine for claps

    for _ in range(np.random.randint(cfg.n_time_masks)):
        t  = random.randint(0, n_frames-1)
        result[:, t] = fill

    return result


# Car-noise pool helpers 
def load_noise_pool(noise_folder: str, sr: int) -> list[np.ndarray]:
    """Load every audio file in *noise_folder* at *sr* into a list."""
    pool = []
    for fname in sorted(os.listdir(noise_folder)):
        if not fname.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
            continue
        audio, _ = lb.load(os.path.join(noise_folder, fname), sr=sr)
        if len(audio) > 0:
            pool.append(audio.astype(np.float64))
            print(f"  Loaded noise file: {fname}  ({len(audio)/sr:.1f} s)")
    if not pool:
        raise ValueError(f"No audio files found in: {noise_folder}")
    return pool


def _sample_noise_chunk(pool: list[np.ndarray], n_samples: int) -> np.ndarray:
    """Return a random *n_samples*-long chunk from a random pool entry."""
    noise = random.choice(pool)
    if len(noise) < n_samples:                          # tile if too short
        noise = np.tile(noise, int(np.ceil(n_samples / len(noise))))
    start = random.randint(0, len(noise) - n_samples)
    return noise[start : start + n_samples]


#  Dataset 
class FineTuneDataset(Dataset):
    """
    Augments raw audio (optionally) and extracts log-mel via FastMelSpec.

    Augmentation order
    ──────────────────
    1. Pitch shift          (audio-level)
    2. Time shift           (audio-level)
    3. Volume scaling       (audio-level)
    4. Synthetic noise      (audio-level, keeps clean-condition robustness)
    5. Car noise mixing     (audio-level, the key new augmentation)
    6. Normalize to [-1,1]  (FastMelSpec also normalises internally; this
                             ensures SNR calculations above are meaningful)
    7. FastMelSpec          (feature extraction)
    8. SpecAugment          (feature-level)
    """

    def __init__(
        self,
        files:         list[str],
        labels:        list[int],
        mel_extractor: FastMelSpec,
        cfg:           AugmentConfig,
        noise_pool:    list[np.ndarray],
        sr:            int   = 15_872,
        augment:       bool  = True,
        add_car_noise: bool  = False,    # apply car noise even when augment=False
        #  Per-augmentation controls
        car_noise_prob:   float = 0.70,      # high — this is the main goal
        car_snr_range:    tuple = (2, 20),   # dB: lower = more car noise
        synth_noise_prob: float = 0.20,      # keep some clean/synthetic variety
        synth_snr_range:  tuple = (5, 30),
        pitch_prob:       float = 0.40,
        pitch_range:      tuple = (-4.0, 4.0),
        time_shift_prob:  float = 0.40,
        time_shift_range: tuple = (-0.4, 0.1),
        volume_prob:      float = 0.40,
        volume_range:     tuple = (0.7, 1.3),
        spec_aug_prob:    float = 0.30,
    ):
        self.files         = files
        self.labels        = labels
        self.mel           = mel_extractor
        self.cfg           = cfg
        self.noise_pool    = noise_pool
        self.sr            = sr
        self.augment       = augment

        self.add_car_noise    = add_car_noise

        self.car_noise_prob   = car_noise_prob
        self.car_snr_range    = car_snr_range
        self.synth_noise_prob = synth_noise_prob
        self.synth_snr_range  = synth_snr_range
        self.pitch_prob       = pitch_prob
        self.pitch_range      = pitch_range
        self.time_shift_prob  = time_shift_prob
        self.time_shift_range = time_shift_range
        self.volume_prob      = volume_prob
        self.volume_range     = volume_range
        self.spec_aug_prob    = spec_aug_prob

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        label    = self.labels[idx]
        audio, _ = lb.load(self.files[idx], sr=self.sr, duration=1.0)
        audio    = audio.astype(np.float64)

        # Pad / trim to exactly 1 second
        if len(audio) < self.sr:
            audio = np.pad(audio, (0, self.sr - len(audio)))
        else:
            audio = audio[: self.sr]

        if self.augment:
            # 1. Pitch shift
            if random.random() < self.pitch_prob:
                audio = _pitch_shift(audio, self.sr, random.uniform(*self.pitch_range))

            # 2. Time shift
            if random.random() < self.time_shift_prob:
                audio = _time_shift(audio, random.uniform(*self.time_shift_range))

            # 3. Volume scaling
            if random.random() < self.volume_prob:
                audio = _volume_scale(audio, random.uniform(*self.volume_range))

            # 4. Synthetic noise (white / pink)
            if random.random() < self.synth_noise_prob:
                snr   = random.uniform(*self.synth_snr_range)
                if random.random() < 0.5:
                    noise = np.random.randn(len(audio))
                else:
                    f         = np.fft.rfftfreq(len(audio))
                    f[0]      = 1.0                         # avoid /0 at DC
                    spectrum  = np.random.randn(len(f)) / np.sqrt(f)
                    noise     = np.fft.irfft(spectrum, n=len(audio))
                audio = _mix_at_snr(audio, noise, snr)

            # 5. Car noise (training path — probabilistic)
            if random.random() < self.car_noise_prob:
                car   = _sample_noise_chunk(self.noise_pool, len(audio))
                snr   = random.uniform(*self.car_snr_range)
                audio = _mix_at_snr(audio, car, snr)

        # 5b. Car noise for validation (always applied when add_car_noise=True)
        elif self.add_car_noise:
            car   = _sample_noise_chunk(self.noise_pool, len(audio))
            snr   = random.uniform(*self.car_snr_range)
            audio = _mix_at_snr(audio, car, snr)

        # 6. Normalize — SNR mixing is done, safe to normalise now
        audio = audio / (np.max(np.abs(audio)) + 1e-9)

        # 7. Log-mel via FastMelSpec 
        log_mel, _, _ = self.mel(audio.astype(np.float32))   # (n_mels, n_frames)

        # 8. SpecAugment (feature-level)
        if self.augment and random.random() < self.spec_aug_prob:
            log_mel = _spec_augment(log_mel, self.cfg)

        # Shape expected by AudioCNN: (1, n_mels, n_frames)
        return torch.tensor(log_mel, dtype=torch.float32).unsqueeze(0), label


# Validation helper

def _class_accuracy_table(correct: np.ndarray, total: np.ndarray, names: list[str]) -> str:
    col_w  = max(len(n) for n in names) + 2
    header = "  ".join(f"{n:>{col_w}}" for n in names)
    accs   = [f"{(c/t*100 if t else 0.0):>{col_w}.1f}%" for c, t in zip(correct, total)]
    return f"  {header}\n  {'  '.join(accs)}"


def _run_validation(model, loader, loss_fn, n_classes: int):
    model.eval()
    val_loss = 0.0
    correct  = np.zeros(n_classes, dtype=np.int64)
    total    = np.zeros(n_classes, dtype=np.int64)
    with torch.no_grad():
        for X, y in loader:
            logits    = model(X)
            val_loss += loss_fn(logits, y).item()
            preds     = logits.argmax(dim=1)
            for cls in range(n_classes):
                mask         = y == cls
                correct[cls] += (preds[mask] == cls).sum().item()
                total[cls]   += mask.sum().item()
    return val_loss / len(loader), correct, total


# Fine-tune loop

def finetune(
    files:           list[str],
    labels:          list[int],
    n_classes:       int,
    class_names:     list[str],
    pretrained_path: str,
    noise_folder:    str,
    mel_fb_path:     str,
    save_path:       str   = "finetuned_model.pt",
    epochs:          int   = 40,
    lr:              float = 3e-4,      # keep it low
    val_split:       float = 0.10,
    patience:        int   = 8,
    freeze_features: bool  = False,     # True → only train the classifier head
    seed:            int   = 42,
):
    """
    Parameters
    ──────────
    pretrained_path : path to the .pt file saved by TrainClassifier.py
    noise_folder    : directory containing the recorded car-noise files
    mel_fb_path     : path to mel_fb.npy (same one used at inference)
    freeze_features : set True for a very quick first pass — tunes only the
                      two Linear layers while keeping the CNN frozen
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    sr = 15_872

    # Car-noise pool
    print(f"\nLoading car noise from: {noise_folder}")
    noise_pool = load_noise_pool(noise_folder, sr)
    print(f"  → {len(noise_pool)} file(s), pool ready\n")

    # Feature extractor (log Mel-spectograms)
    mel_extractor = FastMelSpec(mel_fb_path=mel_fb_path, sr=sr, n_fft=512, hop=512)

    # Load pretrained weights
    model = AudioCNN(n_classes=n_classes)
    state = torch.load(pretrained_path, map_location="cpu")
    model.load_state_dict(state)
    print(f"Loaded pretrained weights from: {pretrained_path}")

    if freeze_features:
        for p in model.features.parameters():
            p.requires_grad = False
        print("Feature layers FROZEN — only classifier head will be updated\n")

    # Stratified train / val split
    files_arr  = np.array(files)
    labels_arr = np.array(labels)
    train_files, val_files, train_labels, val_labels = train_test_split(
        files_arr, labels_arr,
        test_size=val_split, stratify=labels_arr, random_state=seed,
    )
    tqdm.write(f"Train: {len(train_files)}  |  Val: {len(val_files)}")
    for i, name in enumerate(class_names):
        tqdm.write(f"  {name}: {(train_labels==i).sum()} train / {(val_labels==i).sum()} val")
    tqdm.write("")

    # Datasets & loaders
    cfg      = AugmentConfig(n_time_masks=6, n_freq_masks=6)                         # used only for spec-augment mask counts
    train_ds = FineTuneDataset(train_files.tolist(), train_labels.tolist(), 
                               mel_extractor, cfg, noise_pool, sr=sr, augment=True,)
    

    val_ds   = FineTuneDataset(val_files.tolist(), val_labels.tolist(),
                              mel_extractor, cfg, noise_pool, sr=sr,
                              augment=False, add_car_noise=True,) # No augmentation, but car noise applied

    # Weighted sampler
    counts   = np.bincount(train_labels, minlength=n_classes)
    weights  = 1.0 / np.maximum(counts, 1)
    sample_w = [float(weights[l]) for l in train_labels]
    sampler  = WeightedRandomSampler(sample_w, num_samples=len(sample_w), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=32, sampler=sampler, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=32, shuffle=False,   num_workers=4)

    # Optimiser
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt       = torch.optim.AdamW(trainable, lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=4, min_lr=1e-6,
    )
    # Small label smoothing helps regularise when fine-tuning on a small delta
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    # Early-stopping state
    best_val_loss     = float("inf")
    epochs_no_improve = 0

    epoch_bar = tqdm(range(epochs), desc="Epochs", position=0)

    for epoch in epoch_bar:
        # Training pass
        model.train()
        total_loss = 0.0
        correct    = np.zeros(n_classes, dtype=np.int64)
        total      = np.zeros(n_classes, dtype=np.int64)

        batch_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1:>3}/{epochs}",
            position=1, leave=False,
        )
        for X, y in batch_bar:
            opt.zero_grad()
            logits = model(X)
            loss   = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            for cls in range(n_classes):
                mask         = y == cls
                correct[cls] += (preds[mask] == cls).sum().item()
                total[cls]   += mask.sum().item()
            batch_bar.set_postfix(loss=f"{loss.item():.4f}",
                                  lr=f"{opt.param_groups[0]['lr']:.2e}")

        train_acc = correct.sum() / total.sum()
        avg_train = total_loss / len(train_loader)

        # Validation pass 
        avg_val, val_correct, val_total = _run_validation(
            model, val_loader, loss_fn, n_classes
        )
        val_acc = val_correct.sum() / val_total.sum()

        scheduler.step(avg_val)

        tqdm.write(
            f"\nEpoch {epoch+1}"
            f"  train_loss={avg_train:.4f}  train_acc={train_acc*100:.1f}%"
            f"  val_loss={avg_val:.4f}  val_acc={val_acc*100:.1f}%"
            f"  lr={opt.param_groups[0]['lr']:.2e}"
            f"\n  val per-class:\n{_class_accuracy_table(val_correct, val_total, class_names)}\n"
        )
        epoch_bar.set_postfix(val_loss=f"{avg_val:.4f}", val_acc=f"{val_acc*100:.1f}%")

        # ── Checkpoint & early stopping ───────────────────────────────────────
        if avg_val < best_val_loss:
            best_val_loss     = avg_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), save_path)
            tqdm.write(f"  ✓ Best val_loss={best_val_loss:.4f} — saved to {save_path}")
        else:
            epochs_no_improve += 1
            tqdm.write(f"  No improvement {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                tqdm.write(
                    f"\n⚑ Early stopping at epoch {epoch+1}. "
                    f"Best val_loss: {best_val_loss:.4f}. "
                    f"Loading best weights from {save_path}."
                )
                model.load_state_dict(torch.load(save_path))
                break

    return model


# Entry point 

if __name__ == "__main__":
    DATA_FOLDER      = "data"
    NOISE_FOLDER     = "final_car_noise"                       # ← your car recordings
    PRETRAINED_PATH  = "final-project/models/best_model.pt"    # ← original model
    MEL_FB_PATH      = "final-project/FinalFolder/mel_fb.npy"
    SAVE_PATH        = "final-project/models/finetuned_model.pt"

    class_folders = sorted(os.listdir(DATA_FOLDER))
    audio_files: list[str] = []
    labels:      list[int] = []

    for label, folder in enumerate(class_folders):
        folder_path = os.path.join(DATA_FOLDER, folder)
        for fname in os.listdir(folder_path):
            audio_files.append(os.path.join(folder_path, fname))
            labels.append(label)

    print(f"Classes : {class_folders}")
    print(f"Samples : {len(audio_files)}")

    model = finetune(
        files           = audio_files,
        labels          = labels,
        n_classes       = len(class_folders),
        class_names     = class_folders,
        pretrained_path = PRETRAINED_PATH,
        noise_folder    = NOISE_FOLDER,
        mel_fb_path     = MEL_FB_PATH,
        save_path       = SAVE_PATH,
        epochs          = 40,
        lr              = 3e-4,         # 10× lower than original training
        val_split       = 0.10,
        patience        = 8,
        freeze_features = False,        # set True for a fast first-pass run
    )
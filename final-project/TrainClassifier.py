# train.py
import torch
import numpy as np
import librosa as lb
import os
import torch.onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from SoundClassifier import AudioCNN
from torch import nn
from AudioProcessPipeline import AudioAugmentationPipeline, AudioProcessor, AugmentConfig


def export_for_pi(model, save_dir: str, quantize: bool = True):
    model.eval()

    dummy = torch.randn(1, 1, 62, 32)
    path = os.path.join(save_dir, "cnn_model.onnx")

    torch.onnx.export(
        model, (dummy,), path,
        input_names=["melspec"],
        output_names=["logits"],
        dynamic_axes={"melspec": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported to {path}")

    if quantize:
        quant_path = os.path.join(save_dir, "cnn_model_int8.onnx")
        quantize_dynamic(
            model_input=path,
            model_output=quant_path,
            weight_type=QuantType.QInt8,
        )
        print(f"Quantized model saved to {quant_path}")

def createLabels(audio_files: list[str], classes: list[str]):
    classes_dict = {c: i for i, c in enumerate(classes)}
    files = []
    labels = []

    for file in audio_files:
        c = file.split("_")[0]
        c = c.split("\\")[-1]

        files.append(file)
        labels.append(classes_dict[c])

    return files, labels


class AudioDataset(Dataset):
    def __init__(self, files: list[str], labels: list[int], audio_augmenter: AudioAugmentationPipeline, hop: int = 512, augment: bool = True, noise: bool = True, pitch: bool = True, volume: bool = True, spec_aug: bool = True):
        self.X = files
        self.y = labels
        self.aa = audio_augmenter
        self.augment = augment
        self.hop = hop
        self.noise = noise
        self.pitch = pitch
        self.volume = volume
        self.spec_aug = spec_aug

    def __getitem__(self, idx):
        x = self.X[idx]
        audio_data, sr = lb.load(path=x, sr=self.aa.sr, duration=1)

        # If audio is not exactly 1 second long, pad so it can be processed by the network
        audio_len = len(audio_data)
        if audio_len < sr:
            d = sr - audio_len
            audio_data = np.pad(audio_data, pad_width=(0, d)) # type: ignore

        if not self.augment:
            x = self.aa.process(audio=audio_data, noise=False, pitch=False, volume=False, spec_aug=False)
        else:
            x = self.aa.process(audio=audio_data, noise=self.noise, pitch=self.pitch, volume=self.volume, spec_aug=self.spec_aug)

        return torch.tensor(x, dtype=torch.float32).unsqueeze(0), self.y[idx]  # (1, 62, 32)

    def __len__(self):
        return len(self.X)


def _class_accuracy_table(class_correct: np.ndarray, class_total: np.ndarray, class_names: list[str]) -> str:
    """Return a compact per-class accuracy string for tqdm.write."""
    col_w = max(len(n) for n in class_names) + 2
    header = "  ".join(f"{n:>{col_w}}" for n in class_names)
    accs = []
    for correct, total in zip(class_correct, class_total):
        acc = (correct / total * 100) if total > 0 else 0.0
        accs.append(f"{acc:>{col_w}.1f}%")
    row = "  ".join(accs)
    return f"  {header}\n  {row}"


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
                volume_scale_prob=0.4, volume_gain_range=(0.7, 1.3),
                spec_augment_prob=0.5,
                n_freq_masks=1, freq_mask_param=2,
                n_time_masks=1, time_mask_param=2,
            )),
            (0.70, AugmentConfig(
                noise_prob=0.65,      noise_snr_range=(10, 25),
                pitch_shift_prob=0.55, pitch_shift_range=(-3, 3),
                volume_scale_prob=0.55, volume_gain_range=(0.6, 1.4),
                spec_augment_prob=0.65,
                n_freq_masks=1, freq_mask_param=4,
                n_time_masks=1, time_mask_param=4,
            )),
            (0.85, AugmentConfig(
                noise_prob=0.80,      noise_snr_range=(5, 20),
                pitch_shift_prob=0.70, pitch_shift_range=(-5, 5),
                volume_scale_prob=0.70, volume_gain_range=(0.3, 1.5),
                spec_augment_prob=0.80,
                n_freq_masks=2, freq_mask_param=6,
                n_time_masks=2, time_mask_param=6,
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


def train(files, labels, audio_processor, n_classes, class_names: list[str] | None = None, epochs=500, lr=3e-3):
    if class_names is None:
        class_names = [str(i) for i in range(n_classes)]
    dataset = AudioDataset(files, labels, audio_processor, augment=True)

    # Weighted sampler to counter class imbalance
    class_counts = np.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = [float(class_weights[label]) for label in labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

    loader = DataLoader(dataset, batch_size=32, num_workers=4, sampler=sampler, pin_memory=True)
    steps_per_epoch = len(loader)

    model = AudioCNN(n_classes=n_classes)

    # ── Hyperparameter choices ────────────────────────────────────────────────
    # AdamW with slightly higher weight_decay (1e-3) for better regularisation
    # on small audio datasets.
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)

    # OneCycleLR: warms up to max_lr then cosine-anneals to near-zero.
    # Far more effective than ReduceLROnPlateau for fixed-epoch training runs;
    # eliminates the need to hand-tune patience.
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=lr,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        pct_start=0.1,          # 10 % warm-up
        anneal_strategy="cos",
        div_factor=10.0,        # start lr = max_lr / 10
        final_div_factor=1e3,   # end lr = max_lr / 1000
    )

    # Label smoothing (0.1) acts as a regulariser and prevents over-confident
    # predictions — especially useful when the dataset is small.
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    # ─────────────────────────────────────────────────────────────────────────

    aug_scheduler = AugmentationScheduler(audio_processor)

    epoch_bar = tqdm(range(epochs), desc="Epochs", position=0)

    for epoch in epoch_bar:
        model.train()
        total_loss = 0.0
        last_acc = 0.0

        # Per-class accumulators reset every epoch
        class_correct = np.zeros(n_classes, dtype=np.int64)
        class_total   = np.zeros(n_classes, dtype=np.int64)

        batch_bar = tqdm(
            loader,
            desc=f"Epoch {epoch + 1:>4}/{epochs}",
            position=1,
            leave=False,
            total=steps_per_epoch,
        )

        for batch_idx, (X_batch, y_batch) in enumerate(batch_bar):
            opt.zero_grad()

            logits = model(X_batch)
            loss = loss_fn(logits, y_batch)
            loss.backward()

            # Gradient clipping prevents exploding gradients without touching lr
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            opt.step()
            sched.step()  # OneCycleLR steps every batch, not every epoch

            total_loss += loss.item()

            # Batch accuracy
            preds = logits.argmax(dim=1)
            last_acc = (preds == y_batch).float().mean().item()

            # Accumulate per-class hits for the epoch summary
            for cls in range(n_classes):
                mask = y_batch == cls
                class_correct[cls] += (preds[mask] == cls).sum().item()
                class_total[cls]   += mask.sum().item()

            # Current learning rate (same for all param groups here)
            current_lr = opt.param_groups[0]["lr"]

            batch_bar.set_postfix(
                batch=f"{batch_idx + 1}/{steps_per_epoch}",
                loss=f"{loss.item():.4f}",
                acc=f"{last_acc * 100:.1f}%",
                lr=f"{current_lr:.2e}",
            )

        avg_loss = total_loss / steps_per_epoch
        epoch_acc = class_correct.sum() / class_total.sum()  # overall accuracy this epoch
        epoch_bar.set_postfix(avg_loss=f"{avg_loss:.4f}", acc=f"{epoch_acc * 100:.1f}%")

        # Print per-class breakdown below the bars every epoch
        table = _class_accuracy_table(class_correct, class_total, class_names)
        tqdm.write(f"\nEpoch {epoch + 1} — avg loss: {avg_loss:.4f}  acc: {epoch_acc * 100:.1f}%\n{table}\n")

        # Curriculum: ratchet up augmentation difficulty when accuracy improves
        aug_scheduler.step(epoch_acc)

    return model


if __name__ == "__main__":
    data_folder = "data"
    class_folders = os.listdir(data_folder)

    audio_files = []
    labels = []

    # Read all files and label them by their folder name
    for label, folder in tqdm(enumerate(class_folders)):
        p = os.path.join(data_folder, folder)
        files = os.listdir(p)
        for file in files:
            fp = os.path.join(p, file)
            audio_files.append(fp)
            labels.append(label)

    print(f"Example audio path, with corresponding label: {audio_files[0]} | {labels[0]}")
    print(f"Classes: {class_folders}")
    print(np.unique(labels))

    cfg = AugmentConfig(noise_prob=0.5,
                        noise_snr_range=(10, 30),
                        pitch_shift_prob=0.4,
                        pitch_shift_range=(-3, 3),
                        volume_scale_prob=0.4,
                        volume_gain_range=(0.5, 1.5),
                        spec_augment_prob=0.6,
                        n_freq_masks=1,
                        freq_mask_param=1,
                        n_time_masks=1,
                        time_mask_param=1)

    ap = AudioAugmentationPipeline(config=cfg, n_mels=62)
    model = train(
        files=audio_files,
        labels=labels,
        audio_processor=ap,
        n_classes=len(class_folders),
        class_names=class_folders,
        epochs=250,
    )
    torch.save(model, os.path.join("final-project", "models", "cnn_model"))
    export_for_pi(model, save_dir=os.path.join("final-project", "models", "onnx_cnn_model"))
# train.py
import torch
import numpy as np
import librosa as lb
import os
import torch.onnx
from onnxruntime.quantization import quantize_dynamic, QuantType
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from SoundClassifier2 import AudioCNN
from torch import nn
from AudioAugmentationPipelin import AudioAugmentationPipeline, AugmentationScheduler
from config import AugmentConfig


def export_for_pi(model, filename: str, quantize: bool = True):
    model.eval()

    # Create export directory
    save_dir = "final-project/models/onnx_cnn_model"
    os.makedirs(save_dir, exist_ok=True)

    # Dummy input matching your expected input dimensions
    dummy = torch.randn(1, 2, 62, 32) 

    # Export ONNX with Opset 13
    onnx_path = os.path.join(save_dir, filename)

    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        input_names=["melspec"],
        output_names=["logits"],
        opset_version=13,
        dynamo=False,  
        dynamic_axes={
            'melspec': {0: 'batch_size'},
            'logits': {0: 'batch_size'}
        }
    )

    print("Saved:", onnx_path)

    # Quantize
    quant_path = os.path.join(save_dir, filename.rstrip(".onnx") + "_int8.onnx")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quant_path,
        weight_type=QuantType.QInt8,
    )
    print("Saved quantized:", quant_path)



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



class AudioDataset(Dataset):
    def __init__(self, files: list[str], labels: list[int], audio_augmenter: AudioAugmentationPipeline, hop: int = 512, augment: bool = True, noise: bool = True, pitch: bool = True, volume: bool = True, spec_aug: bool = True, timeshift: bool = True):
        self.X = files
        self.y = labels
        self.aa = audio_augmenter
        self.augment = augment
        self.hop = hop
        self.noise = noise
        self.pitch = pitch
        self.volume = volume
        self.spec_aug = spec_aug
        self.timeshift = timeshift

    def __getitem__(self, idx):
        x = self.X[idx]
        audio_data, sr = lb.load(path=x, sr=self.aa.sr, duration=1)

        # If audio is not exactly 1 second long, pad so it can be processed by the network
        audio_len = len(audio_data)
        if audio_len < sr:
            d = sr - audio_len
            audio_data = np.pad(audio_data, pad_width=(0, d)) # type: ignore

        is_clap = (self.y[idx] == 0)

        x = self.aa.process(
            audio=audio_data,
            noise=True,
            pitch=not is_clap,      # pitch shift is less meaningful for claps
            volume=True,
            spec_aug=True,
            time_shift=True,
            polarity_flip=is_clap,  # free augmentation, very effective for claps
            extra_noise=is_clap,    # slightly heavier noise for clap only
        )

        x = self.aa.process(audio=audio_data, noise=self.noise, pitch=self.pitch, volume=self.volume, spec_aug=self.spec_aug, time_shift=self.timeshift)

        return torch.tensor(x, dtype=torch.float32), self.y[idx]  # (2, 62, 32)

    def __len__(self):
        return len(self.X)


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
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
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

    cfg = AugmentConfig(                
                        noise_prob=0.4,       noise_snr_range=(20, 30),
                        pitch_shift_prob=0.4, pitch_shift_range=(-1, 1),
                        time_shift_prob=0.3, time_shift_range=(-0.1, 0.1),
                        volume_scale_prob=0.4, volume_gain_range=(0.8, 1.2),
                        spec_augment_prob=0.2,
                        n_freq_masks=1, freq_mask_param=1,
                        n_time_masks=1, time_mask_param=1,
                        mixup_prob=0.4, alpha=0.2
                        )

    ap = AudioAugmentationPipeline(config=cfg, n_mels=62, hop=512, n_fft=512, sr=15_872)
    model = train(
        files=audio_files,
        labels=labels,
        audio_processor=ap,
        n_classes=len(class_folders),
        class_names=class_folders,
        epochs=250,
    )
    torch.save(model, os.path.join("final-project", "models", "cnn_model2"))
    export_for_pi(model, filename="onnx_cnn_model2")
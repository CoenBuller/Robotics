# train.py
import torch
import numpy as np
import librosa as lb
import os 

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from SoundClassifier import AudioCNN
from torch import nn
from audioProcessor import AudioProcessor


def createLabels(audio_files: list[str], classes: list[str]):
    classes_dict = {c:i for i, c in enumerate(classes)}
    files = []
    labels = []

    for file in audio_files:
        c = file.split("_")[0]
        c = c.split("\\")[-1]

        files.append(file)
        labels.append(classes_dict[c])
    
    return files, labels

def createSpectograms(audio_files: list[str]):
    ap = AudioProcessor(samplerate=16_000, window_duration=1, chunk_duration=0.25)
    spectorgrams = []
    for f in audio_files:
        d, _ = lb.load(f, sr=ap.samplerate, duration=1)
        mfcc = ap.CalcMFCC(soundData=d)
        spectorgrams.append(mfcc)

    return spectorgrams


class AudioDataset(Dataset):
    def __init__(self, spectrograms, labels, augment=False):
        self.X = spectrograms   # List of (13, 32) arrays
        self.y = labels
        self.augment = augment

    def _augment(self, x: np.ndarray) -> np.ndarray:
        # 1. Time shift — roll frames left/right by up to 4 frames
        shift = np.random.randint(-4, 4)
        x = np.roll(x, shift, axis=1)

        # 2. Frequency masking (SpecAugment) — zero out up to 3 mel bands
        f0 = np.random.randint(0, x.shape[0] - 3)
        x[f0:f0 + np.random.randint(1, 3), :] = 0

        # 3. Time masking — zero out up to 4 consecutive frames
        t0 = np.random.randint(0, x.shape[1] - 4)
        x[:, t0:t0 + np.random.randint(1, 4)] = 0

        # 4. Gaussian noise
        x += np.random.normal(0, 0.02, x.shape)
        return x

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        if self.augment:
            x = self._augment(x)
        return torch.tensor(x, dtype=torch.float32), self.y[idx]

    def __len__(self):
        return len(self.X)


def train(spectrograms, labels, n_classes, epochs=100, lr=1e-3):
    dataset = AudioDataset(spectrograms, labels, augment=True)
    loader  = DataLoader(dataset, batch_size=16, shuffle=True)

    model = AudioCNN(n_classes=n_classes)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # LR drops 10× if val loss plateaus — important with small data
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)  # Smoothing helps small data

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        for X_batch, y_batch in loader:
            opt.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            opt.step()
            total_loss += loss.item()
        sched.step(total_loss)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: loss={total_loss/len(loader):.4f}")

    return model


if __name__ == "__main__":

    classes = ["whistle", "harmonica", "silence", "clap"]

    data_folder = "data"
    class_folders = os.listdir(data_folder)
    audio_files = []
    for folder in class_folders:
        p = os.path.join(data_folder, folder)
        files = os.listdir(p)
        for file in files:
            fp = os.path.join(p, file)
            audio_files.append(fp)


    files, labels = createLabels(audio_files=audio_files, classes=classes)

    print(files, labels)
    spectograms = createSpectograms(files)

    model = train(spectrograms=spectograms, labels=labels, n_classes=4, epochs=500)
    torch.save(model, os.path.join("final-project","models", "cnn_model"))

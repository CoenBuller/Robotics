# train.py
import torch
import numpy as np
import librosa as lb
import os 
import torchaudio

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from SoundClassifier import AudioCNN
from torch import nn
from AudioProcessPipeline import AudioAugmentationPipeline, AudioProcessor, AugmentConfig

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


class AudioDataset(Dataset):
    def __init__(self, files: list[str], labels: list[int], audio_augmenter: AudioAugmentationPipeline, hop: int=512, augment: bool=True, noise: bool=True, pitch: bool=True, volume: bool=True, spec_aug: bool=True):
        self.X = files # List of (13, 32) arrays
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

        #  If audio is not exactly 1 second long, we pad the audio data so it can be processed by the network
        audio_len = len(audio_data)
        if audio_len < sr:
            d = sr - audio_len
            audio_data = np.pad(audio_data, pad_width=(0, d))

        # Can choose if we want to augment the data or not
        if not self.augment:
            x = self.aa.process(audio=audio_data, noise=False, pitch=False, volume=False, spec_aug=False)
        else:
            x = self.aa.process(audio=audio_data, noise=self.noise, pitch=self.pitch, volume=self.volume, spec_aug=self.spec_aug)


        return torch.tensor(x, dtype=torch.float32).unsqueeze(0), self.y[idx] # Data has shape of (1, 13, 31)

    def __len__(self):
        return len(self.X)


def train(files, labels, audio_processor, n_classes, epochs=100, lr=1e-3):
    dataset = AudioDataset(files, labels, audio_processor, augment=True) # Create a dataset of all the filenames and labels

    class_counts = np.bincount(labels)
    class_weights = 1 / class_counts # If a class is more present within the dataset, we want a lower weight for them to get sampled to negate class imbalance in the dataset
    class_weights = [float(class_weights[label]) for label in labels]

    # Due to class imbalance we will wuse a weighted random sampler, where each datapoint gets a weight to get picked. Higher weight = more likely to get picked.
    sampler = WeightedRandomSampler(weights=class_weights, 
                                    num_samples=len(class_weights), 
                                    replacement=True)
    loader  = DataLoader(dataset, batch_size=32, num_workers=4, sampler=sampler)

    model = AudioCNN(n_classes=n_classes)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=100)
    loss_fn = nn.CrossEntropyLoss()  # Smoothing helps small data

    for epoch in tqdm(range(epochs)):
        model.train()
        total_loss = 0
        next_print = 5
        for X_batch, y_batch in loader:
            opt.zero_grad()
            loss = loss_fn(model(X_batch), y_batch)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            next_print -= 1
            if next_print <= 0:
                tqdm.write(f"Last loss: {loss}")
                next_print = 50
        sched.step(total_loss)


    return model


if __name__ == "__main__":

    # classes = ["whistle", "harmonica", "silence", "clap"]

    data_folder = "data"
    class_folders = os.listdir(data_folder)

    audio_files = []
    labels = []

    # Readin all files to load in and labelling them. 
    for label, folder in tqdm(enumerate(class_folders)):
        p = os.path.join(data_folder, folder)
        files = os.listdir(p)
        for file in files:
            fp = os.path.join(p, file)
            audio_files.append(fp)
            labels.append(label)

    print(f"Example audio path, with corresponding label: {audio_files[0]} | {labels[0]} ")
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

    ap = AudioAugmentationPipeline(config=cfg)
    model = train(files=audio_files, labels=labels, audio_processor=ap, n_classes=4, epochs=500)
    torch.save(model, os.path.join("final-project","models", "cnn_model"))

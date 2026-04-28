import numpy as np
import torch 

from torch import Tensor
from torch import nn

class AudioCNN(nn.Module):
    def __init__(self, n_classes: int, n_mfcc: int = 13):
        super().__init__()

        self.n_classes = n_classes
        self.n_mfcc = n_mfcc
        
        self.features = nn.Sequential(
            # Block 1 — local patterns (~3 frames)
            nn.Conv1d(self.n_mfcc, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Block 2 — higher-level patterns
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64, self.n_classes),
            nn.Softmax(-1)
        )

    def forward(self, x: Tensor):
        x = self.features(x)
        x = x.mean(dim=-1)
        return self.classifier(x)

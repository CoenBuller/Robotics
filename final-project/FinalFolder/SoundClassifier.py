import torch 

from torch import Tensor
from torch import nn

class AudioCNN(nn.Module):
    def __init__(self, n_classes: int, n_mfcc: int = 62):
        super().__init__()

        self.n_classes = n_classes
        self.n_mfcc = n_mfcc
       
        self.features = torch.nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(5,5), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(3,3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(num_features=32),
            nn.AdaptiveAvgPool2d(output_size=(1,1))
        ) 

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=32, out_features=n_classes),
            nn.Softmax(dim=-1)
        )

    def forward(self, x: Tensor):
        # Input (x) shape: (B, 1, 62, 32)
        x = self.features(x)
        # Features output shape: (B, 32, 1, 1)
        x = torch.flatten(x, start_dim=1, end_dim=-1)
        return self.classifier(x) # Output shape : (B, 4)

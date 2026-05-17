import numpy as np
import librosa as lb


def augment(y, sr):
    variants = [y]
    # Pitch shift
    variants.append(lb.effects.pitch_shift(y, sr=sr, n_steps=np.random.uniform(-2, 2)))
    # Add noise
    noise = np.random.randn(len(y)) * 0.005
    variants.append(y + noise)
    return variants
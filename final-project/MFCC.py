import numpy as np
import scipy as sp

def CalcMFCC(soundData: np.ndarray, hanningWindow: np.ndarray, melFilters: np.ndarray, x_fft: np.ndarray) -> tuple[np.ndarray, int]:
    fft = np.abs(np.fft.rfft(soundData.flatten(), n=len(hanningWindow)*2 - 1))

    pitch = x_fft[np.argmax(fft)] # Calculates the pitch

    # Calculates the mfcc 
    hanningFft = fft * hanningWindow
    melFft = np.dot(melFilters, hanningFft)
    dctFft = sp.fftpack.dct(melFft, type=2)

    return dctFft, pitch

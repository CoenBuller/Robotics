"""Code can be used to plot the FFT of the sounddata. But might be redundant for now"""

import numpy as np
import matplotlib.pyplot as plt
import sounddevice as sd

from matplotlib.animation import FuncAnimation

def update(frame, audio_queue):
    data = None
    # Flush the queue to get the absolute latest chunk
    while not audio_queue.empty():
        data = audio_queue.get_nowait()
    
    if data is None:
        return line, vline

    # Flatten the (512, 1) array to (512,) for the FFT
    flat_data = data.flatten()
    fft_data = np.abs(np.fft.rfft(flat_data))
    peak_freq = np.argmax(fft_data)
    # Update the existing line instead of removing/replotting (much faster)
    line.set_ydata(fft_data)
    peak = x_fft[peak_freq]
    vline.set_xdata([peak, peak])
    print(peak)
    return line, vline

fig, ax = plt.subplots()
x_fft = np.fft.rfftfreq(sd.default.blocksize, 1/sd.default.samplerate)
line, = ax.plot(x_fft, np.zeros(len(x_fft)))
vline = ax.axvline(x=0, color='r', linestyle='--')

ax.set_ylim(0, 100000) # Adjust based on expected volume
ax.set_title("Live FFT")
ax.set_xlabel("Frequency (Hz)")
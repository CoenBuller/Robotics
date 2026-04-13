import sounddevice as sd
import numpy as np
import queue 
import librosa as lb
import argparse
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from audioProcessor import AudioProcessor
parser = argparse.ArgumentParser()
parser.add_argument("--d", type=int, help="Decide which device you want to record with (int)")
parser.add_argument("--samplerate", type=int, help="Devine the sample rate at which you want to record your audio (int)", default=4000)
parser.add_argument("--channels", type=int, help="Number of channels (int)", default=1)
parser.add_argument("--blocksize", type=int, help="Blocksize at which you want to process the sounddata", default=256)
parser.add_argument("--plot", type=bool, help="Do you want to plot the frequency intesities", default=False)


args = parser.parse_args()
device_idx = args.d

sd.default.samplerate = args.samplerate
sd.default.channels = args.channels
sd.default.blocksize = args.blocksize
sd.default.dtype = np.int16 

audio_queue = queue.Queue()
ap = AudioProcessor(blocksize=args.blocksize, samplerate=args.samplerate)

fig, ax = plt.subplots()
line, = ax.plot(ap.x_fft, np.zeros(len(ap.x_fft)))
vline = ax.axvline(x=0, color='r', linestyle='--')

ax.set_ylim(0, 100000) # Adjust based on expected volume
ax.set_title("Live FFT")
ax.set_xlabel("Frequency (Hz)")

def callback(indata, frames, time, status):
    audio_queue.put(indata.copy()) # Puts data into queue

stream = sd.InputStream(device=device_idx, channels=1, callback=callback)
with stream:
    print('listening')
    try:
        while True:
            soundData = None
            ani = FuncAnimation(fig, ap.extract_data, fargs=(line, vline, audio_queue, args.plot), interval=30, blit=True)
            if args.plot:
                plt.show()
                
    except KeyboardInterrupt:
        print("Done processing live data")

    



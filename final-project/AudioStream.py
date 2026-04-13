import sounddevice as sd
import numpy as np
import queue 
import argparse
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation
from audioProcessor import AudioProcessor

##################### Argument Parser #####################
parser = argparse.ArgumentParser()
parser.add_argument("--d", type=int, help="Decide which device you want to record with (int)")
parser.add_argument("--samplerate", type=int, help="Devine the sample rate at which you want to record your audio (int)", default=4000)
parser.add_argument("--channels", type=int, help="Number of channels (int)", default=1)
parser.add_argument("--blocksize", type=int, help="Blocksize at which you want to process the sounddata", default=256)
parser.add_argument("--plot", type=str, help="Do you want to plot the frequency intesities", default="false")

args = parser.parse_args()

assert args.plot in ['true', 'false'], '--plot must be either "true" or "false"'
plot = True
if args.plot == "false":
    plot = False


sd.default.samplerate = args.samplerate
sd.default.channels = args.channels
sd.default.blocksize = args.blocksize
sd.default.dtype = np.int16 

##################### Setup Queue and Audio Processor #####################
audio_queue = queue.Queue()
ap = AudioProcessor(blocksize=args.blocksize, samplerate=args.samplerate)

##################### Initialize Figure #####################
fig, ax = plt.subplots()
line, = ax.plot(ap.x_fft, np.zeros(len(ap.x_fft)))
vline = ax.axvline(x=0, color='r', linestyle='--')

ax.set_ylim(0, 100000) # Adjust based on expected volume
ax.set_title("Live FFT")
ax.set_xlabel("Frequency (Hz)")

##################### Callback Function for Processing Input #####################
def callback(indata, frames, time, status):
    audio_queue.put(indata.copy()) # Puts data into queue

##################### Initialize and Start Stream #####################
stream = sd.InputStream(device=args.d, channels=1, callback=callback)
with stream:
    print('listening')
    print(plot)
    try:
        while True:
            if plot:
                soundData = None
                ani = FuncAnimation(fig, ap.update_plot, fargs=(line, vline, audio_queue), interval=30, blit=True)
                plt.show()
            else:
                soundData = None
                while not audio_queue.empty():
                    soundData = audio_queue.get()
                    hanningFft, mfcc, pitch, amplitude = ap.CalcMFCC(soundData=soundData)
                    print(pitch, amplitude)
                
    except KeyboardInterrupt:
        print("Done processing live data")

    



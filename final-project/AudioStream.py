# I hate tensorflow warnings 
import sounddevice as sd
import numpy as np
<<<<<<< HEAD
import queue 
import argparse
import matplotlib.pyplot as plt
import torch
import os
=======
>>>>>>> 2e14eefbc6f316ec87e2ec2833358a3d78d6422e

sd.default.samplerate = 16_000
sd.default.channels = 1
sd.default.blocksize = 256
sd.default.dtype = np.int16

<<<<<<< HEAD
##################### Argument Parser #####################
parser = argparse.ArgumentParser()
parser.add_argument("--d", type=int, help="Decide which device you want to record with (int)")
parser.add_argument("--samplerate", type=int, help="Devine the sample rate at which you want to record your audio (int)", default=16_000)
parser.add_argument("--channels", type=int, help="Number of channels (int)", default=1)
parser.add_argument("--blocksize", type=int, help="Blocksize at which you want to process the sounddata", default=256)
parser.add_argument("--duration", type=float, help="How long should the audio chunks be", default=1)
parser.add_argument("--plot", type=str, help="Do you want to plot the frequency intesities", default="false")
parser.add_argument("--callback_time", type=float, help="This parameter determines how much time passes before we process newly incoming data using the callback function", default=0.25)

args = parser.parse_args()

assert args.plot in ['true', 'false'], '--plot must be either "true" or "false"'
plot = True
if args.plot == "false":
    plot = False

sd.default.samplerate = args.samplerate
sd.default.channels = args.channels
sd.default.blocksize = int(args.samplerate * args.callback_time)
sd.default.dtype = np.int16 

##################### Setup Queue and Audio Processor #####################
audio_queue = queue.Queue()
ap = AudioProcessor(
                    samplerate=args.samplerate, 
                    chunk_duration=args.callback_time, 
                    n_fft=1024, 
                    n_mels=13
                    )

##################### Load Model #####################
model = torch.load(os.path.join("final-project","models", "cnn_model"), weights_only=False)
model.eval()
CLASSES = {0 : "Whistle", 1: "harmonica", 2: "silence", 3: "clap"}

##################### Initialize Figure #####################
fig, ax = plt.subplots()
line, = ax.plot(ap.x_fft, np.zeros(len(ap.x_fft)))
vline = ax.axvline(x=0, color='r', linestyle='--')

ax.set_ylim(0, 100000) # Adjust based on expected volume
ax.set_title("Live FFT")
ax.set_xlabel("Frequency (Hz)")

##################### Callback Function for Processing Input #####################
def callback(indata, frames, time, status):
    if status:
        print(status)
    audio_queue.put(indata.copy()) # Puts data into queue

##################### Initialize and Start Stream #####################
stream = sd.InputStream(device=args.d, channels=1, callback=callback)
sound = None
with stream:
    print('listening')
    try:
        if plot:
            # FuncAnimation is blocking via plt.show() — set up once, outside the loop
            ani = FuncAnimation(
                fig, ap.update_plot,
                fargs=(line, vline, audio_queue),
                interval=30,
                blit=True
            )
            plt.show() 
        else:
            while True:
                try:
                    # Block until data arrives (up to 0.5s), avoiding busy-wait
                    chunk = audio_queue.get(timeout=0.5)
                except queue.Empty:
                    continue  # No data yet, loop back and wait

                ap.update_window(chunk)
                mfcc = torch.tensor(ap.CalcMFCC(soundData=ap.window, hop=512), dtype=torch.float32)[None, ...]
                with torch.no_grad():
                    m = model(mfcc)
                    pred = torch.argmax(model(mfcc)).item()
                    s = CLASSES[pred]

                    # if s != sound:
                    print(f"Detecting {m}")
                        # sound = s

                
    except KeyboardInterrupt:
        print("Done processing live data")
=======
duration = 5 
>>>>>>> 2e14eefbc6f316ec87e2ec2833358a3d78d6422e

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    outdata[:] = indata
    print(len(indata))
    
with sd.Stream(callback=callback):
    sd.sleep(int(duration * 1000))


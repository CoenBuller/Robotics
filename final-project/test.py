import numpy as np
import sounddevice as sd
import queue 

audio_queue = queue.Queue()

def callback(indata, frames, time, status):
    audio_queue.put(indata.copy()) # Puts data into queue

with sd.InputStream(device=1, channels=1, callback=callback):
    print("Processing the inputdata")
    try:
        while True:
            data = audio_queue.get() # Extract data

            if np.max(data) > 0.5:
                print("Very loud noise detected, could execute action here")
                
    except KeyboardInterrupt:
        print("Done processing live data")
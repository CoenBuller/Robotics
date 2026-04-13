import sounddevice as sd
import numpy as np

sd.default.samplerate = 16_000
sd.default.channels = 1
sd.default.blocksize = 256
sd.default.dtype = np.int16

duration = 5 

def callback(indata, outdata, frames, time, status):
    if status:
        print(status)
    outdata[:] = indata
    print(len(indata))
    
with sd.InputStream(callback=callback):
    sd.sleep(int(duration * 1000))


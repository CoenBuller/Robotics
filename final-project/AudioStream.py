import sounddevice as sd
import numpy as np
import queue 
import librosa as lb

from MFCC import CalcMFCC

sd.default.samplerate = 4_000
sd.default.channels = 4
sd.default.blocksize = 256
sd.default.dtype = np.int16 

audio_queue = queue.Queue()
hanningWindow = np.hanning(sd.default.blocksize)
melFilters = lb.filters.mel(sr=sd.default.samplerate, n_fft=sd.default.blocksize*2-1, n_mels=13)
n_fft_internal = sd.default.blocksize * 2 - 1
x_fft = np.fft.rfftfreq(n_fft_internal, 1/sd.default.samplerate)

def callback(indata, frames, time, status):
    audio_queue.put(indata.copy()) # Puts data into queue

stream = sd.InputStream(device=1, channels=1, callback=callback)
with stream:
    print('listening')
    try:
        while True:
            soundData = None
            while not audio_queue.empty():
                soundData = audio_queue.get()

                mfcc, pitch = CalcMFCC(soundData=soundData,
                                       hanningWindow=hanningWindow,
                                       melFilters=melFilters,
                                       x_fft=x_fft)
                
                print(pitch)
                
    except KeyboardInterrupt:
        print("Done processing live data")

    



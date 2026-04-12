import sounddevice as sd
import numpy as np

CHUNK = 256
RATE = 16_000
LEN = 10 

p = pa.PyAudio()

stream = p.open(
    format=pa.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK
)

player = p.open(
    format=pa.paInt16, channels=1, rate=RATE, output=True, frames_per_buffer=CHUNK
)

for i in range(int(LEN * RATE / CHUNK)):  # go for a LEN seconds
    data = np.frombuffer(stream.read(CHUNK), dtype=np.int16)
    
    
stream.stop_stream()
stream.close()
p.terminate()



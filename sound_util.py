import sounddevice as sd
import scipy.io.wavfile as wav

def record_sample(filename, duration=1.0, fs=44100):
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    wav.write(filename, fs, recording)

def get_stream(callback, blocksize=11025, fs=44100):
    return sd.InputStream(channels=1, samplerate=fs, callback=callback, blocksize=blocksize)
import numpy as np
import scipy as sp
import librosa as lb


class AudioProcessor:
    def __init__(self, samplerate=4000, window_duration=1, chunk_duration=0.25, n_fft=4096, n_mels=13):
        self.samplerate = samplerate
        self.chunk_size = int(samplerate * chunk_duration)  # 1000 samples @ 4kHz
        self.n_fft = n_fft                                  # Power of 2 → fast FFT
        self.window_size = int(samplerate * window_duration)
        self.n_mels = n_mels

        self.window = np.zeros(self.window_size)

        self.pitch = 0.0
        self.amp = 0.0
        self.note = None
        self.octave = None

        # --- Precomputed constants ---
        # Hanning sized to chunk, applied to time-domain signal (before FFT)
        self.hanningWindow = np.hanning(self.window_size)

        # Mel filterbank expects n_fft//2 + 1 bins (rfft output size)
        self.melFilters = lb.filters.mel(sr=samplerate, n_fft=n_fft, n_mels=n_mels)

        # Frequency axis for rfft output
        self.x_fft = np.fft.rfftfreq(n_fft, 1.0 / samplerate)

        self.notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        self.len_notes = len(self.notes)

    
    def update_window(self, frames):
        frames = frames.flatten()
        n = len(frames)
        np.roll(self.window, -n)  # In-place roll, no allocation
        self.window[-n:] = frames

    def freq_to_note(self, freq):

        if freq == 0:
            return self.notes[0], 0
        note_number = 12 * np.log2(freq / 440) + 49  
        note_number = round(note_number)
            
        note = (note_number - 1 ) % self.len_notes
        note = self.notes[note]
        
        octave = (note_number + 8 ) // self.len_notes
        
        return note, octave

    def CalcMFCC(self, soundData: np.ndarray, hop: int = 512) -> np.ndarray:

        sd = soundData.flatten()

        # Avoid spectral leakage by using a hanning window
        windowed = sd * self.hanningWindow     
        fft = np.abs(np.fft.rfft(windowed, n=self.n_fft))

        # Determine the peak frequence for pitch detection
        argmax = np.argmax(fft)
        pitch = self.x_fft[argmax]
        amplitude = fft[argmax]
        self.note, self.octave = self.freq_to_note(pitch)
        self.pitch, self.amp = pitch, amplitude

        # MFCC extraction from the sound data
        mfcc = lb.feature.mfcc(y=sd, sr=self.samplerate, n_mfcc=self.n_mels, hop_length=hop, norm='ortho')


        return mfcc

    def update_plot(self, frame, line, vline, audio_queue):
        soundData = None
        while not audio_queue.empty():
            soundData = audio_queue.get_nowait()

        if soundData is not None:
            fft, mfcc, pitch, amp = self.CalcMFCC(soundData)
            print(f"{self.octave} | {self.note} | {amp:.2f}")
            line.set_ydata(fft)
            vline.set_xdata([pitch, pitch])

        return line, vline

    def other_function(self):
        # This function can access the latest data whenever it wants
        print(f"Current Pitch: {self.pitch}, Amplitude: {self.amp}")

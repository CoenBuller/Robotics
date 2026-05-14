import librosa as lb
import numpy as np
import torch

from torchaudio import transforms as T

class AudioProcessor:
    def __init__(self, samplerate=15_872, window_duration=1, chunk_duration=0.25, n_fft=2048, n_mels=62):
        self.samplerate = samplerate
        self.chunk_size = int(samplerate * chunk_duration)  
        self.n_fft = n_fft                                  
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

        self.mel_transform = T.MelSpectrogram(
                                        sample_rate=samplerate,
                                        n_fft=n_fft,
                                        hop_length=512,
                                        n_mels=n_mels,
                                        )
        self.amplitude_to_db = T.AmplitudeToDB(top_db=90)

    
    def update_window(self, frames):
        frames = frames.flatten()
        n = len(frames)
        self.window = np.roll(self.window, -n)  # np.roll is NOT in-place; result must be assigned
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
    
    def _onset_strength(self, log_mel: np.ndarray) -> np.ndarray:
        # Positive first-order difference across time axis
        diff = np.diff(log_mel, axis=1)              # (n_mels, n_frames-1)
        onset = np.mean(np.maximum(0, diff), axis=0) # (n_frames-1,)
        
        # Pad to match mel frame count
        onset = np.pad(onset, (1, 0))                # (n_frames,)
        return onset / (onset.max() + 1e-9)

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
        mfcc = lb.feature.mfcc(y=sd, sr=self.samplerate, n_mfcc=self.n_mels, hop_length=hop)   # shape: (n_mels, n_frames) 

        # onset = self._onset_strength(log_mel)
        # onset_2d = np.tile(onset, (self.n_mels, 1))

        # result = np.stack([log_mel, onset_2d], axis=0)                                       # (2, n_mels, n_frames)
  

        return mfcc.astype(np.float32)
    


    def other_function(self):
        # This function can access the latest data whenever it wants
        print(f"Current Pitch: {self.pitch}, Amplitude: {self.amp}")


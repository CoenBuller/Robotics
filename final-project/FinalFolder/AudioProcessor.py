import numpy as np

from PitchExtraction import FastMelSpec
from scipy.signal import stft, istft

class AudioProcessor:
    def __init__(self, samplerate=15_872, window_duration=1, chunk_duration=0.25, n_fft=2048, n_mels=62, hop=512):
        self.samplerate = samplerate
        self.chunk_size = int(samplerate * chunk_duration)  
        self.n_fft = n_fft                                  
        self.window_size = int(samplerate * window_duration)
        self.n_mels = n_mels
        self.hop = hop

        self.window = np.zeros(self.window_size)

        self.pitch = 0.0
        self.amp = 0.0
        self.note = None
        self.octave = None

        self.car_noise = np.fft.rfft(np.abs(np.load("average_motor_noise.npy")))

        self.notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        self.len_notes = len(self.notes)

        self.mel_transform = FastMelSpec(
                                         sr=samplerate,
                                         n_fft=n_fft,
                                         hop=hop
                                         )
        


    def spectral_subtraction(self, audio, n_fft=2048):

        audio_fft = np.fft.rfft(audio)

        magnitude = np.abs(audio_fft)
        phase = np.angle(audio_fft)

        cleaned_magnitude = magnitude - self.car_noise
        cleaned_magnitude = np.maximum(cleaned_magnitude, 0)

        cleaned_fft = cleaned_magnitude * np.exp(1j * phase)

        cleaned_audio = np.fft.irfft(cleaned_fft)

        return cleaned_audio

    
    def update_window(self, frames):
        frames = frames.flatten()
        n = len(frames)
        self.window = np.roll(self.window, -n)  # np.roll is NOT in-place; result must be assigned
        self.window[-n:] = frames
        self.audio = np.concatenate([self.audio, frames])

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

    def CalcMel(self, soundData: np.ndarray) -> np.ndarray:

        sd = soundData.flatten()

        # mel spectogram extraction from the sound data
        mel_spectogram, freq,  amplitude= self.mel_transform(audio=sd)   # shape: (n_mels, n_frames) 
        
        # Store note and octave
        self.note, self.octave = self.freq_to_note(freq)
        self.pitch, self.amp = freq, amplitude

        return mel_spectogram.astype(np.float32)
    


    def other_function(self):
        # This function can access the latest data whenever it wants
        print(f"Current Pitch: {self.pitch}, Amplitude: {self.amp}")


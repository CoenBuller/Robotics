import numpy as np
import scipy as sp
import librosa as lb


class AudioProcessor:
    def __init__(self, blocksize=256, samplerate=4000):
        self.pitch = 0
        self.amp = 0
        self.mfcc = None
        self.note = None
        self.octave = None

        self.hanningWindow = np.hanning(blocksize)

        self.n_fft_internal = blocksize * 2 - 1
        self.melFilters = lb.filters.mel(sr=samplerate, n_fft=self.n_fft_internal, n_mels=13)
        self.x_fft = np.fft.rfftfreq(self.n_fft_internal, 1/samplerate)

        self.notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
        self.len_notes = len(self.notes)

    def freq_to_note(self, freq):

        if freq == 0:
            return self.notes[0], 0
        note_number = 12 * np.log2(freq / 440) + 49  
        note_number = round(note_number)
            
        note = (note_number - 1 ) % self.len_notes
        note = self.notes[note]
        
        octave = (note_number + 8 ) // self.len_notes
        
        return note, octave

    def CalcMFCC(self, soundData: np.ndarray) -> tuple[np.ndarray,np.ndarray, int, int]:
        fft = np.abs(np.fft.rfft(soundData.flatten(), n=len(self.hanningWindow)*2 - 1))

        argmax_fft = np.argmax(fft)
        pitch = self.x_fft[argmax_fft] # Calculates the pitch
        self.note, self.octave = self.freq_to_note(pitch)
        amplitude = fft[argmax_fft]

        # Calculates the mfcc 
        hanningFft = fft * self.hanningWindow
        melFft = np.dot(self.melFilters, hanningFft)
        dctFft = sp.fftpack.dct(melFft, type=2)

        return hanningFft, dctFft, pitch, amplitude

    def update_plot(self, frame, line, vline, audio_queue):
        soundData = None
        while not audio_queue.empty():
            soundData = audio_queue.get()

        if soundData is not None:
            # Process data
            fft, mfcc, pitch, amp = self.CalcMFCC(soundData)
            
            # Store values for other functions to use
            self.mfcc = mfcc
            self.pitch = pitch
            self.amp = amp

            line.set_ydata(fft)
            vline.set_xdata([pitch, pitch])
            
        return line, vline

    def other_function(self):
        # This function can access the latest data whenever it wants
        print(f"Current Pitch: {self.pitch}, Amplitude: {self.amp}")

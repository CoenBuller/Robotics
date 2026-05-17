# I hate tensorflow warnings -> Using tflite_runtime to keep it clean and lightweight
import sounddevice as sd
import numpy as np
import queue 
import argparse
import os
import tensorflow as tf

from AudioProcessor import AudioProcessor
from time import time
from motor_controls import MotorControl

sd.default.samplerate = 15_872
sd.default.channels = 1
sd.default.blocksize = 256
sd.default.dtype = np.int16

##################### Argument Parser #####################
parser = argparse.ArgumentParser()
parser.add_argument("--d", type=int, help="Decide which device you want to record with (int)")
parser.add_argument("--samplerate", type=int, help="Devine the sample rate at which you want to record your audio (int)", default=15_872)
parser.add_argument("--channels", type=int, help="Number of channels (int)")
parser.add_argument("--blocksize", type=int, help="Blocksize at which you want to process the sounddata", default=512)
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
                    n_fft=512, 
                    n_mels=62,
                    hop=512
                    )


##################### Load TFLite Model #####################
# Appended .tflite extension to the path
model_path = os.path.join("final-project", "FinalFolder", "cnn_model2F_float32.tflite")
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Get model input and output structural details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


CLASSES = {0 : "clap", 1: "harmonica", 2: "silence", 3: "whistle"}

##################### Motor Control #####################
car = MotorControl(
                   power_forward=20,
                   power_backward=20,
                   power_steer=5,
                   high_notes=set(['D#', 'E', 'F', 'F#', 'G', 'G#'])
                   )

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
        while True:
            try:
                # Block until data arrives (up to 0.5s), avoiding busy-wait
                chunk = audio_queue.get(timeout=0.5)
            except queue.Empty:
                continue  # No data yet, loop back and wait

            time0 = time()
            ap.update_window(chunk)
            
            if not car.forward and not car.backward and not car.turning:
                audio = ap.window
            else:
                audio = ap.spectral_subtraction(ap.window)
            
            # 1. Calculate raw mel spectrogram matrix using NumPy
            mel_data = ap.CalcMel(ap.window)
            
            # 2. Force conversion to target data type and reshape to exactly (1, 1, 62, 32)
            mel_tensor = np.array(mel_data, dtype=input_details[0]['dtype'])
            mel_tensor = np.reshape(mel_tensor, (1, 62, 32, 1))
            
            # 3. Set the input tensor data 
            interpreter.set_tensor(input_details[0]['index'], mel_tensor)
            
            # 4. Invoke TFLite Interpreter inference
            interpreter.invoke()
            
            # 5. Extract output tensor and strip empty dimensions
            m = interpreter.get_tensor(output_details[0]['index']).squeeze()
            
            time1 = time() - time0
            
            pred = np.argmax(m)
            s = CLASSES[int(pred)]

            move = car(pred, ap.note)

            # if s != sound:
            if s != "silence":
                print(f"Detecting {s} | pitch : {str(ap.octave)+str(ap.note)} | inference time: {time1} | probability: {m[pred]} | move: {move}")

    except KeyboardInterrupt:
        print("Done processing live data")
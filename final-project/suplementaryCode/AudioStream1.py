# I hate tensorflow warnings 
import sounddevice as sd
import numpy as np
import queue 
import argparse
import torch
import os
from time import time

import onnxruntime as ort
from AudioProcessPipeline import AudioProcessor
from PitchExtraction import FastMFCC

sd.default.samplerate = 15_872
sd.default.channels = 1
sd.default.blocksize = 256
sd.default.dtype = np.int16

##################### Argument Parser #####################
parser = argparse.ArgumentParser()
parser.add_argument("--d", type=int, help="Decide which device you want to record with (int)")
parser.add_argument("--samplerate", type=int, help="Devine the sample rate at which you want to record your audio (int)", default=15_872)
parser.add_argument("--channels", type=int, help="Number of channels (int)")
parser.add_argument("--blocksize", type=int, help="Blocksize at which you want to process the sounddata", default=256)
parser.add_argument("--duration", type=float, help="How long should the audio chunks be", default=1)
parser.add_argument("--plot", type=str, help="Do you want to plot the frequency intesities", default="false")
parser.add_argument("--callback_time", type=float, help="This parameter determines how much time passes before we process newly incoming data using the callback function", default=0.25)
parser.add_argument("--use_fast", type=str, default="compare", help="'librosa' = original only, 'fast' = FastMFCC only, 'compare' = run both")

args = parser.parse_args()

assert args.plot in ['true', 'false'], '--plot must be either "true" or "false"'
assert args.use_fast in ['librosa', 'fast', 'compare'], '--use_fast must be librosa | fast | compare'
plot = True
if args.plot == "false":
    plot = False

sd.default.samplerate = args.samplerate
sd.default.channels = args.channels
sd.default.blocksize = int(args.samplerate * args.callback_time)
sd.default.dtype = np.int16 


##################### Setup Queue and Audio Processor #####################
audio_queue = queue.Queue()
ap = AudioProcessor(samplerate=args.samplerate,chunk_duration=args.callback_time, n_fft=2048, n_mels=62)
fast = FastMFCC(sr=args.samplerate, n_fft=2048, hop=512, n_mels=128, n_mfcc=62)

##################### Load Model #####################
model_path = os.path.join("models", "onnx_cnn_model", "cnn_model.onnx")
# model = torch.load(model_path, weights_only=False)
# model.eval()
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

CLASSES = {0 : "silence", 1: "harmonica", 2: "clap", 3: "whistle"}


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

            ap.update_window(chunk)

            # Original librosa path
            t0 = time()
            mfcc_lib = ap.CalcMFCC(soundData=ap.window, hop=512)
            t_lib = (time() - t0) * 1000

            # FastMFCC path
            t0 = time()
            mfcc_fast, pitch_per_frame, amp_per_frame = fast(ap.window)
            t_fast = (time() - t0) * 1000

            # Pick which one feeds the model
            mfcc = mfcc_fast if args.use_fast == 'fast' else mfcc_lib
            mfcc = mfcc[None, None, ...].astype(np.float32)

            m = session.run([output_name], {input_name: mfcc})
            pred = np.argmax(m[0], axis=1)
            s = CLASSES[int(pred[0])]

            if args.use_fast == 'compare':
                mfcc_f = mfcc_fast[None, None, ...].astype(np.float32)
                m_f = session.run([output_name], {input_name: mfcc_f})
                pred_f = int(np.argmax(m_f[0], axis=1)[0])
                s_f = CLASSES[pred_f]
                agree = (int(pred[0]) == pred_f)
                max_diff = float(np.abs(mfcc_lib - mfcc_fast).max())
                tag = "OK " if agree else "!! "
                print(f"{tag}lib={s:<10} fast={s_f:<10} | "
                    f"t_lib={t_lib:5.2f}ms t_fast={t_fast:5.2f}ms ({t_lib/max(t_fast,1e-6):4.1f}x) | "
                    f"max|Δ|={max_diff:6.3f} | pitch≈{pitch_per_frame.mean():6.1f}Hz")
            else:
                print(f"Detecting {s} | pitch : {str(ap.octave)+str(ap.note)} | inference time: {t_lib if args.use_fast=='librosa' else t_fast:.2f}ms")
    except KeyboardInterrupt:
        print("Done processing live data")
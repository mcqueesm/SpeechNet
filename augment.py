import numpy as np
import random
import itertools
import librosa
import IPython.display as ipd
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft
from glob import glob
import os


DIR1 = '/Users/jonifinal/*/*.wav'
DIR2 = '/Users/jonifinal/motorcycle/motorcycle'

def white_noise(data):
    wn = np.random.randn(len(data))
    data_wn = data + 0.005*wn
    return data_wn

def white_file():
    wn = np.random.randn(16000)
    return wn

def data_roll(data):
    data_roll = np.roll(data, -1600)
    return data_roll

def stretch(data, rate=1):
    input_length = 16000
    data = librosa.effects.time_stretch(data, rate)
    if len(data)>input_length:
        data = data[:input_length]
    else:
        data = np.pad(data, (0, max(0, input_length - len(data))), "constant")

    return data

wav_files = [os.path.abspath(x) for x in glob(DIR1)]

minus_wav = [x[:-4] for x in wav_files]

'''
for x in range(1, 601):
    wavfile.write('whitenoise' + str(x) + '.wav', 16000, white_file())
    

count = 1
for f in wav_files:
    y, sr = librosa.load(f, sr = 16000, mono=True)
    y = y * 32767 / max(0.01, np.max(np.abs(y)))
    wavfile.write(DIR2 + str(count) + '.wav', sr, y.astype(np.int16))
    count += 1
'''

count = 1
for i, f in enumerate(wav_files):
    sr, wav = wavfile.read(f)
    dr = data_roll(wav)
    wavfile.write(minus_wav[i] + '_bw_roll' + '.wav', sr, dr)
    count += 1
    
'''
for x in range (1, 61):
    sr, wav = wavfile.read(dir + str(x) + '.wav')
    for y in range(1, 31):

        
        roll1 = np.roll(wav, 100*y)
        roll2 = np.roll(wav, -100*y)
        wavfile.write(outdir + '_rf' + str(x) + '_' + str(y) + '.wav', sr, roll1)
        wavfile.write(outdir + '_rb' + str(x) + '_' + str(y) + '.wav', sr, roll2)
'''
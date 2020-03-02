from sys import byteorder
from array import array
from struct import pack
from scipy.signal import stft
from keras.models import load_model

import pyaudio
import wave
import numpy as np

import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile

from glob import glob


import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

THRESHOLD = 750
CHUNK_SIZE = 1024
FORMAT = pyaudio.paInt16
RATE = 16000

def is_silent(snd_data):
    "Returns 'True' if below the 'silent' threshold"
    return max(snd_data) < THRESHOLD

def normalize(snd_data):
    "Average the volume out"
    MAXIMUM = 16384
    times = float(MAXIMUM)/max(abs(i) for i in snd_data)

    r = array('h')
    for i in snd_data:
        r.append(int(i*times))
    return r

def trim(snd_data):
    "Trim the blank spots at the start and end"
    def _trim(snd_data):
        snd_started = False
        r = array('h')

        for i in snd_data:
            if not snd_started and abs(i)>THRESHOLD:
                snd_started = True
                r.append(i)

            elif snd_started:
                r.append(i)
        return r

    # Trim to the left
    snd_data = _trim(snd_data)

    # Trim to the right
    snd_data.reverse()
    snd_data = _trim(snd_data)
    snd_data.reverse()
    return snd_data

def add_silence(snd_data, seconds):
    "Add silence to the start and end of 'snd_data' of length 'seconds' (float)"
    r = array('h', [0 for i in range(int(seconds*RATE))])
    r.extend(snd_data)
    r.extend([0 for i in range(int(seconds*RATE))])
    return r

def record():
    """
    Record a word or words from the microphone and 
    return the data as an array of signed shorts.

    Normalizes the audio, trims silence from the 
    start and end, and pads with 0.5 seconds of 
    blank sound to make sure VLC et al can play 
    it without getting chopped off.
    """
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=1, rate=RATE,
        input=True, output=True,
        frames_per_buffer=CHUNK_SIZE)

    num_silent = 0
    snd_started = False

    r = array('h')

    while 1:
        # little endian, signed short
        snd_data = array('h', stream.read(CHUNK_SIZE))
        if byteorder == 'big':
            snd_data.byteswap()
        r.extend(snd_data)

        silent = is_silent(snd_data)

        if silent and snd_started:
            num_silent += 1
        elif not silent and not snd_started:
            snd_started = True

        if snd_started and num_silent > 30:
            break

    sample_width = p.get_sample_size(FORMAT)
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    
    r = normalize(r)
    r = trim(r)
    r = add_silence(r, 0.5)
    return sample_width, np.asarray(r)

def record_to_file(path):
    "Records from the microphone and outputs the resulting data to 'path'"
    sample_width, data = record()
    data = pack('<' + ('h'*len(data)), *data)

    wf = wave.open(path, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(sample_width)
    wf.setframerate(RATE)
    wf.writeframes(data)
    wf.close()
    
def norm_for_cnn(x):
    wav = x
    wav = wav.astype(np.float32) / np.iinfo(np.int16).max
    return wav

def process_wav_file(x, threshold_freq=5500, eps=1e-10):
        # Read wav file to array
        wav = x
        # Sample rate
        L = 16000
        # If longer then randomly truncate
        if len(wav) > L:
            i = np.random.randint(0, len(wav) - L)
            wav = wav[i:(i+L)]  
        # If shorter then randomly add silence
        elif len(wav) < L:
            rem_len = L - len(wav)
            silence_part = np.random.randint(-100,100,16000).astype(np.float32) / np.iinfo(np.int16).max
            j = np.random.randint(0, rem_len)
            silence_part_left  = silence_part[0:j]
            silence_part_right = silence_part[j:rem_len]
            wav = np.concatenate([silence_part_left, wav, silence_part_right])
        # Create spectrogram using discrete FFT (change basis to frequencies)
        freqs, times, spec = stft(wav, L, nperseg = 400, noverlap = 240, nfft = 512, padded = False, boundary = None)
        # Cut high frequencies
        if threshold_freq is not None:
            spec = spec[freqs <= threshold_freq,:]
            freqs = freqs[freqs <= threshold_freq]
        # Log spectrogram
        amp = np.log(np.abs(spec)+eps)
    
        return wav, np.expand_dims(amp, axis=2)
    
if __name__ == '__main__':
    our_model = load_model('joni_model4.h5')

    
    
    count = 52
    try:
        while True:
            #print("Let's hear some Joni-speak")
            print("Round " + str(count))
            sw, r = record()

            r = norm_for_cnn(r)

            data, r = process_wav_file(r)

            wavfile.write('/Users/jonirecord/train_sign' + str(count) + '.wav', RATE, data)

            count += 1
            '''
            pred = np.argmax(our_model.predict(np.expand_dims(r, axis=0))[0])
            print("Prediction:")
            if pred==0: 
                print('Joni')
            elif pred==1:
                print('Bird')
            elif pred==2:
                print('Nemo')
          '''
    except KeyboardInterrupt:
        pass

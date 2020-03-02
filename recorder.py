import sounddevice as sd
from scipy.io.wavfile import write

fs = 16000  # Sample rate
seconds = 60  # Duration of recording
print('begin')
myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
sd.wait()  # Wait until recording is finished
write('joni_chunk1.wav', fs, myrecording)  # Save as WAV file 
print('finished')
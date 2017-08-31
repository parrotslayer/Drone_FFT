import pyaudio
import numpy as np
from matplotlib import pyplot as plt
import wave

CHUNKSIZE = 4096
FORMAT = pyaudio.paInt32
CHANNELS = 2
RATE = 48000 
RECORD_SECONDS = 0.1
WAVE_OUTPUT_FILENAME = "test.wav"

# initialize portaudio
p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNKSIZE)

frames = [] # A python-list of chunks(numpy.ndarray)
frames_str = []

for _ in range(0, int(RATE / CHUNKSIZE * RECORD_SECONDS)):
    data = stream.read(CHUNKSIZE)
    frames.append(np.fromstring(data, dtype=np.int32))
    frames_str.append(data)
    numpydata = np.fromstring(data, dtype=np.int32)

# close stream
stream.stop_stream()
stream.close()
p.terminate()
print('Done Recording')

#extract channels
left = numpydata[0::2]
right = numpydata[1::2]

# Output as wav file
#wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
#wf.setnchannels(CHANNELS)
#wf.setsampwidth(p.get_sample_size(FORMAT))
#wf.setframerate(RATE)
#wf.writeframes(b''.join(frames_str))
#wf.close()

# FFT
import scipy.signal
from scipy.fftpack import fft
Fs = RATE
T = 1.0/Fs
N = CHUNKSIZE

yf_L = scipy.fftpack.fft(left)
yf_R = scipy.fftpack.fft(right)
xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

freqs = xf[1:]  # dont plot first element to remove DC component
# Create power spectral density 
psd_L = 2.0/N * np.abs(yf_L[0:N/2])[1:]
psd_R = 2.0/N * np.abs(yf_R[0:N/2])[1:]

# Peak Detection
from detect_peaks import detect_peaks

# detect peaks and show the m on a plot
ind_L = detect_peaks(psd_L, mph=5e6, mpd=3, show=True)
ind_R = detect_peaks(psd_R, mph=5e6, mpd=3, show=True)

# Peak Filtering
minF = 950      #min freq Hz
maxF = 1050     #max freq Hz

#check if anything lies within the range
peaks_L_freq = []
peaks_L_amp = []
peaks_R_freq = []
peaks_R_amp = []

k = 0
for i in range(len(ind_L)):
    if freqs[ind_L[i]] > minF and freqs[ind_L[i]] < maxF:
        peaks_L_freq.append(freqs[ind_L[i]])
        peaks_L_amp.append(psd_L[ind_L[i]])

for i in range(len(ind_R)):        
    if freqs[ind_R[i]] > minF and freqs[ind_R[i]] < maxF:
        peaks_R_freq.append(freqs[ind_R[i]])
        peaks_R_amp.append(psd_R[ind_R[i]])
                
print(peaks_L_amp)
print(peaks_R_amp)

peak_diff = peaks_L_amp[0] - peaks_R_amp[0]
print(peak_diff)

if peak_diff > 0:
    #rotate left, CCW
    azi = azi - inc
else:
    #rotate right, CW
    azi = azi + inc
    
        
    



import pyaudio
import numpy as np
from matplotlib import pyplot as plt
import wave
import maestro
import time

# Recording parameters
CHUNKSIZE = 4096
FORMAT = pyaudio.paInt32
CHANNELS = 2
RATE = 48000 
RECORD_SECONDS = 0.1
WAVE_OUTPUT_FILENAME = "test.wav"

#Move the servo to starting location
elev = 7800
azi_max = 6500
azi_min = 5500
azi = azi_min
inc = 10
servo = maestro.Controller()
servo.setTarget(0,azi)  #set servo to move to center position
servo.setTarget(1,elev)     #elevation
servo.close

#minimum height for peak detection
peak_height = 1e6
min_diff = 1e5      #minimum difference between peaks

#arrays for plotting
peaks_L_freq = []
peaks_L_amp = []
peaks_R_freq = []
peaks_R_amp = []

#infinite loop
for azi in range (azi_min, azi_max, inc):
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

    #extract channels
    left = numpydata[0::2]
    right = numpydata[1::2]

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

    # Find which index to use for desired frequency
   # for i in range(0,len(freqs)):
    #    print(i)
     #   print(freqs[i])
    
    # Find PSD at frequency of interest
    index = 724
    peaks_L_amp.append(psd_L[index])
    peaks_R_amp.append(psd_R[index])
    
    #Move the servo
    servo = maestro.Controller()
    servo.setTarget(0,azi)  #set servo to move to center position
    servo.setTarget(1,elev)     #elevation
    servo.close

    print(azi)
    
angle = range(azi_min,azi_max,inc)
plt.plot(angle, peaks_L_amp)
plt.plot(angle, peaks_R_amp)
plt.show(block = False)

plt.figure()
diff = [m - n for m,n in zip(peaks_L_amp,peaks_R_amp)]
plt.plot(angle, diff)
plt.plot((5400, 6600), (0, 0), 'r-')
plt.plot((6000, 6000), (min(diff), max(diff)), 'k-')
plt.show(block = False)

# file-output.py
#f = open('fft_parrot_15_09_2017.txt','w')
#f.write('hello world')
#f.close()

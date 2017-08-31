import pyaudio
import numpy as np
from matplotlib import pyplot as plt
import wave
import maestro
import time

CHUNKSIZE = 4096
FORMAT = pyaudio.paInt32
CHANNELS = 2
RATE = 48000 
RECORD_SECONDS = 0.1
WAVE_OUTPUT_FILENAME = "test.wav"

i = 0
#infinite loop
while(1):
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

    yf = scipy.fftpack.fft(right)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

    freqs = xf[1:]  # dont plot first element to remove DC component
    psd = 2.0/N * np.abs(yf[0:N/2])[1:]


    # Peak Detection
    from detect_peaks import detect_peaks

    # set minimum peak height = 0 and minimum peak distance = 20
    ind = detect_peaks(psd, mph=3e5, mpd=8, show=False)
    #print(ind)
    #print(psd[ind])
    i = i+1
    print(i)

    #begin moving
    azi = 5000 + 100*i
    elev = 5000 + 100*i
    servo = maestro.Controller()
    servo.setTarget(0,azi)  #set servo to move to center position
    servo.setTarget(1,elev)     #elevation
    servo.close

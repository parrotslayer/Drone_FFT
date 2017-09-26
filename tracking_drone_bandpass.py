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
azi = 6000
elev = 7000
azi_max = 8000
azi_min = 4000
inc = 50
servo = maestro.Controller()
servo.setTarget(0,azi)  #set servo to move to center position
servo.setTarget(1,elev)     #elevation
servo.close

#minimum height for peak detection
min_diff = 0      #minimum difference between peaks

loop = 0
#infinite loop
while(loop < 5):
    loop = loop + 1
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
    left = numpydata[1::2]
    right = numpydata[0::2]

    # FFT
    import scipy.signal
    from scipy.fftpack import fft
    Fs = RATE
    T = 1.0/Fs
    N = CHUNKSIZE

    yf_L = scipy.fftpack.fft(left)
    yf_R = scipy.fftpack.fft(right)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

    freqs = xf  # dont plot first element to remove DC component
    # Create power spectral density 
    psd_L = 2.0/N * np.abs(yf_L[0:N/2])
    psd_R = 2.0/N * np.abs(yf_R[0:N/2])

    # Band Pass Filter. Filter out elements outside of this window
    index2freq = 1.0/(2.0*T)/(N/2)
    minF = 8000      #min freq Hz
    maxF = 9000     #max freq Hz
    HpassFreq = 1000     #ignore values below this freq
    Hpass = round(HpassFreq/index2freq)
    index_min = round(minF/index2freq)
    index_max = round(maxF/index2freq)
    NF_L = np.average(psd_L[Hpass:])    # Find the noise floor
    NF_R = np.average(psd_R[Hpass:])
    SNR = 1.2     # gain, not in dB. Using a 3dB SNR
    peak_height_L = NF_L*SNR     #calc the min height for a signal
    peak_height_R = NF_R*SNR     #calc the min height for a signal

    testnum = 6
    plt.figure()
    NF1, = plt.plot(xf[Hpass:],psd_L[Hpass:],label="Raw Data")
    NF2, = plt.plot((0,Fs/2),(peak_height_L, peak_height_L),'r--',label = "Noise Floor")
    NF3, = plt.plot((0,Fs/2),(peak_height_L*SNR, peak_height_L*SNR),'k-',label = "Minimum Signal")
    NF4, = plt.plot( (minF,minF), (0,max(psd_L[10:])), label = "Min Frequency" )
    NF5, = plt.plot( (maxF,maxF), (0,max(psd_L[10:])), label = "Max Frequency" )
    plt.legend(handles = [NF1,NF2,NF3,NF4,NF5])
    plt.show(block=False)
    plt.suptitle('Noise Floor Test {0} Loop {1}'.format(testnum,loop))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.savefig('NF_Test{0}_Loop{1}'.format(testnum,loop))
    
    psd_L[0:index_min] = 0  #Apply band pass filter
    psd_R[0:index_min] = 0
    psd_L[index_max:] = 0
    psd_R[index_max:] = 0

    # Peak Detection
    from detect_peaks import detect_peaks

    # detect peaks and show the m on a plot
    ind_L = detect_peaks(psd_L, mph=peak_height_L, mpd=1, show=False)
    ind_R = detect_peaks(psd_R, mph=peak_height_R, mpd=1, show=False)

    # Peak Filtering

    #check if anything lies within the range
    peaks_L_freq = []
    peaks_L_amp = []
    peaks_R_freq = []
    peaks_R_amp = []

    for i in range(len(ind_L)):
            peaks_L_freq.append(freqs[ind_L[i]])
            peaks_L_amp.append(psd_L[ind_L[i]])

    for i in range(len(ind_R)):        
            peaks_R_freq.append(freqs[ind_R[i]])
            peaks_R_amp.append(psd_R[ind_R[i]])
    
    # only move if detection has occured
    if len(peaks_L_amp) > 0 and len(peaks_R_amp) > 0:
        # find difference in peaks
        peak_diff = peaks_L_amp[0] - peaks_R_amp[0]
        # Print and plot Differentials
        print('Loop {0}'.format(loop))
        print('Frequency = {0} \n Amplitude = {1}'.format(xf[ind_L],psd_L[ind_L]))

        plt.figure()
        left_plot, = plt.plot(freqs, psd_L, label="psd_L")
        plt.hold(True)
        right_plot, = plt.plot(freqs, psd_R, label="psd_R")
        plt.legend(handles = [left_plot, right_plot])
        plt.suptitle('Left and Right Channels Test {0} Loop {1}'.format(testnum,loop))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.savefig('LR_Test{0}_Loop{1}'.format(testnum,loop))
        plt.show(block = False)
       
        #rotate the PTU
        if abs(peak_diff) > min_diff:
            if peak_diff > 0:
                #rotate left, CCW
                azi = azi - inc
                if azi < azi_min:
                    azi = azi_min
            else:
                #rotate right, CW
                azi = azi + inc
                if azi > azi_max:
                    azi = azi_max
                

        #Move the servo
        servo = maestro.Controller()
        servo.setTarget(0,azi)  #set servo to move to center position
        servo.setTarget(1,elev)     #elevation
        servo.close

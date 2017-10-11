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
elev = 8000
azi_max = 8000
azi_min = 4000
inc = 50
servo = maestro.Controller()
servo.setTarget(0,azi)  #set servo to move to center position
servo.setTarget(1,elev)     #elevation
servo.close

loop = 0
#infinite loop
while(loop < 2):
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

    freqs = xf  # dont plot first element to remove DC component
    # Create power spectral density 
    psd_L = 2.0/N * np.abs(yf_L[0:N/2])
    psd_R = 2.0/N * np.abs(yf_R[0:N/2])

    # Band Pass Filter. Filter out elements outside of this window
    index2freq = 1.0/(2.0*T)/(N/2)
    minF = 8000      #min freq Hz
    maxF = 10000     #max freq Hz
    HpassFreq = 1000     #ignore values below this freq when calc noise floor
    Hpass = round(HpassFreq/index2freq)
    index_min = round(minF/index2freq)
    index_max = round(maxF/index2freq)
    NF_L = np.average(psd_L[Hpass:])    # Find the noise floor
    NF_R = np.average(psd_R[Hpass:])

    error = psd_L - psd_R   #error signal
    
    SNR = 2     # gain, not in dB. Using a 3dB SNR
    peak_height_L = NF_L*SNR     #calc the min height for a signal
    peak_height_R = NF_R*SNR     #calc the min height for a signal

    testnum = 9
    plt.figure()
    NF1, = plt.plot(xf[Hpass:],psd_L[Hpass:],label="Raw Data Left")
    NF2, = plt.plot(xf[Hpass:],psd_R[Hpass:],label="Raw Data Right")
    NF3, = plt.plot((0,Fs/2),(peak_height_L, peak_height_L),'r--',label = "Noise Floor")
    NF4, = plt.plot((0,Fs/2),(peak_height_L*SNR, peak_height_L*SNR),'k-',label = "Minimum Signal")
    NF5, = plt.plot( (minF,minF), (0,max(psd_L[Hpass:])), label = "Min Frequency" )
    NF6, = plt.plot( (maxF,maxF), (0,max(psd_L[Hpass:])), label = "Max Frequency" )
    plt.legend(handles = [NF1,NF2,NF3,NF4,NF5,NF6])
    plt.show(block=False)
    plt.suptitle('Noise Floor Test {0} Loop {1}'.format(testnum,loop))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.savefig('NF_Test{0}_Loop{1}'.format(testnum,loop))


    # Determine Variance of Signal and Noise
    # Frequencues where the signal starts and ends
    sig_begin = 8000/index2freq
    sig_end = 10000/index2freq
    std_sig_L = np.std(psd_L[sig_begin:sig_end], dtype=np.float64)
    std_sig_R = np.std(psd_R[sig_begin:sig_end], dtype=np.float64)
    std_noise_L = np.std(psd_L[sig_end:], dtype=np.float64)
    std_noise_R = np.std(psd_L[sig_end:], dtype=np.float64)

    psd_norm_L = np.divide(psd_L, (std_sig_L + 2*std_noise_L) )
    psd_norm_R = np.divide(psd_R, (std_sig_R + 2*std_noise_L) )
    error_norm = psd_norm_L - psd_norm_R

    fig = plt.figure()
    ax1 = fig.add_subplot(211)
    NF1, = ax1.plot(xf[Hpass:],psd_norm_L[Hpass:],label="Raw Data Left")
    NF2, = ax1.plot(xf[Hpass:],psd_norm_R[Hpass:],label="Raw Data Right")
    NF3, = ax1.plot( (minF,minF), (0,max(psd_norm_R[Hpass:])), label = "Min Frequency" )
    NF4, = ax1.plot( (maxF,maxF), (0,max(psd_norm_R[Hpass:])), label = "Max Frequency" )
    plt.legend(handles = [NF1,NF2,NF3,NF4])
    plt.suptitle('Normalised Output Test {0} Loop {1}'.format(testnum,loop))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')

    ax2 = fig.add_subplot(212)
    NF3, = ax2.plot(xf[Hpass:],error_norm[Hpass:],label = "Error Signal")
    NF6, = ax2.plot( (0,Fs/2), (0,0),'k-',label = "Zero Axis" )
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.legend(handles = [NF3])
    plt.show(block=False)
    plt.savefig('Normalised_Test{0}_Loop{1}'.format(testnum,loop))    
    
    #print out maximum values for plotting
    print("Loop Number = {0}".format(loop))
    print(max(psd_norm_L[sig_begin:sig_end]))
    print(max(psd_norm_R[sig_begin:sig_end]))



    


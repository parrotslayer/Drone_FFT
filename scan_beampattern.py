import pyaudio
import numpy as np
from matplotlib import pyplot as plt
import wave
import maestro
import time
import scipy.signal
from scipy.fftpack import fft
from detect_peaks import detect_peaks
from scipy.signal import butter, lfilter, freqz


# Recording parameters
CHUNKSIZE = 4096
FORMAT = pyaudio.paInt32
CHANNELS = 2
RATE = 48000 
RECORD_SECONDS = 0.1

#Move the servo to starting location
elev = 7800
azi_max = 6750 + 125
azi_min = 5250
azi = azi_min
inc = 5
servo = maestro.Controller()
servo.setTarget(0,azi)  #set servo to move to center position
servo.setTarget(1,elev)     #elevation
servo.close

time.sleep(1)

#arrays for plotting
peaks_L_freq = []
peaks_L_amp = []
peaks_R_freq = []
peaks_R_amp = []
peaks_L_avg = []
peaks_R_avg = []

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

    freqs = xf  # dont plot first element to remove DC component
    # Create power spectral density 
    psd_L = 2.0/N * np.abs(yf_L[0:N/2])
    psd_R = 2.0/N * np.abs(yf_R[0:N/2])
    
    # Find PSD at frequency of interest
    index2freq = 1.0/(2.0*T)/(N/2)
    minF = 8000      #min freq Hz
    maxF = 10000     #max freq Hz
    HpassFreq = 100     #ignore values below this freq when calc noise floor
    Hpass = round(HpassFreq/index2freq)
    index_min = round(minF/index2freq)
    index_max = round(maxF/index2freq)
    NF_L = np.average(psd_L[Hpass:])    # Find the noise floor
    NF_R = np.average(psd_R[Hpass:])

    SNR = 1     # gain, not in dB. Using a 3dB SNR
    peak_height = (NF_L+NF_R)/2*SNR     #calc the min height for a signal
    
    psd_L[0:index_min] = 0  #Apply band pass filter
    psd_R[0:index_min] = 0
    psd_L[index_max:] = 0
    psd_R[index_max:] = 0

    #Add the maximum peaks to the list
    peaks_L_avg.append(max(psd_L))
    peaks_R_avg.append(max(psd_R))
    
    #Move the servo
    servo = maestro.Controller()
    servo.setTarget(0,azi)  #set servo to move to center position
    servo.setTarget(1,elev)     #elevation
    servo.close

    #print(azi)

testfreq = 9000
testnum = 15
ff = 25    #ignore X first steps to shift graph

angle = range(azi_min,azi_max,inc)
angle[:] = [x - 6000 for x in angle]
angle = [x*0.06 for x in angle]   #convert angle to degrees
angle[:] = [x - 4 for x in angle] #offset angle
diff = [m - n for m,n in zip(peaks_L_avg,peaks_R_avg)]
sum_sig = [m + n for m,n in zip(peaks_L_avg,peaks_R_avg)]

# Filter requirements.
order = 3
fs = 30       # sample rate, Hz
cutoff = 3.667  # desired cutoff frequency of the filter, Hz

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y


# Get the filter coefficients so we can check its frequency response.
b, a = butter_lowpass(cutoff, fs, order)

diff_filt = butter_lowpass_filter(diff, cutoff, fs, order)
sum_filt = butter_lowpass_filter(sum_sig, cutoff, fs, order)
peaks_L_filt = butter_lowpass_filter(peaks_L_avg, cutoff, fs, order)
peaks_R_filt = butter_lowpass_filter(peaks_R_avg, cutoff, fs, order)
diff_filt2 = [m - n for m,n in zip(peaks_L_filt,peaks_R_filt)]

fig = plt.figure()
ax1 = fig.add_subplot(211) 
NF1, = ax1.plot(angle[ff:], peaks_L_avg[ff:], label = "Left Channel")
NF2, = ax1.plot(angle[ff:], peaks_R_avg[ff:], label = "Right Channel")
plt.legend(handles = [NF1,NF2])
plt.show(block = False)
plt.suptitle('Left and Right Channels Tone Frequency = {1} Hz Test {0}'.format(testnum, testfreq))
plt.xlabel('Angle (Degrees)')
plt.ylabel('Amplitude')

ax2 = fig.add_subplot(212) 
NF11, = ax2.plot(angle[ff:], peaks_L_filt[ff:], label = "Left Filt".format(cutoff))
NF21, = ax2.plot(angle[ff:], peaks_R_filt[ff:], label = "Right Filt".format(cutoff))
plt.legend(handles = [NF11,NF21])
plt.show(block = False)
plt.xlabel('Angle (Degrees)')
plt.ylabel('Amplitude')
plt.savefig('LR_Channels_Test{0}'.format(testnum))

fig = plt.figure()
ax1 = fig.add_subplot(211)
NF3, = ax1.plot(angle[ff:], diff[ff:], label = "Unfiltered")
ax1.plot((-60, 60), (0, 0), 'k-')
ax1.plot((0, 0), (min(diff), max(diff)), 'k-')
plt.legend(handles = [NF3])
plt.xlabel('Angle (Degrees)')
plt.ylabel('Amplitude')

ax2 = fig.add_subplot(212)
NF4, = ax2.plot(angle[ff:], diff_filt[ff:], label = "Fc = {0}".format(cutoff))
ax2.plot((-60, 60), (0, 0), 'k-')
ax2.plot((0, 0), (min(diff), max(diff)), 'k-')
plt.legend(handles = [NF4])
plt.suptitle('Error Signal Tone Frequency = {1} Hz Test {0}'.format(testnum,testfreq,cutoff))
plt.xlabel('Angle (Degrees)')
plt.ylabel('Amplitude')
plt.show(block = False)
plt.savefig('Error_Signal_Test{0}'.format(testnum))

plt.figure()
NF31, = plt.plot(angle[ff:], sum_sig[ff:], label = "Sum Unfiltered")
NF32, = plt.plot(angle[ff:], sum_filt[ff:], label = "Sum Fc = {0}".format(cutoff))
plt.plot((0, 0), (min(sum_sig), max(sum_sig)), 'k-')
plt.legend(handles = [NF31, NF32])
plt.show(block = False)
plt.suptitle('Sum Signal Tone Frequency = {1} Hz Test {0}'.format(testnum,testfreq))
plt.xlabel('Angle (Degrees)')
plt.ylabel('Amplitude')
plt.savefig('Sum_Signal_Test{0}'.format(testnum))

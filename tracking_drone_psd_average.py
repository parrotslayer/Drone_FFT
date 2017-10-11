import pyaudio
import numpy as np
from matplotlib import pyplot as plt
import wave
import maestro
import time
import scipy.signal
from scipy.fftpack import fft
from detect_peaks import detect_peaks

# Recording parameters
CHUNKSIZE = 4096
FORMAT = pyaudio.paInt32
CHANNELS = 2
RATE = 48000 
RECORD_SECONDS = 0.1

#Move the servo to starting location
azi = 6000  # Initial starting values
elev = 7000
azi_max = 8000
azi_min = 4000
servo = maestro.Controller()
servo.setTarget(0,azi)  #set servo to move to center position
servo.setTarget(1,elev)     #elevation
servo.close
# Wait for servo to finish turning
time.sleep(1)
# Begin scanning in CW
inc_scan = 100   # Increment for scanning
scan_CW = 1

while(1):
    # Initialize portaudio
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
    # Close stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Extract channels
    left = numpydata[1::2]
    right = numpydata[0::2]

    # Perform FFT
    Fs = RATE
    T = 1.0/Fs
    N = CHUNKSIZE
    yf_L = scipy.fftpack.fft(left)
    yf_R = scipy.fftpack.fft(right)
    xf = np.linspace(0.0, 1.0/(2.0*T), N/2)
    freqs = xf
    
    # Create power spectral density 
    psd_L = 2.0/N * np.abs(yf_L[0:N/2])
    psd_R = 2.0/N * np.abs(yf_R[0:N/2])

    # Band Pass Filter. Filter out elements outside of this window
    index2freq = 1.0/(2.0*T)/(N/2)
    minF = 8000      #min freq Hz
    maxF = 10000     #max freq Hz
    # Ignore values below this freq when calc noise floor due to 1/f noise
    HpassFreq = 100     
    Hpass = round(HpassFreq/index2freq)
    index_min = round(minF/index2freq)
    index_max = round(maxF/index2freq)
    
    # Normalise the signal from L and R
    sig_end = round(20000/index2freq)
    std_sig_L = np.std(psd_L[index_min:index_max], dtype=np.float64)
    std_sig_R = np.std(psd_R[index_min:index_max], dtype=np.float64)
    std_noise_L = np.std(psd_L[sig_end:], dtype=np.float64)
    std_noise_R = np.std(psd_L[sig_end:], dtype=np.float64)
    
    # Normalising value calculated from the standard deviations
    norm = std_sig_L + std_sig_R + (std_noise_L + std_noise_R)
    psd_L = psd_L/norm
    psd_R = psd_R/ norm

    # Find the noise floor (Normalised)
    # Noise floor is average of L and R channels averaged 
    NF_LR = (np.average(psd_L[Hpass:]) + np.average(psd_R[Hpass:]))/2    
    SNR = 2     # gain, not in dB. Using a +3dB SNR
    peak_height_LR = NF_LR*SNR     #calc the min height for a signal

    # Apply band pass filter
    psd_L[0:index_min] = 0  
    psd_R[0:index_min] = 0
    psd_L[index_max:] = 0
    psd_R[index_max:] = 0

    # Detect peaks and show the m on a plot
    ind_L = detect_peaks(psd_L, mph=peak_height_LR, mpd=1, show=False)
    ind_R = detect_peaks(psd_R, mph=peak_height_LR, mpd=1, show=False)

    # Peak Filtering, see if over N signals counted
    # Minimum number of peaks required for a drone to be confirmed
    min_num_peaks = 3   

    # Move towards target if drone is detected
    # Need both L and R channels to see N peaks (Debatable?)
    if len(ind_L) >= min_num_peaks and len(ind_R) >= min_num_peaks:    
        # Difference between means of L and R channels in the band of interest
        error_sig = np.mean(peaks_L[index_min:index:max]) - np.mean(peaks_R[index_min:index:max])
        # Sum of means of L and R channels in the band of interest
        sum_sig = np.mean(peaks_L[index_min:index:max]) + np.mean(peaks_R[index_min:index:max])
       
        offset_diff = 0.1*NF_LR     #offset for min sum calc
        # Minimum sum value found as a function of the noise floor or HARD CODED
        min_sum = 2*NF_LR + offset_diff
        
        # Only move if sum above a certain value to prevent jittering on boresight                
        if abs(sum_sig) > min_sum:

            # Calculate amount to move by
            inc = 50
            
            # Rotate based on sign of error signal
            if error_sig > 0:
                #rotate CCW, towards Right
                azi = azi - inc
                # Dont rotate past hard limit
                if azi < azi_min:
                    azi = azi_min
            else:
                #rotate CW, towards Left
                azi = azi + inc
                # Dont rotate past hard limit
                if azi > azi_max:
                    azi = azi_max
                    
    # If no drone detected continue rotating the dish to scan for a drone          
    else:
        if scan_CW == 1:
            azi = azi + inc_scan
            # Once reach end, scan other direction
            if azi > azi_max - inc_scan:
                scan_CW = 0
        else:
            azi = azi - inc_scan
            # Once reach end, scan other direction
            if azi < azi_min + inc_scan:
                scan_CW = 1

    # Move the servo
    servo = maestro.Controller()
    servo.setTarget(0,azi)  #set servo to move to center position
    servo.setTarget(1,elev)     #elevation
    servo.close

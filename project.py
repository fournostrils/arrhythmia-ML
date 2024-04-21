import wfdb
from wfdb import processing
import pandas as pd
import numpy as np
from scipy.signal import medfilt, find_peaks, butter, lfilter
import matplotlib.pyplot as plt
import pywt # type: ignore


# Function to save ECG data to CSV
def save_ecg_to_csv(record_name):
    # Read ECG record
    record = wfdb.rdrecord(record_name, pn_dir='mitdb')

    # Extract signal data
    ecg_data = record.p_signal
    lead_names = record.sig_name

    # Calculate time information based on sampling frequency
    sampling_frequency = record.fs  # Sampling frequency in samples per second
    num_samples = len(ecg_data)    # Total number of samples
    time = 1/sampling_frequency * np.arange(0,num_samples)

    # Create a DataFrame to store stuff
    df = pd.DataFrame(columns=['Time']+lead_names)
    df[lead_names] = ecg_data
    df['Time'] = time


################# median filter ###################
    baseline = medfilt(df[lead_names[0]], 71)       #cite the guy for this
    baseline = medfilt(baseline, 215)
    df[lead_names[0]] = df[lead_names[0]]-np.asfarray(baseline)
###################################################

############### bandpass filter ###################
    highcut = 15
    lowcut = .5

    nyq = 0.5 * sampling_frequency
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(5, [low, high], btype='band')
    df[lead_names[0]] = lfilter(b, a, df[lead_names[0]])
####################################################

################# median filter ###################
    baseline = medfilt(df[lead_names[0]], 71)       #cite the guy for this
    baseline = medfilt(baseline, 215)
    df[lead_names[0]] = df[lead_names[0]]-np.asfarray(baseline)
###################################################

################### moving window integrations ######################
    def moving_window(data, window_size):
        window = np.ones(window_size) / window_size
        integrated_signal = np.convolve(data, window, mode='same')
        return integrated_signal
    
    df[lead_names[0]] = moving_window(df[lead_names[0]], 3)
#####################################################################

    # Save DataFrame to CSV file
    filename = f'{record_name}.csv'
    df.to_csv(filename, index=False)
    print(f'Saved {filename}')
    

def feature_extract(record_name):
    
    def threshold_peaks(data, threshold_value=0.5):
        mean_peak_height = np.mean(data)
        threshold = mean_peak_height * threshold_value
        peaks, _ = find_peaks(data, height=threshold)
        return peaks


    data = pd.read_csv(f"C:\\Users\\rigga\\Documents\\BMEN 207\\Honors project\\{record_name}.csv")
    lead = data.columns[1]

    data['gradient'] = np.gradient(data[lead])        #first derivative peak is very reliably in the middle of the increasing part of the r wave
    r_peaks = find_peaks(np.asfarray(data['gradient']),prominence = 0.3*max(data['gradient']), distance = 72, height=0.25*max(data['gradient']))[0]
    print('1st derivative peaks length:' + str(len(r_peaks)))
    peak_dev1 = r_peaks
    print(peak_dev1)
    
    for i in range(2):
        data['gradient'] = np.gradient(data['gradient'])  #third derivative usually finds middle of descending part of r wave
        r_peaks = find_peaks(np.asfarray(data['gradient']),prominence = 0.3*max(data['gradient']), distance = 72, height=0.25*max(data['gradient']))[0]
        print(f'{i+2} derivative r_peaks length= {len(r_peaks)}')
    
    #r_peaks = (peak_dev1 + r_peaks) /2
    

    # Read annotation data
    ann = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')

    # Extract annotation symbols and sample indices
    symbols = ann.symbol
    sample_indices = ann.sample
    #time_points = sample_indices/sampling_frequency
####################### harrison code #################
    check = np.where(np.logical_or(symbols=='~',symbols=='|'))[0]

    sample_indices=np.delete(sample_indices,check, axis=0)
    symbols=np.delete(symbols,check, axis=0)

    #find indices of r wave peaks
    #r_peaks = find_peaks(np.asfarray(data['gradient']),prominence = 0.3*max(data['gradient']), distance = 72, height=0.25*max(data['gradient']))[0]
        #find r-peaks missed
    #r2r_peaks = np.diff(r_peaks)
        #check=np.where(r2r_peaks>400 )[0]
    #r_peaks = np.insert(r_peaks, check+1,(r_peaks[check]+r_peaks[check+1])/2)
    '''
    if len(r_peaks) > len(sample_indices):
        r_peaks = r_peaks[:len(sample_indices)]
    elif len(r_peaks) < len(sample_indices):
        sample_indices = sample_indices[:len(r_peaks)]'''
#######################################################

    #find indices of r wave peaks
    #r_peaks = find_peaks(np.asfarray(data['Both filters']),prominence = 0.3*max(data['Both filters']), distance = 72)[0]

    '''
    for i in range(len(r_peaks)-1):
        if abs(fwd_derivative[i]) < abs(mean_fwd_dev):
            r_peaks.pop(i)
    '''
    output_x = np.array([])
    output_y = np.array([])

    
    for i in range(len(r_peaks)-1):
        #find r-r distance between peaks and store in array
        #ask about r-r array being one shorter than other one and how we should deal with that
        if i == (len(r_peaks)-1):
            break
        else:
            rr_post = r_peaks[i] - r_peaks[i+1]
    '''
        window = []
        if r_peaks[i] > 90 and r_peaks[i]+90 <len(data) and np.isclose(r_peaks[i], sample_indices[i], atol=20):
                window = data[lead][r_peaks[i]-90:r_peaks[i]+90] #window
                ##### calculations #####
                #find locations of wave peaks around r wave
                px, py = np.argmax(window[0:40]), np.max(window[0:40])
                qx,qy = np.argmin(window[75:85]), np.min(window[75:85])
                sx, sy = np.argmin(window[95:105]), np.min(window[95:105])
                tx, ty = np.argmax(window[150:180]), np.max(window[150:180])

                #find distance between r wave and each other wave
                rx,ry = r_peaks[i], data[lead][r_peaks[i]]

                rp_dist, rq_dist, rs_dist, rt_dist = np.sqrt((rx-px)**2 + (ry-py)**2), np.sqrt((rx-qx)**2 + (ry-qy)**2), np.sqrt((rx-sx)**2 + (ry-sy)**2), np.sqrt((rx-tx)**2 + (ry-ty)**2)
                #perhaps make array to store distances; do more research

                #find ratio between heights of r peak and each other peak (p/r)
                rp_ratio = py/ry
                rq_ratio = qy/ry
                rs_ratio = sy/ry
                rt_ratio = ty/ry

                #make array of importnat values for this window
                window_vals = np.array([rr_post, rp_dist, rq_dist, rs_dist, rt_dist, rp_ratio, rq_ratio, rs_ratio, rt_ratio])
                output_x = np.append(output_x, window_vals)
            #output_y = np.append() #tbd
        
    '''
    ''' prob aren't using wavelet transform
    #use wavelet transform to extract features using wavelet module
    db1 = pywt.Wavelet('db1')
    coeffs = pywt.wavedec(data[lead], db1, level=3)
    wavel = coeffs[0]
    '''

    ######################## plots #########################
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))
    
    #plot fo raw data
    axs[0].plot(data['Time'],data[lead])
    axs[0].plot(data['Time'][r_peaks], data[lead][r_peaks], 'x')
    axs[0].set_title('2nd derivative')
    axs[0].grid(True)
    
    #plot of double filtered data
    '''    axs[1].plot(data['Time'],data[lead])
    axs[1].plot(data['Time'][sample_indices], data[lead][sample_indices], 'x')
    axs[1].set_title('annotations')
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()'''
    axs[1].plot(data['Time'],data[lead])
    axs[1].plot(data['Time'][peak_dev1], data[lead][peak_dev1], 'x')
    axs[1].plot(data['Time'][sample_indices], data[lead][sample_indices], 'r.')
    axs[1].set_title('1st derivative')
    axs[1].grid(True)
    plt.tight_layout()
    plt.show()
    ########################################################


    return output_x

# Example usage
#save_ecg_to_csv('103')
feature_extract('103')



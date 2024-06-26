import wfdb
from wfdb import processing
import pandas as pd
import numpy as np
from scipy.signal import medfilt, find_peaks
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

    '''
    # Read annotation data
    ann = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')

    # Extract annotation symbols and sample indices
    symbols = ann.symbol
    sample_indices = ann.sample
    #time_points = sample_indices/sampling_frequency
    '''

    # Create a DataFrame to store stuff
    df = pd.DataFrame(columns=['Time']+lead_names)
    df[lead_names] = ecg_data
    df['Time'] = time
    

    '''# Assign annotation symbols to corresponding sample indices in DataFrame
    for i in range(len(symbols)):
        symbol = symbols[i]  # Get the symbol at index i
        sample_index = sample_indices[i]  # Get the sample index at index i
        # Check if the sample index is within the range of the DataFrame
        if 0 <= sample_index < len(ecg_data):
            # Assign the annotation symbol to the corresponding index in the DataFrame
            df.at[sample_index,'Annotation'] = symbol
    '''

    
    baseline = medfilt(df[lead_names[0]], 71)       #cite the guy for this
    baseline = medfilt(baseline, 215)
    df[lead_names[0]] = df[lead_names[0]]-np.asfarray(baseline)
    
    df[lead_names[0]]=(df[lead_names[0]]-np.mean(df[lead_names[0]]))/np.std(df[lead_names[0]])

    # Save DataFrame to CSV file
    filename = f'{record_name}.csv'
    df.to_csv(filename, index=False)
    print(f'Saved {filename}')
    

def feature_extract(record_name):
    data = pd.read_csv(f"C:\\Users\\rigga\\Documents\\BMEN 207\\Honors project\\{record_name}.csv")
    lead = data.columns[1]

    # Read annotation data
    ann = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')

    # Extract annotation symbols and sample indices
    symbols = ann.symbol
    sample_indices = ann.sample
    #time_points = sample_indices/sampling_frequency

    #find indices of r wave peaks
    r_peaks = find_peaks(np.asfarray(data[lead]),prominence = 0.4*max(data[lead]), distance = 72)[0]

    #find euclidean distance between features
    #generate 180 sample window around r peak
    output_x = np.array([])
    output_y = np.array([])

    
    for i in range(len(r_peaks)-1):
        #find r-r distance between peaks and store in array
        #ask about r-r array being one shorter than other one and how we should deal with that
        if i == (len(r_peaks)-1):
            break
        else:
            rr_post = r_peaks[i] - r_peaks[i+1]
    
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
        
    
    #use wavelet transform to extract features using wavelet module
    db1 = pywt.Wavelet('db1')
    coeffs = pywt.wavedec(data[lead], db1, level=3)
    wavel = coeffs[0]
    
    #plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 5))

    axs[0].plot(data['Time'],data[lead])
    axs[0].plot(data['Time'][r_peaks], data[lead][r_peaks], 'x')
    axs[0].set_title('Normalized and filtered ECG data')
    

    axs[1].plot(np.arange(0,len(wavel)),wavel)
    axs[1].set_title('Wavelet transform ECG data')

    plt.tight_layout()
    plt.show()
    
    return output_x

# Example usage
save_ecg_to_csv('103')
print(feature_extract('103'))

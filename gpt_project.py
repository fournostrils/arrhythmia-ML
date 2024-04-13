import wfdb
import pandas as pd
import numpy as np
from scipy.signal import medfilt, find_peaks
import matplotlib.pyplot as plt


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

    # Read annotation data
    ann = wfdb.rdann(record_name, 'atr', pn_dir='mitdb')

    # Extract annotation symbols and sample indices
    symbols = ann.symbol
    sample_indices = ann.sample
    time_points = sample_indices/sampling_frequency

    # Create an empty DataFrame to store annotations
    df = pd.DataFrame(columns=['Time', 'Annotation']+lead_names)
    df[lead_names] = ecg_data
    df['Time'] = time
    df['Annotation'] = '----'
    
    # Assign annotation symbols to corresponding sample indices in DataFrame
    for i in range(len(symbols)):
        symbol = symbols[i]  # Get the symbol at index i
        sample_index = sample_indices[i]  # Get the sample index at index i
        # Check if the sample index is within the range of the DataFrame
        if 0 <= sample_index < len(ecg_data):
            # Assign the annotation symbol to the corresponding index in the DataFrame
            df.at[sample_index,'Annotation'] = symbol
    
    

    baseline = medfilt(df[lead_names[0]], 71)       #cite the guy for this
    baseline = medfilt(baseline, 215)
    df[lead_names[0]] = df[lead_names[0]]-np.asfarray(baseline)
    
    df[lead_names[0]]=(df[lead_names[0]]-np.mean(df[lead_names[0]]))/np.std(df[lead_names[0]])

    # Save DataFrame to CSV file
    filename = f'{record_name}.csv'
    df.to_csv(filename, index=False)
    print(f'Saved {filename}')

def ecg_plot(record_name):
    data = pd.read_csv(f"C:\\Users\\rigga\\Documents\\BMEN 207\\Honors project\\{record_name}.csv")
    lead = data.columns[2]
    r_peaks = find_peaks(np.asfarray(data[lead]),prominence = 0.4*max(data[lead]), distance = 72)[0]
    #wfdb.processing.gqrs_detect(sig=data['MLII'][0:3000],fs=360)

    #plot
    plt.figure()
    plt.plot(data['Time'],data[lead])
    plt.plot(data['Time'][r_peaks], data[lead][r_peaks], 'x')
    plt.show()

# Example usage
save_ecg_to_csv('103')
ecg_plot('103')

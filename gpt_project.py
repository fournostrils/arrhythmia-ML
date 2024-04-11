import wfdb
import pandas as pd
import numpy as np

# Function to save ECG data to CSV
def save_ecg_to_csv(record_name):
    # Read ECG record
    record = wfdb.rdrecord(record_name, pn_dir='mitdb')

    # Extract signal data
    ecg_data = record.p_signal
    lead_names = record.sig_name
    print(lead_names)
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
    print(len(symbols))
    print(len(sample_indices))
    # Save DataFrame to CSV file
    filename = f'{record_name}.csv'
    df.to_csv(filename, index=False)
    print(f'Saved {filename}')
    print(len(ecg_data))
    print(len(df['Time']))
# Example usage
save_ecg_to_csv('101')

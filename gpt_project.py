import wfdb
import pandas as pd

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
    duration_seconds = num_samples / sampling_frequency  # Duration of signal in seconds
    time = pd.timedelta_range(start='0s', end=pd.Timedelta(seconds=duration_seconds), periods=num_samples)

    # Convert to DataFrame
    df = pd.DataFrame(ecg_data, columns=lead_names)

    # Add time column to DataFrame
    df['Time'] = time

    # Save DataFrame to CSV file
    filename = f'{record_name}.csv'
    df.to_csv(filename, index=False)
    print(f'Saved {filename}')

# List of record names in MIT-BIH Arrhythmia Database
record_names = wfdb.get_record_list('mitdb')

# Save ECG data to CSV for each record
for record_name in record_names:
    save_ecg_to_csv(record_name)
import wfdb
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

record = wfdb.rdrecord('100',pn_dir='mitdb')

ecg_data = record.p_signal
sampling_frequency = record.fs
num_samp = len(ecg_data)
duration = num_samp/sampling_frequency
time = pd.timedelta_range(start='0s', end = pd.Timedelta(seconds=duration), periods = num_samp)

df = pd.DataFrame(ecg_data, columns=[f'ECG_{i}' for i in range(ecg_data.shape[1])])
#df.to_csv('ecg_data.csv', index=False)

df['Time'] = time
plt.figure()
plt.plot(df['Time'],df['ECG_0'])
plt.show()
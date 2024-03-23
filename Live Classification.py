import numpy as np
import pandas as pd
from scipy.signal import spectrogram
from skimage.measure import block_reduce
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from muselsl import record


def GetSpectrogramForEEGData(eeg_data):
    # Constants
    fs = 256  # Sampling rate
    segment_length = 1  # seconds
    frame_length = segment_length * fs

    # Calculate spectrogram for each channel
    spectrograms = []
    for i in range(1, 5):  # Assuming EEG data is in columns 1 to 4
        f, t, Sxx = spectrogram(eeg_data.iloc[:, i], fs=fs, nperseg=frame_length, noverlap=0)
        spectrograms.append(Sxx)

    # Concatenate spectrograms from all channels
    Sxx_combined = np.concatenate(spectrograms, axis=0)

    # Transpose to match CNN input format
    Sxx_combined = np.transpose(Sxx_combined)

    return Sxx_combined


# Load scaler
scaler = StandardScaler()
scaler_params = pd.read_csv("EEG_Spectrogram_Scaler.csv", header=None).values.ravel()
scaler.mean_ = scaler_params[:4]
scaler.scale_ = scaler_params[4:]

# Load the trained model
model = load_model("EEG_Spectrogram_CNN.h5")

# Record EEG data for 5 seconds
record(5, 'output.csv')

# Load the recorded EEG data
eeg_data = pd.read_csv("output.csv")

# Process the EEG data into spectrograms
Sxx = GetSpectrogramForEEGData(eeg_data)

# Scale the data
x_scaled = scaler.transform(Sxx)

# Predict the class
predictions = model.predict(x_scaled)
predicted_class = np.argmax(predictions)

print(f"The predicted class is: {predicted_class}")

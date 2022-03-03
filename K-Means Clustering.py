import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import numpy, scipy, matplotlib.pyplot as plt, sklearn, librosa, IPython.display


FIG_SIZE = (15,10)

filename = "2.wav"

# load audio file with Librosa
signal, sample_rate = librosa.load(filename, sr=22050)


# Load the audio file into an array
x, fs = librosa.load(filename)
print (fs)

# Plotting audio signal
librosa.display.waveplot(x, fs)

# Detect onsets
onset_frames = librosa.onset.onset_detect(x, sr=fs, delta=0.04, wait=4)
onset_times = librosa.frames_to_time(onset_frames, sr=fs)
onset_samples = librosa.frames_to_samples(onset_frames)

# Listen to detected onsets
x_with_beeps = sonify.clicks(onset_times, fs, length=len(x))
IPython.display.Audio(x + x_with_beeps, rate=fs)

# Feature Extraction

# Zero Crossing Rate
def extract_features(x, fs):
    zcr = librosa.zero_crossings(x).sum()
    energy = scipy.linalg.norm(x)
    return [zcr, energy]

frame_sz = fs*0.090
features = numpy.array([extract_features(x[i:i+frame_sz], fs) for i in onset_samples])
print (features.shape)

# Feature Scaling
min_max_scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1, 1))
features_scaled = min_max_scaler.fit_transform(features)
print (features_scaled.shape)
print (features_scaled.min(axis=0))
print (features_scaled.max(axis=0))

# Plotting the features

plt.scatter(features_scaled[:,0], features_scaled[:,1])
plt.xlabel('Zero Crossing Rate (scaled)')
plt.ylabel('Spectral Centroid (scaled)')

# K-Means
model = sklearn.cluster.KMeans(n_clusters=2)
labels = model.fit_predict(features_scaled)
print (labels)

# Plot the results

plt.scatter(features_scaled[labels==0,0], features_scaled[labels==0,1], c='b')
plt.scatter(features_scaled[labels==1,0], features_scaled[labels==1,1], c='r')
plt.xlabel('Zero Crossing Rate (scaled)')
plt.ylabel('Energy (scaled)')
plt.legend(('Class 0', 'Class 1'))

# Onset assigned to Class 0
x_with_beeps = sonify.clicks(onset_times[labels==0], fs, length=len(x))
IPython.display.Audio(x + x_with_beeps, rate=fs)

# Onset assigned to Class 1
x_with_beeps = sonify.clicks(onset_times[labels==1], fs, length=len(x))
IPython.display.Audio(x + x_with_beeps, rate=fs)


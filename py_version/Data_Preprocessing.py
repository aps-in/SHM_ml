import numpy as np

# Graphing libraries
import matplotlib.pyplot as plt


data = np.load('data.npy', allow_pickle = True)
labels = np.load('labels.npy', allow_pickle = True)

print('Number of data points: ',data.shape[0])
print('Number of sensors: ',data.shape[1])
print('Signal length: ',data.shape[2])
print('Classes: ', np.unique(labels))

fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 8))
fig.subplots_adjust(hspace=.35)
((ax1, ax2), (ax3, ax4)) = axs
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
ax1.plot(data[0, 0, :], label = 'Vibration signal')
ax1.set_title('Sensor 1')
ax1.set(ylabel='Amplitude', xlabel='Time')
ax2.plot(data[0, 4, :], label = 'Vibration signal')
ax2.set_title('Sensor 4')
ax2.set(ylabel='Amplitude', xlabel='Time')
ax3.plot(data[0, 8, :], label = 'Vibration signal')
ax3.set_title('Sensor 8')
ax3.set(ylabel='Amplitude', xlabel='Time')
ax4.plot(data[0, 12, :], label = 'Vibration signal')
ax4.set_title('Sensor 12')
ax4.set(ylabel='Amplitude', xlabel='Time')

data = (data - np.min(data, axis = 2, keepdims = True))/(np.max(data, axis = 2, keepdims = True) - \
                                                                    np.min(data, axis = 2, keepdims = True))


fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 8))
fig.subplots_adjust(hspace=.35)
((ax1, ax2), (ax3, ax4)) = axs
plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
ax1.plot(data[0, 0, :], label = 'Vibration signal')
ax1.set_title('Sensor 1')
ax1.set(ylabel='Amplitude', xlabel='Time')
ax2.plot(data[0, 4, :], label = 'Vibration signal')
ax2.set_title('Sensor 4')
ax2.set(ylabel='Amplitude', xlabel='Time')
ax3.plot(data[0, 8, :], label = 'Vibration signal')
ax3.set_title('Sensor 8')
ax3.set(ylabel='Amplitude', xlabel='Time')
ax4.plot(data[0, 12, :], label = 'Vibration signal')
ax4.set_title('Sensor 12')
ax4.set(ylabel='Amplitude', xlabel='Time')


np.save('data_processed.npy', data)
np.save('labels_processed.npy', labels)
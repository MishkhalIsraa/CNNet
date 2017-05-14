import matplotlib
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from scipy import stats
import tensorflow as tf
#% matplotlib inline
#plt.style.use('ggplot')

dataset = pd.read_csv('/Users/israamishkhal/Desktop/train.csv')

def windows(data, size):
   start = 0
   while start < data.count():
yield start, start + size
 start += (size / 2)
def segment_signal(data, window_size=90):
  segments = np.empty((0, window_size, 3))
labels = np.empty((0))
 for (start, end) in windows(data["Acc-sma()"], window_size):
  x = data["x-axis"][start:end]
 y = data["y-axis"][start:end]
  z = data["z-axis"][start:end]
 if (len(dataset["Acc-sma()"][start:end]) == window_size):
            segments = np.vstack([segments, np.dstack([x, y, z])])
            labels = np.append(labels, stats.mode(data["Activity"][start:end])[0][0])
    return segments, labels
segments, labels = segment_signal(dataset)
labels = np.asarray(pd.get_dummies(labels), dtype = np.int8)
reshaped_segments = segments.reshape(len(segments), 1,90, 3)

train_test_split = np.random.rand(len(reshaped_segments)) < 0.70
train_x = reshaped_segments[train_test_split]
train_y = labels[train_test_split]
test_x = reshaped_segments[~train_test_split]
test_y = labels[~train_test_split]



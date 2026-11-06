

import numpy as np
import matplotlib.pyplot as plt
import os

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.models import Sequential
from keras.layers import (
    Input,
    Conv1D,
    MaxPooling1D,
    Flatten,
    Dense
)


# ==========================
# Load Dataset
# ==========================

base_path = os.path.dirname(__file__)

file_path = os.path.join(base_path,
                         "..",
                         "MATLAB",
                         "Dataset.mat")

data = loadmat(file_path)

Pattern = data["Pattern"]
Amplitude = data["Amplitude"]
Real = data["Real"]
Imag = data["Imag"]

N = Amplitude.shape[1]

Amplitude = Amplitude.astype(np.float32)
Real = Real.astype(np.float32)
Imag = Imag.astype(np.float32)

# ==========================
# TensorFlow Array Factor
# ==========================

def array_factor_tf(amplitude, phase):

    # amplitude : (batch_size, N)
    # phase     : (batch_size, N)   (radian)

    N = amplitude.shape[1]

    d = 0.5
    k = 2 * np.pi

    theta = tf.range(-90.0, 91.0, 1.0)
    theta_rad = theta * np.pi / 180.0

    # (N,)
    n = tf.cast(tf.range(N), tf.float32)

    # (181, N)
    phase_shift = (
        k * d *
        tf.expand_dims(tf.sin(theta_rad), axis=1) *
        tf.expand_dims(n, axis=0)
    )

    # (batch,1,N)
    phase = tf.expand_dims(phase, axis=1)

    # (1,181,N)
    phase_shift = tf.expand_dims(phase_shift, axis=0)

    total_phase = phase + phase_shift

    real = tf.expand_dims(amplitude, axis=1) * tf.cos(total_phase)
    imag = tf.expand_dims(amplitude, axis=1) * tf.sin(total_phase)

    AF_real = tf.reduce_sum(real, axis=2)
    AF_imag = tf.reduce_sum(imag, axis=2)

    AF = tf.sqrt(AF_real**2 + AF_imag**2)

    AF = AF / tf.reduce_max(AF, axis=1, keepdims=True)

    AF_dB = 20.0 * tf.math.log(AF + 1e-12) / tf.math.log(10.0)

    return AF_dB


phase_test = tf.atan2(Imag, Real)

pattern_test = array_factor_tf(
    tf.constant(Amplitude, dtype=tf.float32),
    tf.constant(phase_test, dtype=tf.float32)
)

print(pattern_test.shape)

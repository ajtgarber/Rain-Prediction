#TODO: Randomize the order of the entries so the validation set isn't just the first 10,000 entries
# current dataset is 73.47% zero rain values, that's the accuracy we need to beat

import pandas as pd
import numpy as np

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow.keras import layers

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# disable GPU acceleration (need to set up AMD GPU)
tf.config.set_visible_devices([], 'GPU')

train_data = pd.read_csv("enriched_input.csv", low_memory=False, skiprows=1, names=["tmpf", "dwpf", "drct", "sped", "mslp", "tmpf_change", "dwp_dep_change", "mslp_change", "rain"])
train_data = train_data.sample(frac=1).reset_index(drop=True) # randomly shuffle the dataset
print(train_data.head())

# remember that zero values for rain are listed as NaN, and no observations for MSLP are also NaN

rain_features = train_data.copy()
rain_labels   = rain_features.pop("rain")

rain_features = np.array(rain_features)
rain_labels   = np.array(rain_labels)

#reserve 10,000 entries for validation
x_val         = rain_features[-100000:]
y_val         = rain_labels[-100000:]
rain_features = rain_features[:-100000]
rain_labels   = rain_labels[:-100000]

normalize = layers.Normalization()
normalize.adapt(rain_features)

rain_model = tf.keras.Sequential([
    normalize,
    layers.Dense(3, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(2, activation='relu'),
    layers.Dense(1)
])
early_stopping_monitor = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', #really should be val_loss when you actually have a validation dataset
    min_delta=0,
    patience=2,
    verbose=0,
    mode='auto',
    baseline=None,
    restore_best_weights=True
)
rain_model.compile(loss = tf.keras.losses.MeanSquaredError(),
    optimizer = tf.keras.optimizers.Adam(),
    metrics=[tf.keras.metrics.BinaryAccuracy()])

rain_model.fit(rain_features, rain_labels, epochs=20, shuffle=True, callbacks=[early_stopping_monitor], validation_data=(x_val, y_val))
rain_model.save('rain_model')
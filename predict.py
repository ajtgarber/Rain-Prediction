import pandas as pd
import numpy as np
import os
import logging
import time
import datetime

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)
from tensorflow.keras import layers

input_data = np.array([32.2, 27.0, 29, 2.2, 1012, 0.72, 0.72, -1.24])
print("input_data: " + str(input_data))

print()
rain_model = tf.keras.models.load_model("rain_model")

output_prediction = rain_model.predict(input_data)[0][0]
print("Model Prediction: " + str(output_prediction) + " inches")
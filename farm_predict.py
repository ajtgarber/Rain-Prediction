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

import mysql.connector

query = """
SELECT
	curr.tempf,
	curr.dewptf,
	curr.winddir,
	curr.windspeedmph,
	curr.baromin * 33.8638 AS "BAROMIN",
	curr.tempf - AVG(prev_hour.tempf) AS "TEMP CHANGE",
	(curr.tempf - curr.dewptf) - (AVG(prev_hour.tempf) - AVG(prev_hour.dewptf)) AS "DPT DEPRESSION CHANGE",
	(curr.baromin * 33.8638) - AVG(prev_hour.baromin * 33.8638) AS "PRESSURE CHANGE"
FROM
	weather.farm_weather curr
	
	JOIN farm_weather prev_hour ON
		DATE(prev_hour.dateutc) = DATE(curr.dateutc)
		AND HOUR(prev_hour.dateutc) = HOUR(curr.dateutc)-1
WHERE
	curr.unixtime = (
		SELECT
			MAX(farm.unixtime)
		FROM 
			weather.farm_weather farm
	)
GROUP BY curr.tempf, curr.dewptf, curr.winddir, curr.windspeedmph, curr.baromin
"""

def run_query(cursor, query):
    cursor.execute(query)
    return cursor.fetchone()

database = mysql.connector.connect(
    host="192.168.1.200",
    user="ajtgarber",
    password="jacobgarber1",
    database="weather"
)
cursor = database.cursor()

while True:
    results         = run_query(cursor, query)

    temperature     = float(results[0])
    dewpoint        = float(results[1])
    winddir         = float(results[2])
    windspeed       = float(results[3])
    air_pressure    = float(results[4])
    temp_change     = float(results[5])
    dpt_dep_change  = float(results[6])
    pressure_change = float(results[7])

    input_data = [temperature, dewpoint, winddir, windspeed, air_pressure, temp_change, dpt_dep_change, pressure_change]
    print("input_data: " + str(input_data))

    print()
    rain_model = tf.keras.models.load_model("rain_model")

    #test_data = np.array([32.2, 27.0, 29, 2.2, 1012, 0.72, 0.72, -1.24])
    output_prediction = rain_model.predict(input_data)[0][0]
    print("Model Prediction: " + str(output_prediction) + " inches")

    current_time = datetime.datetime.now()
    timestamp = current_time.timestamp()
    new_data = {
        "timestamp" : current_time,
        "current_time" : timestamp,
        "output_prediction" : str(output_prediction)
    }
    insert_query = """
    INSERT INTO rain_predictions (timestamp, unixtime, location, predicted_rainfall) VALUES (%(timestamp)s, %(current_time)s, "farm", %(output_prediction)s)
    """
    cursor.execute(insert_query, new_data)
    database.commit()
    time.sleep(3600)
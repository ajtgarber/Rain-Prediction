Processing Steps:
1. Download METAR data for nearby airports going back as far as possible [Source](https://mesonet.agron.iastate.edu/request/download.phtml) Use the below columns:
	  * Air Temperature (F)
	  * Dew Point (F)
	  * Wind Direction
	  * Wind Speed (MPH)
	  * Sea Level Pressure (mb)
	  * 1 hour Precipitation(inch)
2. Clean up the dataset (run `python3 cleanup.py`)
	* Remove lines that have unexpected nulls (expect nulls in precip data, interpret as 0.00)
3. Load the dataset into a SQL database
	```SQL
		CREATE TABLE IF NOT EXISTS `observations` (
		  `station` varchar(5) NOT NULL,
		  `timestamp` datetime NOT NULL,
		  `temperature` float DEFAULT 0,
		  `dewpoint` float DEFAULT 0,
		  `winddir` float DEFAULT 0,
		  `windspeed` float DEFAULT 0,
		  `pressure` float DEFAULT 0,
		  `rain` float DEFAULT 0,
		  PRIMARY KEY (`station`,`timestamp`)
		) ENGINE=InnoDB DEFAULT CHARSET=latin1 COLLATE=latin1_swedish_ci;
	```
4.  Run the enrichment query and export the results to `enriched_input.csv`
	```SQL
SELECT 
	curr.station, 
	curr.TIMESTAMP, 
	HOUR(curr.timestamp) AS "hour",
	curr.temperature AS "tmpf",
	curr.dewpoint AS "dwpf", 
	curr.winddir AS "drct",
	curr.windspeed AS "sped",
	curr.pressure AS "mslp", 
	curr.temperature - prev_hour.temperature AS "tmpf_change",
	(curr.temperature - curr.dewpoint) - (prev_hour.temperature - prev_hour.dewpoint) AS "dwp_dep_change",
	curr.pressure - prev_hour.pressure AS "mslp_change",
	next_hour.rain
FROM rain_pred.observations curr

JOIN rain_pred.observations prev_hour ON
	curr.station = prev_hour.station
	AND DATE_SUB(curr.timestamp, INTERVAL 1 HOUR) = prev_hour.timestamp
	
JOIN rain_pred.observations next_hour ON
	curr.station = next_hour.station
	AND DATE_ADD(curr.timestamp, INTERVAL 1 HOUR) = next_hour.timestamp
	
WHERE
	curr.pressure <> 0
	AND prev_hour.pressure <> 0
```
5. Run the training script `python3 rain.py`
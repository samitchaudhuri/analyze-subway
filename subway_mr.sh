#! /bin/bash

# Find riderhip by station
#cat ../../../datasets/subway_data/turnstile_data_master_with_weather.csv | python subway_station_mapper.py | sort | python subway_station_reducer.py

# Find riderhip by weather
#cat ../../../datasets/subway_data/turnstile_data_master_with_weather.csv | python subway_weather_mapper.py | sort | python subway_weather_reducer.py


# Find busiest hour
cat ../../../datasets/subway_data/turnstile_data_master_with_weather.csv | python subway_hour_mapper.py | sort | python subway_hour_reducer.py

import os
import requests
from datetime import datetime
from collections import defaultdict

OWM_API_KEY = os.environ.get('OWM_API_KEY', 'fedd08f01977325150beff4337f14fd4')

WIND_DIRECTIONS = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                    'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']

def deg_to_compass(deg):
    idx = round((deg or 0) / 22.5) % 16
    return WIND_DIRECTIONS[idx]

def get_coordinates(city_name):
    url = f"http://api.openweathermap.org/geo/1.0/direct?q={city_name}&limit=1&appid={OWM_API_KEY}"
    resp = requests.get(url, timeout=10).json()
    print(f"DEBUG- Geocoding Raw Response: {resp}")
    if not resp or isinstance(resp, dict):
        return None, None
    return resp[0]['lat'], resp[0]['lon']

def get_current_weather(lat, lon):
    url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&units=metric&appid={OWM_API_KEY}"
    return requests.get(url, timeout=10).json()

def get_forecast_list(lat, lon):
    """Free 5-day/3-hour forecast: returns 40 entries (3-hour steps)"""
    url = f"https://api.openweathermap.org/data/2.5/forecast?lat={lat}&lon={lon}&units=metric&appid={OWM_API_KEY}"
    resp = requests.get(url, timeout=10).json()
    return resp.get('list', [])

def get_weather_bundle(city_name):
    """Fetch current weather + 5-day/3-hour forecast (free tier, no subscription)"""
    lat, lon = get_coordinates(city_name)
    print(f"DEBUG- lat={lat}, lon={lon}")
    if lat is None:
        return None
    current = get_current_weather(lat, lon)
    print(f"DEBUG- current weather response: {current}")
    forecast_list = get_forecast_list(lat, lon)
    print(f"DEBUG- forecast list length: {len(forecast_list)}")
    return {'current': current, 'forecast_list': forecast_list}

def aggregate_daily(forecast_list):
    """Group 3-hour entries into daily aggregates. Returns dict: date_str -> agg dict"""
    daily = defaultdict(list)
    for entry in forecast_list:
        date_str = datetime.utcfromtimestamp(entry['dt']).strftime('%Y-%m-%d')
        daily[date_str].append(entry)

    result = {}
    for date_str, entries in daily.items():
        temps = [e['main']['temp'] for e in entries]
        result[date_str] = {
            'min_temp': min(temps),
            'max_temp': max(temps),
            'avg_humidity': sum(e['main']['humidity'] for e in entries) / len(entries),
            'avg_pressure': sum(e['main']['pressure'] for e in entries) / len(entries),
            'avg_wind_speed': sum(e['wind']['speed'] for e in entries) / len(entries),
            'avg_wind_deg': sum(e['wind'].get('deg', 0) for e in entries) / len(entries),
            'avg_clouds': sum(e['clouds']['all'] for e in entries) / len(entries),
            'total_rain': sum(e.get('rain', {}).get('3h', 0) for e in entries),
            'morning_temp': next((e['main']['temp'] for e in entries
                                  if 8 <= datetime.utcfromtimestamp(e['dt']).hour <= 10), temps[0]),
            'afternoon_temp': next((e['main']['temp'] for e in entries
                                    if 14 <= datetime.utcfromtimestamp(e['dt']).hour <= 16), temps[-1]),
        }
    return result

def map_daily_to_features(agg, medians, prev_day_rain=0):
    """Map one day's aggregated forecast to the model's feature dict"""
    wind_deg = agg['avg_wind_deg']
    return {
        'MinTemp': round(agg['min_temp'], 1),
        'MaxTemp': round(agg['max_temp'], 1),
        'Temp9am': round(agg['morning_temp'], 1),
        'Temp3pm': round(agg['afternoon_temp'], 1),
        'Rainfall': round(agg['total_rain'], 1),
        'Evaporation': medians.get('Evaporation', 5.0),
        'Sunshine': medians.get('Sunshine', 7.0),
        'WindGustDir': deg_to_compass(wind_deg),
        'WindGustSpeed': round(agg['avg_wind_speed'] * 3.6, 1),
        'WindDir9am': deg_to_compass(wind_deg),
        'WindSpeed9am': round(agg['avg_wind_speed'] * 3.6 * 0.85, 1),
        'WindDir3pm': deg_to_compass(wind_deg),
        'WindSpeed3pm': round(agg['avg_wind_speed'] * 3.6, 1),
        'Humidity9am': round(agg['avg_humidity']),
        'Humidity3pm': max(round(agg['avg_humidity']) - 15, 10),
        'Pressure9am': round(agg['avg_pressure'], 1),
        'Pressure3pm': round(agg['avg_pressure'], 1),
        'Cloud9am': round(agg['avg_clouds'] / 100 * 8),
        'Cloud3pm': round(agg['avg_clouds'] / 100 * 8),
        'RainToday': 'Yes' if prev_day_rain > 1 else 'No',
    }

def map_forecast_entry_to_features(entry, medians, prev_rain=0):
    """For 'next hour' approx prediction using the nearest 3-hour forecast block"""
    wind_deg = entry['wind'].get('deg', 0)
    temp = entry['main']['temp']
    return {
        'MinTemp': round(entry['main'].get('temp_min', temp - 1), 1),
        'MaxTemp': round(entry['main'].get('temp_max', temp + 1), 1),
        'Temp9am': round(temp, 1),
        'Temp3pm': round(temp, 1),
        'Rainfall': round(entry.get('rain', {}).get('3h', 0), 1),
        'Evaporation': medians.get('Evaporation', 5.0),
        'Sunshine': medians.get('Sunshine', 7.0),
        'WindGustDir': deg_to_compass(wind_deg),
        'WindGustSpeed': round(entry['wind']['speed'] * 3.6, 1),
        'WindDir9am': deg_to_compass(wind_deg),
        'WindSpeed9am': round(entry['wind']['speed'] * 3.6, 1),
        'WindDir3pm': deg_to_compass(wind_deg),
        'WindSpeed3pm': round(entry['wind']['speed'] * 3.6, 1),
        'Humidity9am': entry['main']['humidity'],
        'Humidity3pm': entry['main']['humidity'],
        'Pressure9am': entry['main']['pressure'],
        'Pressure3pm': entry['main']['pressure'],
        'Cloud9am': round(entry['clouds']['all'] / 100 * 8),
        'Cloud3pm': round(entry['clouds']['all'] / 100 * 8),
        'RainToday': 'Yes' if prev_rain > 1 else 'No',
    }
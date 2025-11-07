import requests
import pandas as pd
import time

# Define locations with their respective coordinates
locations = {
    "Spain": [
        {"name": "Castellon", "latitude": 39.986, "longitude": -0.051},
        {"name": "Valencia", "latitude": 39.469, "longitude": -0.376},
        {"name": "Alicante", "latitude": 38.345, "longitude": -0.484},
        {"name": "Murcia", "latitude": 37.988, "longitude": -1.13},
        {"name": "Seville", "latitude": 37.389, "longitude": -5.984},
        {"name": "Huelva", "latitude": 37.263, "longitude": -6.944},
        {"name": "Almería", "latitude": 36.841, "longitude": -2.467}
    ],
    "Egypt": [
        {"name": "Beheira", "latitude": 31.034, "longitude": 30.468},
        {"name": "Dakahlia", "latitude": 31.041, "longitude": 31.379},
        {"name": "Fayoum", "latitude": 29.309, "longitude": 30.841},
        {"name": "Ismailia", "latitude": 30.596, "longitude": 32.271}
    ],
    "South Africa": [
        {"name": "Limpopo", "latitude": -23.401, "longitude": 29.418},
        {"name": "Mpumalanga", "latitude": -25.397, "longitude": 30.964},
        {"name": "KwaZulu-Natal", "latitude": -29.858, "longitude": 31.021}
    ]
}

# API Base URL
base_url = "https://archive-api.open-meteo.com/v1/archive"

# Function to fetch data with retry logic
def fetch_weather_data(region, country, max_retries=3, retry_delay=20):
    params = {
        "latitude": region["latitude"],
        "longitude": region["longitude"],
        "start_date": "2000-01-01",
        "end_date": "2024-11-22",
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,rain_sum,wind_speed_10m_max,sunshine_duration",
        "timezone": "auto"
    }
    retries = 0

    while retries < max_retries:
        try:
            response = requests.get(base_url, params=params)
            if response.status_code == 200:
                data = response.json().get("daily", {})
                print(f"Data fetched successfully for {region['name']}, {country}")
                return pd.DataFrame(data)
            elif response.status_code == 429:
                print(f"Rate limit hit for {region['name']}, {country}. Retrying in {retry_delay} seconds...")
                retries += 1
                time.sleep(retry_delay)
            else:
                print(f"Error fetching data for {region['name']}, {country}: {response.status_code}")
                return None
        except requests.RequestException as e:
            print(f"Request failed for {region['name']}, {country}: {e}")
            return None

    print(f"Max retries reached for {region['name']}, {country}. Skipping...")
    return None

# Loop through locations and fetch data
all_data = []
for country, regions in locations.items():
    for region in regions:
        df = fetch_weather_data(region, country)
        if df is not None:
            df["Region"] = region["name"]
            df["Country"] = country
            all_data.append(df)

# Combine all data into a single DataFrame
if all_data:
    final_df = pd.concat(all_data, ignore_index=True)
    print(final_df.head())
    # Save to CSV
    final_df.to_csv("weather.csv", index=False)
    print("Historical weather data saved")
else:
    print("No data retrieved.")

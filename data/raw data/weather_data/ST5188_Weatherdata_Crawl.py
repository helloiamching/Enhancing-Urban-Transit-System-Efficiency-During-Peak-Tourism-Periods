import os
import pandas as pd
import requests
from tqdm import tqdm


folder_path = r"D:\WangChing\weather_data"

def fetch_rainfall_data(timestamp):
    url = f"https://api-open.data.gov.sg/v2/real-time/api/rainfall?date={timestamp}"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        if "data" in data and "readings" in data["data"]:
            rainfall_data = {}
            for reading_entry in data["data"]["readings"]:
                reading_timestamp = reading_entry["timestamp"][:19]  # Truncate to match format YYYY-MM-DDTHH:mm
                for reading in reading_entry["data"]:
                    station_id = reading["stationId"]
                    rainfall = reading.get("value", 0)
                    rainfall_data[(reading_timestamp, station_id)] = rainfall
            return rainfall_data
    return {}

for file in tqdm(os.listdir(folder_path)):
    if file.startswith("dataset_") and file.endswith(".csv"):
        print(f"Fetching rainfall data for {file}...")
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        
        unique_dates = df["timestamp"].str[:10].unique()
        rainfall_data = {}
        
        for date in tqdm(unique_dates):
            daily_rainfall = fetch_rainfall_data(date)
            rainfall_data.update(daily_rainfall)
        
        # print(rainfall_data)
        
        df["rainfall"] = df.apply(lambda row: rainfall_data.get((row["timestamp"][:19], row["station_id"]), None), axis=1)
        
        df.to_csv(file_path, encoding='utf-8', index=False)
        print(f"Updated {file} with rainfall data.")
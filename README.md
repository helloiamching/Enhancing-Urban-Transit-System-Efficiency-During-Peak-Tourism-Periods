# Data Structure

## Raw Data

The raw dataset contains four subsets:

- `ridership`
- `tourist`
- `transport_mode_train`
- `weather`

Each dataset has its own data cleaning procedures, detailed within the respective script or notebook files.

---

## Final Data

There are two final data files:

- `train_data.scv — covers the period from **November to December 2024**. This dataset is used for **training and testing**.
- `val_data.csv` — covers **January 2025**. This dataset is used for **validation**.

Both datasets share the same structure with the following columns:

| Column               | Description                                  |
|----------------------|----------------------------------------------|
| `timestamp`          | Date and hour of observation (YYYY/MM/DD HH:00) |
| `day_type`           | Type of day (e.g., WEEKENDS/HOLIDAY, WEEKDAY) |
| `station_id`         | MRT station code                             |
| `tap_in`             | Number of passengers tapping in              |
| `tap_out`            | Number of passengers tapping out             |
| `total_ridership`    | Total tap in + tap out                        |
| `Visitors`           | Tourist count estimation                     |
| `latitude`           | Latitude of the station                      |
| `longitude`          | Longitude of the station                     |
| `air_temperature`    | Temperature in Celsius                       |
| `relative_humidity`  | Relative humidity (%)                        |
| `wind_speed`         | Wind speed (m/s)                             |
| `rainfall`           | Rainfall (mm)                                |

---

## Sample Data Preview

| timestamp       | day_type          | station_id | tap_in | tap_out | total_ridership | Visitors | latitude     | longitude    | air_temperature | relative_humidity | wind_speed | rainfall |
|-----------------|-------------------|------------|--------|---------|------------------|----------|--------------|--------------|------------------|-------------------|-------------|----------|
| 2025/1/1 15:00  | WEEKENDS/HOLIDAY  | BP10       | 1735   | 1940    | 3675             | 2676     | 1.384520796  | 103.7708086 | 30.0             | 69.6              | 6.0         |          |
| 2025/1/1 16:00  | WEEKENDS/HOLIDAY  | BP10       | 1720   | 1889    | 3609             | 2697     | 1.384520796  | 103.7708086 | 24.6             | 99.6              | 9.39999     | 2.6      |
| 2025/1/1 17:00  | WEEKENDS/HOLIDAY  | BP10       | 1823   | 2125    | 3948             | 2987     | 1.384520796  | 103.7708086 | 24.8             | 99.6              | 4.5         | 0        |
| 2025/1/1 18:00  | WEEKENDS/HOLIDAY  | BP10       | 1561   | 2438    | 3999             | 3021     | 1.384520796  | 103.7708086 | 27.3             | 81.4              | 3.8         | 0        |
| 2025/1/1 19:00  | WEEKENDS/HOLIDAY  | BP10       | 1218   | 2281    | 3499             | 2326     | 1.384520796  | 103.7708086 | 27.1             | 90.9              | 1.1         | 0        |

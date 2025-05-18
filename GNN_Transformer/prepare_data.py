import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import math
import time
import gc
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import matplotlib.pyplot as plt
import re

    
def prepare_data(df, seq_length, pred_length):
    
    df['ridership'] = df['tap_in'] + df['tap_out']
    df.drop(columns=['tap_in', 'tap_out', 'Latitude', 'Longitude', 'is_transfer_hub'], inplace=True)
    df['ridership'] = df['ridership'].fillna(0)

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour   
    

    df.loc[df['date'].isin([pd.to_datetime('2025-01-29').date(), pd.to_datetime('2025-01-30').date()]), 'is_holiday'] = 1

   
    df['is_holiday'] = df['is_holiday'].fillna(0).astype(int)
        
    df = df[df['station_id'] != 'CG2']
    df = df[~df['station_id'].str.startswith(('PE', 'PW', 'SE', 'SW'))]

    df['station_id'] = df['station_id'].replace({
        'CE1/DT16': 'DT16',
        'CG1/DT35': 'DT35'
    })


    station_merge_map = {
        'DT10': 'DT10/TE11',
        'TE11': 'DT10/TE11'
    }

    df['logical_station_id'] = df['station_id'].replace(station_merge_map)

    df = df.groupby(['logical_station_id', 'date', 'hour'], as_index=False).agg({
        'ridership': 'sum',
        'air_temperature': 'first',
        'relative_humidity': 'first',
        'wind_speed': 'first',
        'rainfall': 'first',
        'is_holiday': 'first',
        'timestamp': 'first',
        'Visitors': 'first'
    })
    df = df.rename(columns={'logical_station_id': 'station_id'})

    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
    
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    #all combines：station × data × 0~23 hour
    def station_sort_key(station_id):
        match = re.match(r'([A-Z]+)(\d+)', station_id)
        if match:
            prefix, num = match.groups()
            return (prefix, int(num))
        else:
            return (station_id, 0)

    stations = sorted(df['station_id'].unique(), key=station_sort_key)

    dates = df['date'].unique()
    hours = np.arange(24)

    full_index = pd.MultiIndex.from_product(
        [stations, dates, hours],
        names=['station_id', 'date', 'hour']
    )

    # merge with original dataframe
    df_full = pd.DataFrame(index=full_index).reset_index()
    df = pd.merge(df_full, df, how='left', on=['station_id', 'date', 'hour'])

    df['timestamp'] = pd.to_datetime(df['date'].astype(str)) + pd.to_timedelta(df['hour'], unit='h')

    
    holiday_map = df[['date', 'is_holiday']].dropna().drop_duplicates().set_index('date')['is_holiday']
    weekend_map = df[['date', 'is_weekend']].dropna().drop_duplicates().set_index('date')['is_weekend']
    df['is_holiday'] = df['date'].map(holiday_map).fillna(0).astype(int)
    df['is_weekend'] = df['date'].map(weekend_map).fillna(0).astype(int)

    
    df['is_synthetic'] = df['ridership'].isna()

    df['ridership'] = df['ridership'].fillna(0)

    feature_cols = [
        'air_temperature', 'relative_humidity', 'wind_speed', 'rainfall',
        'is_holiday', 'is_weekend', 'Visitors', 
        'hour_sin', 'hour_cos', 'dayofweek_sin', 'dayofweek_cos', 
        'ridership'  
    ]
    scaled_features = ['air_temperature', 'relative_humidity', 'wind_speed', 'rainfall', 'Visitors']
    input_feature_cols = feature_cols + ['is_synthetic']
    
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

    scalers = {}
    for col in scaled_features:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)  
        scaler = RobustScaler()
        df[col + '_scaled'] = scaler.fit_transform(df[[col]])
        scalers[col] = scaler
    
    df['ridership'] = pd.to_numeric(df['ridership'], errors='coerce').fillna(0)
    df['ridership_log'] = np.log1p(df['ridership'])
    
    #scaler_ridership = RobustScaler()
    #df['ridership_scaled'] = scaler_ridership.fit_transform(df[['ridership_log']])

    #scalers['ridership'] = scaler_ridership
    
    
    X, y, hour_labels, timestamp_labels, loss_masks, future_holidays = [], [], [], [], [], []
    
    station_data = {}
    
    for station in stations:
        station_df = df[df['station_id'] == station].sort_values('timestamp')
        
        features = []
        for _, row in station_df.iterrows():
            row_features = []
            for col in input_feature_cols:
                if col in scaled_features:
                    row_features.append(row[col + '_scaled'])
                elif col == 'ridership':
                    row_features.append(row['ridership_log'])
                else:
                    row_features.append(row[col])
            features.append(row_features)
        features = np.array(features)
        
        station_X = []
        station_y = []
        station_hour_labels = []
        station_timestamps = []
        station_loss_masks = []
        station_future_holidays = []
        station_future_weekends = []
        
        for i in range(0, len(station_df) - seq_length - pred_length + 1, 1):
            station_X.append(features[i:i+seq_length])
            station_y.append(station_df.iloc[i+seq_length:i+seq_length+pred_length]['ridership_log'].values)

            station_hour_labels.append(station_df.iloc[i+seq_length:i+seq_length+pred_length]['hour'].values)
            station_future_holidays.append(station_df.iloc[i+seq_length:i+seq_length+pred_length]['is_holiday'].values)
            station_future_weekends.append(station_df.iloc[i+seq_length:i+seq_length+pred_length]['is_weekend'].values)
                
            station_timestamps.append(station_df.iloc[i+seq_length]['timestamp'])
            loss_mask = station_df.iloc[i+seq_length:i+seq_length+pred_length]['is_synthetic'].values == False
            station_loss_masks.append(loss_mask)
        
        station_data[station] = {
            'X': np.array(station_X),
            'y': np.array(station_y),
            'hour_labels': np.array(station_hour_labels),
            'timestamp_labels': np.array(station_timestamps),
            'loss_masks': np.array(station_loss_masks),
            'future_holidays': np.array(station_future_holidays),
            'future_weekends': np.array(station_future_weekends)
        }
    
    min_samples = min([len(station_data[s]['X']) for s in stations])
    
    num_stations = len(stations)
    X = np.zeros((min_samples, num_stations, station_data[stations[0]]['X'].shape[1], station_data[stations[0]]['X'].shape[2]))
    y = np.zeros((min_samples, num_stations, station_data[stations[0]]['y'].shape[1]))
    hour_labels = np.zeros((min_samples, num_stations, station_data[stations[0]]['hour_labels'].shape[1]))
    loss_masks = np.zeros((min_samples, num_stations, station_data[stations[0]]['loss_masks'].shape[1]))
    future_holidays = np.zeros((min_samples, num_stations, station_data[stations[0]]['future_holidays'].shape[1]))
    future_weekends = np.zeros((min_samples, num_stations, station_data[stations[0]]['future_weekends'].shape[1]))
    timestamp_labels = np.array(station_data[stations[0]]['timestamp_labels'][:min_samples]) # [B]
    station_ids = np.array(stations).reshape(1, num_stations)
    station_ids = np.tile(station_ids, (min_samples, 1)) # [B, N]
    
    for i, station_idx in enumerate(range(num_stations)):
        station = stations[station_idx]
        X[:, station_idx] = station_data[station]['X'][:min_samples] # [B, N, T, F]
        y[:, station_idx] = station_data[station]['y'][:min_samples] # [B, N, pred]
        hour_labels[:, station_idx] = station_data[station]['hour_labels'][:min_samples] # [B, N, pred]
        loss_masks[:, station_idx] = station_data[station]['loss_masks'][:min_samples] # [B, N, pred]
        future_holidays[:, station_idx] = station_data[station]['future_holidays'][:min_samples] # [B, N, pred]
        future_weekends[:, station_idx] = station_data[station]['future_weekends'][:min_samples]

    X = torch.tensor(X, dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32)
    hour_labels = torch.tensor(hour_labels, dtype=torch.int64)
    loss_masks = torch.tensor(loss_masks, dtype=torch.bool)
    future_holidays = torch.tensor(future_holidays, dtype=torch.float32)
    future_weekends = torch.tensor(future_weekends, dtype=torch.float32)
    
    return X, y, hour_labels, timestamp_labels, station_ids, loss_masks, future_holidays, future_weekends, scalers
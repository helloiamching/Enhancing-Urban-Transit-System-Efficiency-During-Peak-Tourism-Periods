# MRT Ridership Forecasting - GNN

This module implements an GNN model to predict MRT ridership, including tap-in and tap-out passenger flows, using historical data as outcome variable and outside features as predictive variables.


## Model Overview

The model uses a sliding window approach (past 6 hours) to predict future `tap_in` and `tap_out` volumes for each MRT station.  
Input features include:

- Tourist count
- Weather conditions (temperature, humidity, wind speed, rainfall)
- Station type (transfer hub or not)
- Holiday indicator


## Model Architecture

- GATConv (4 heads + ReLU, Dropout rate =0.05)
- GATConv (1 head + ReLU) 
- SAGEConv (ReLU)
- GRU 
- Dense (ReLU, Dropout rate =0.05)
- Dense (ReLU)
- Dense 



## Environment Setup
Install dependencies using pip at the begining:

```python
!pip install torch_geometric
```


# How to Run

1. Input organized files `train_data.csv` and `val_data.csv`, along with the package `edge_index.py`, which was written by our group member, Nie Haiyi.

2. Run step 1, data cleaning procedure to set up the data with correct format.

3. Run step 2, create graph structure. In this project, we directly link to the package `edge_index.py`, which contains the graph with MRT stations as nodes and real routes as edges. However, in other cases, the graph can also be set up according to personal needs in this step.

4. Run step 3, time series features and targets. To capture the time-related lags, traditional ACF and PACF are conducted to test the reasonable lags. In this case, we use lags under 3, and also standardize the data in this step. The output includes the original data values, standardized data values, and the station codes, where station codes help the time-related procedure be concatenated with the spatial one.

5. Run step 4, define GNN model. The GNN model is defined as a 7-layer model, and can be personalized according to project demands.

6. Run step 5, Training function. To fulfill the need of customized model structure, existing popular training packages are unavailable. Therefore, the train_model function is defined here. Visualization of training output is also defined in this section.

7. Run step 6, Modelling. Run the main structure of null, reduced and full model. The main() function defines the whole structure and helps to print the final output.

| Model         | Variables |
|---------------|-----------|
| **Null Model** | latitude, longitude, timestamp (year, month, day, hour, weekend/weekday), weather (air_temperature, relative_humidity, wind_speed, rainfall) |
| **Reduced Model** | latitude, longitude, timestamp (year, month, day, hour, weekend/weekday), weather (air_temperature, relative_humidity, wind_speed, rainfall), **visitors**, **is_transfer_hub** |
| **Full Model** | latitude, longitude, timestamp (year, month, day, hour, weekend/weekday), weather (air_temperature, relative_humidity, wind_speed, rainfall), **Visitors**, **is_transfer_hub**, **hour (hour_sin, hour_cos)** |




## Evaluation Results
| Model                   | MSE        | RMSE    | MAE      | MAPE    | SMAPE   | R²     |
|------------------------|------------|---------|----------|---------|---------|--------|
| Null Weekday Model     | 5,370,901  | 2317.52 | 1437.9994| 31.20%  | 18.79%  | 0.8262 |
| Null Weekend Model     | 520,318    | 721.33  | 502.5416 | 95.13%  | 22.74%  | 0.8871 |
| Null Combined Model    | 3,503,573  | 1871.78 | 1074.2975| 38.11%  | 18.18%  | 0.8750 |
| Reduced Weekend Model  | 536,365    | 732.37  | 520.7784 | 102.62% | 23.39%  | 0.8836 |
| Reduced Weekday Model  | 6,304,048  | 2510.79 | 1606.3489| 38.39%  | 20.65%  | 0.7960 |
| Reduced Combined Model | 4,344,276  | 2084.29 | 1195.0671| 38.11%  | 18.89%  | 0.8450 |
| Full Weekend Model     | 490,941.16 | 700.67  | 490.7978 | 90.12%  | 22.55%  | 0.8935 |
| Full Weekday Model     | 4,830,803.50| 2197.91| 1351.6155| 48.01%  | 17.96%  | 0.8437 |
| Full Combined Model    | 3,524,991.00| 1877.50| 1089.5063| 42.05%  | 18.57%  | 0.8742 |

.


## Reproducibility

The project has been tested with the following environment:

- Python version: 3.11.12
- PyTorch version: 2.6.0+cu124
- torch_geometric version: 2.6.1
- pandas version: 2.2.2
- networkx version: 3.4.2
- NumPy version: 2.0.2
- scikit-learn version: 1.6.1
- SciPy version: 1.14.1
- tqdm version: 4.67.1








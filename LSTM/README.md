# MRT Ridership Forecasting - LSTM Module

This module implements an LSTM-based approach to predict MRT ridership, including tap-in and tap-out passenger flows, using historical data and contextual features.



## Model Overview

The model uses a sliding window approach (past 6 hours) to predict future `tap_in` and `tap_out` volumes for each MRT station.  
Input features include:

- Tourist count
- Weather conditions (temperature, humidity, wind speed, rainfall)
- Station type (transfer hub or not)
- Holiday indicator



## Model Architecture

- LSTM (64 units)
- Dropout (rate = 0.2)
- Dense (32 units, ReLU)
- Dense (2 units output: tap_in and tap_out)





## Environment Setup

Install dependencies using pip:

```python
pip install pandas numpy matplotlib scikit-learn tensorflow
```


## How to Run


1. Start from the raw preprocessed dataset `Final_MRT_Data.csv`.

2. Normalize the features to generate `Final_MRT_Data_Normalized.csv`.  
 

3. Place the normalized file `Final_MRT_Data_Normalized.csv` in the working directory.

4. Run the training script:

```python
MRT_LSTM.ipynb
```
This will:

- Construct sliding window sequences
- Split data into training and test sets
- Train the model
- Evaluate performance
- Save visualizations to `results/`

## Evaluation Results
| Dataset       | MAE     | RMSE    | 
|---------------|---------|---------|
| Full Dataset  | 0.0202  | —       | 
| Holiday Only  | 0.0144  | 0.0196  | 
A vectorized prediction plot for the best-performing station is saved as `tap_in_prediction_vector.pdf`.



## Reproducibility

The project has been tested with the following environment:

- Python version: 3.10  
- TensorFlow version: 2.12.0  
- NumPy version: 1.24.3  
- pandas version: 2.0.3  
- scikit-learn version: 1.2.2  
- matplotlib version: 3.7.1  








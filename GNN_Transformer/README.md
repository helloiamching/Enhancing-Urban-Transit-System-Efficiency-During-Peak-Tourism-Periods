# MRT Ridership Forecasting – GNN-Transformer

This module implements a hybrid GNN-Transformer model to forecast MRT ridership using spatiotemporal features from historical data.

## Project Structure

The directory contains the following Python files:

- `GNN-Transformer.py` – Main script for training and evaluation.  
- `prepare_data.py` – Module for preprocessing features and constructing sliding windows.  
- `edge_index.py` – Defines the MRT station graph structure using adjacency relationships.

The main script automatically imports the other two modules during execution.

## Model Overview

The model uses a sliding window approach to predict future hourly ridership across all MRT stations.

Input features include:

- Tourist count  
- Weather conditions (temperature, humidity, wind speed, rainfall)  
- Hour of day (sine/cosine encoding)  
- Holiday indicator  
- Past tap-in and tap-out values  

## Model Architecture

- **Spatial Encoder**:  
  - GCNConv layer to capture local station-level interactions

- **Temporal Encoder**:  
  - Transformer Encoder (multi-head attention, position encoding)  
  - LayerNorm, Dropout (0.1–0.3)

- **Output Layer**:  
  - Fully connected layer to output next hour’s ridership for each node

## Environment Setup

Install dependencies using pip:

```bash
pip install torch torch-geometric pandas numpy matplotlib scikit-learn
```

*GPU is recommended for training.*

## How to Run

1. Ensure dataset is placed in the working directory.  
2. Run the training script:

```bash
python GNN-Transformer.py
```

This will:

- Preprocess input data and generate sliding windows  
- Construct graph edges using `edge_index.py`  
- Train model with early stopping on validation loss  
- Evaluate performance  
Note: To display loss curves or prediction plots during training, uncomment the relevant `matplotlib` lines in the script.

## Evaluation Results

The training loss consistently decreases over time, while the validation loss fluctuates across sliding windows. This suggests that the model may be overfitting to specific training segments and failing to generalize well across varying temporal patterns.


## Reproducibility

This module has been tested with the following environment:

- Python version: 3.9  
- torch 2.5.1+cu121  
- torch-geometric 2.6.1  
- numpy 2.0.2  
- pandas 2.2.3  
- scikit-learn 1.6.1  
- matplotlib 3.9.4


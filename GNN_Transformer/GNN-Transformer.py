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
import matplotlib.pyplot as plt
from torch_geometric.nn import GCNConv
from torch.nn import TransformerEncoder, TransformerEncoderLayer
import re
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import TimeSeriesSplit
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import CosineAnnealingLR

from prepare_data import prepare_data
from edge_index import edge


# ========== 1. Model Build ==========
class GNNTransformer(nn.Module):
    def __init__(self, input_dim, time_length, hidden_dim, out_dim, n_heads, n_layers, dropout=0.1):
        super(GNNTransformer, self).__init__()

        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # --- Step 1: Input ---
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # --- Step 2: Temporal （Transformer）---
        encoder_layer = TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.temporal_encoder = TransformerEncoder(encoder_layer, num_layers=n_layers)

        # --- Step 3: Spatial（GCN）---
        self.gcn1 = GCNConv(hidden_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)

        # --- Step 4: Decoder ---
        self.decoder_proj = nn.Linear(6, hidden_dim)   
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),  
            nn.Dropout(0.1),           
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, future_time_feats):
        """
        Args:
            x: [B, N, T, F]
            edge_index: [2, num_edges]
            future_holidays: [B, N, out_dim]
            future_weekends: [B, N, out_dim]
        Returns:
            out: [B, N, out_dim]
        """
        B, N, T, Feature = x.shape

        # --- Step 1: Input ---
        x = self.input_proj(x)                          # [B, N, T, hidden_dim]

        # --- Step 2: Temporal ---
        x = x.reshape(B * N, T, self.hidden_dim)
        x = self.temporal_encoder(x)                    # [B*N, T, hidden_dim]
        x = x.mean(dim=1)                               
        x = x.view(B, N, self.hidden_dim)               # [B, N, hidden_dim]

        # --- Step 3: Spatial ---
        x_out = []
        for b in range(B):
            xb = x[b]                                   # [N, hidden_dim]
            xb = self.gcn1(xb, edge_index)
            xb = F.relu(xb)          
            xb = self.dropout(xb)
            
            xb = self.gcn2(xb, edge_index)
            xb = F.relu(xb)  
            
            x_out.append(xb)
        x = torch.stack(x_out, dim=0)                   # [B, N, hidden_dim]

        # --- Step 4: Decoder （future covariates）---

        decoder_encoded = self.decoder_proj(future_time_feats)
        # → [B, N, out_dim, hidden_dim]
        
        # encoder  [B, N, out_dim, hidden_dim]
        encoder_expanded = x.unsqueeze(2).expand(-1, -1, self.out_dim, -1)
        
        # encoder + decoder
        combined = torch.cat([encoder_expanded, decoder_encoded], dim=-1)  # [B, N, out_dim, 2*hidden_dim]
        
        # Final prediction [B, N, out_dim]
        combined_reshaped = combined.view(B*N*self.out_dim, -1)
        out = self.output_layer(combined_reshaped)  # [B*N*out_dim, 1]
        out = out.view(B, N, self.out_dim)  #  [B, N, out_dim]
        #out = torch.relu(out)

        return out


def build_decoder_time_features(hour_labels, timestamp_labels, future_holidays, future_weekends):
    import numpy as np
    import torch
    import pandas as pd

    B, N, pred = hour_labels.shape

    hour_sin = np.sin(2 * np.pi * hour_labels.numpy() / 24)
    hour_cos = np.cos(2 * np.pi * hour_labels.numpy() / 24)
    dayofweek_sin = np.zeros((B, N, pred))
    dayofweek_cos = np.zeros((B, N, pred))

    for b in range(B):
        base_time = pd.to_datetime(timestamp_labels[b])
        for t in range(pred):
            dt = base_time + pd.Timedelta(hours=t)
            dow = dt.weekday()
            dayofweek_sin[b, :, t] = np.sin(2 * np.pi * dow / 7)
            dayofweek_cos[b, :, t] = np.cos(2 * np.pi * dow / 7)

    decoder_feats = np.stack([
        hour_sin,
        hour_cos,
        dayofweek_sin,
        dayofweek_cos,
        future_holidays.numpy(),
        future_weekends.numpy()
    ], axis=-1)  # [B, N, pred, 6]

    return torch.tensor(decoder_feats, dtype=torch.float32)



# ========== 2. Windows data ==========
def create_rolling_loaders(X, y, loss_masks, decoder_feats, batch_size, train_size=600, val_size=24, step=24):
    rolling_loaders = []
    num_samples = X.shape[0]

    end_point = num_samples - val_size
    start_points = range(0, end_point - train_size + 1, step)

    for start in start_points:
        train_idx = slice(start, start + train_size)
        val_idx = slice(start + train_size, start + train_size + val_size)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        mask_train, mask_val = loss_masks[train_idx], loss_masks[val_idx]
        decoder_train, decoder_val = decoder_feats[train_idx], decoder_feats[val_idx]

        train_dataset = TensorDataset(X_train, y_train, mask_train, decoder_train)
        val_dataset = TensorDataset(X_val, y_val, mask_val, decoder_val)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

        rolling_loaders.append((train_loader, val_loader))

    return rolling_loaders

# ========== 3. loss function ==========
def inverse_scale_ridership(y_log_pred, log_cap=20):

    if torch.is_tensor(y_log_pred):
        y_log = y_log_pred.detach().cpu().numpy()
    else:
        y_log = y_log_pred
    
    y_log = np.clip(y_log, a_min=None, a_max=log_cap)
    
    y_orig = np.expm1(y_log)
    
    return y_orig

def masked_mse(y_pred, y_true, loss_mask):
    
    y_true_orig = torch.tensor(inverse_scale_ridership(y_true), device=y_true.device)
    weights = torch.sqrt(y_true_orig + 1.0)  
    loss = (y_pred - y_true) ** 2
    mask = loss_mask.float()
    weights = weights * mask
    return (loss * weights).sum() / (weights.sum() + 1e-8)



# ========== 4. Train + Validation ==========
def compute_denominator_for_mase(y_true, seasonality=24):

    if len(y_true.shape) > 1:
        y_true = y_true.flatten()
    
    if len(y_true) <= seasonality:
        return None
    
    errors = np.abs(y_true[seasonality:] - y_true[:-seasonality])
    
    mean_error = np.mean(errors)
    return mean_error if mean_error > 0 else None

def train_one_epoch(model, dataloader, optimizer, edge_index):
    model.train()
    total_loss = 0
    total_mse = 0
    total_mae = 0
    total_mape = 0
    total_smape = 0
    total_count = 0

    all_preds = []
    all_trues = []
    all_valid_masks = []

    for x, y, mask, decoder in dataloader:
        x, y, mask, decoder = x.to(device), y.to(device), mask.to(device), decoder.to(device)
        optimizer.zero_grad()
        y_pred = model(x, edge_index, decoder)

        loss = masked_mse(y_pred, y, mask)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

        with torch.no_grad():
            all_preds.append(y_pred.detach().cpu())
            all_trues.append(y.detach().cpu())
            all_valid_masks.append(mask.detach().cpu())

            y_true_np = inverse_scale_ridership(y.detach())
            y_pred_np = inverse_scale_ridership(y_pred.detach())
            mask_np = mask.detach().cpu().numpy()

            valid = (mask_np == 1)
            if valid.sum() == 0:
                continue

            total_count += valid.sum()
            total_mse += ((y_pred_np - y_true_np) ** 2)[valid].sum()
            total_mae += np.abs(y_pred_np - y_true_np)[valid].sum()
            total_mape += (np.abs(y_pred_np - y_true_np) / np.maximum(y_true_np, 1.0))[valid].sum() * 100
            total_smape += (np.abs(y_pred_np - y_true_np) / (np.abs(y_pred_np) + np.abs(y_true_np) + 1e-8))[valid].sum() * 200

    avg_loss = total_loss / len(dataloader)
    avg_mse = total_mse / total_count if total_count > 0 else np.nan
    avg_rmse = np.sqrt(avg_mse) if not np.isnan(avg_mse) else np.nan
    avg_mae = total_mae / total_count if total_count > 0 else np.nan
    avg_mape = total_mape / total_count if total_count > 0 else np.nan
    avg_smape = total_smape / total_count if total_count > 0 else np.nan

    #MASE
    all_trues_np = np.concatenate([inverse_scale_ridership(t) for t in all_trues]).flatten()
    mase_denom = compute_denominator_for_mase(all_trues_np)
    mase = avg_mae / mase_denom if mase_denom is not None else np.nan

    metrics = {
        'loss': avg_loss,
        'rmse': avg_rmse,
        'mape': avg_mape,
        'smape': avg_smape,
        'mase': mase
    }

    return metrics


def evaluate_model(model, dataloader, edge_index, correction_factor=1.0):
    model.eval()
    all_true, all_pred = [], []
    total_loss = 0
    
    with torch.no_grad():
        for x, y, mask, decoder in dataloader:
            x, y, mask, decoder = x.to(device), y.to(device), mask.to(device), decoder.to(device)
            #y_pred = model(x, edge_index, decoder) * correction_factor
            y_pred = model(x, edge_index, decoder)
            loss = masked_mse(y_pred, y, mask)
            total_loss += loss.item()
            
            y_true_np = inverse_scale_ridership(y)
            mask_np = mask.detach().cpu().numpy() if torch.is_tensor(mask) else mask
            y_pred_np = inverse_scale_ridership(y_pred)
            y_pred_np[~mask_np] = 0.0
            
            all_true.append(y_true_np)
            all_pred.append(y_pred_np)
    

    y_true_flat = np.concatenate([t.flatten() for t in all_true])
    y_pred_flat = np.concatenate([p.flatten() for p in all_pred])
    valid_mask = y_true_flat > 0
    
    if np.sum(valid_mask) > 0:
        # MSE,RMSE
        mse = np.mean((y_pred_flat[valid_mask] - y_true_flat[valid_mask])**2)
        rmse = np.sqrt(mse)
        
        # MAE
        mae = np.mean(np.abs(y_pred_flat[valid_mask] - y_true_flat[valid_mask]))
        
        # MAPE
        mape = np.mean(np.abs(y_pred_flat[valid_mask] - y_true_flat[valid_mask]) / 
                      y_true_flat[valid_mask]) * 100
        
        # SMAPE
        smape = np.mean(2 * np.abs(y_pred_flat[valid_mask] - y_true_flat[valid_mask]) / 
                       (np.abs(y_pred_flat[valid_mask]) + np.abs(y_true_flat[valid_mask]) + 1e-8)) * 100
        
        # MASE
        mase_denom = compute_denominator_for_mase(y_true_flat[valid_mask])
        if mase_denom is not None:
            mase = mae / mase_denom
        else:
            mase = np.nan
    else:
        mse = rmse = mae = mape = smape = mase = np.nan
    
    # avg_loss
    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else np.nan
    

    metrics = {
        'loss': avg_loss,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'smape': smape,
        'mase': mase
    }
    
    return metrics, all_true, all_pred

if __name__ == "__main__":
   
    torch.manual_seed(42)
    np.random.seed(42)
    
    df_train = pd.read_csv('../data/train_data.csv')
    df_val = pd.read_csv('../data/val_data.csv')

    df = pd.concat([df_train, df_val], ignore_index=True)
    
    mask = df['station_id'].str.startswith('BP') | (df['station_id'] == 'DT2')
    '''
    mask = (
        df['station_id'].str.startswith('CC')
        | (df['station_id'] == 'NS24/NE6/CC1') 
        | (df['station_id'] == 'EW8/CC9') 
        | (df['station_id'] == 'NE12/CC13')
        | (df['station_id'] == 'NS17/CC15')
        | (df['station_id'] == 'EW21/CC22')
        | (df['station_id'] == 'NE1/CC29')
        | (df['station_id'] == 'DT8') #45
        | (df['station_id'] == 'DT10') #46
        | (df['station_id'] == 'TE11') #46
        | (df['station_id'] == 'NE7/DT12') #48
        | (df['station_id'] == 'EW12/DT14') #50
        | (df['station_id'] == 'CE1/DT16') #51
        | (df['station_id'] == 'DT25') #60
        | (df['station_id'] == 'DT27') #61
        | (df['station_id'] == 'EW7') #75
        | (df['station_id'] == 'EW9') #76
        | (df['station_id'] == 'NS25/EW13') #79
        | (df['station_id'] == 'EW16/NE3/TE17') #82
        | (df['station_id'] == 'EW20') #86
        | (df['station_id'] == 'EW22') #87
        | (df['station_id'] == 'NE5') #99
        | (df['station_id'] == 'NE11') #103
        | (df['station_id'] == 'NE13') #104
        | (df['station_id'] == 'NS16') #123
        | (df['station_id'] == 'NS18') #124
        | (df['station_id'] == 'NS23') #128
        | (df['station_id'] == 'TE8') #137
    )
    '''
    df = df[mask]
    
    seq_length=720
    pred_length=24 
    
    X, y, hour_labels, timestamp_labels, station_ids, loss_masks, future_holidays, future_weekends, scalers=prepare_data(df, seq_length, pred_length)
    edge_index=edge()
    
    target_nodes = torch.tensor(list(range(0, 12)))
    #target_nodes = torch.tensor(list(range(12, 40)))
    
    mask = torch.isin(edge_index[0], target_nodes) | torch.isin(edge_index[1], target_nodes)
    filtered_edge_index  = edge_index[:, mask]
    
    unique_nodes = torch.unique(filtered_edge_index)
    id_map = {old.item(): new for new, old in enumerate(unique_nodes)}
    
    edge_index = torch.tensor([
        [id_map[i.item()] for i in filtered_edge_index[0]],
        [id_map[i.item()] for i in filtered_edge_index[1]],
    ], dtype=torch.long)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    future_time_feats = build_decoder_time_features(hour_labels, timestamp_labels, future_holidays, future_weekends)
    
    model = GNNTransformer(
            input_dim=X.shape[-1],
            time_length=X.shape[2],
            hidden_dim=64,
            out_dim=y.shape[-1],
            n_heads=4,
            n_layers=2,
            dropout=0.1
        ).to(device)
    
    edge_index = edge_index.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    #scheduler = CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)
    
    test_size = 24
    train_val_size = len(X) - test_size
    
    # train and test
    X_train_val, y_train_val = X[:train_val_size], y[:train_val_size]
    loss_masks_train_val = loss_masks[:train_val_size]
    future_time_feats_train_val = future_time_feats[:train_val_size]
    
    X_test, y_test = X[train_val_size:], y[train_val_size:]
    loss_masks_test = loss_masks[train_val_size:]
    future_time_feats_test = future_time_feats[train_val_size:]
    
    # test
    test_dataset = TensorDataset(X_test, y_test, loss_masks_test, future_time_feats_test)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)
    
    
    rolling_loaders = create_rolling_loaders(
        X_train_val, y_train_val, loss_masks_train_val, future_time_feats_train_val,
        batch_size=32, train_size=720, val_size=24, step=24
    )
    '''
    rolling_loaders = create_rolling_loaders(
        X_train_val, y_train_val, loss_masks_train_val, future_time_feats_train_val,
        batch_size=4, train_size=600, val_size=24, step=24
    )
    '''
    #print(f"windows: {len(rolling_loaders)}")
    
    all_train_history = []
    all_val_history = []
    
    for window_id, (train_loader, val_loader) in enumerate(rolling_loaders):
        print(f"\n=== Window {window_id+1} ===")
    
        train_metrics_history = []
        val_metrics_history = []
    
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        best_model_state = None
    
        for epoch in range(50):
            # === Train ===
            train_metrics = train_one_epoch(model, train_loader, optimizer, edge_index)
            train_metrics_history.append(train_metrics)
            '''
            print(f"[Epoch {epoch+1}] Train: "
                  f"Loss={train_metrics['loss']:.4f}, RMSE={train_metrics['rmse']:.2f}, "
                  f"MAPE={train_metrics['mape']:.2f}%, SMAPE={train_metrics['smape']:.2f}%, MASE={train_metrics['mase']:.4f}")
            '''
    
            # === Validation ===
            val_metrics, val_true, val_pred = evaluate_model(model, val_loader, edge_index)
            val_metrics_history.append(val_metrics)
            '''
            print(f"[Epoch {epoch+1}] Val: "
                  f"Loss={val_metrics['loss']:.4f}, RMSE={val_metrics['rmse']:.2f}, "
                  f"MAPE={val_metrics['mape']:.2f}%, SMAPE={val_metrics['smape']:.2f}%, MASE={val_metrics['mase']:.4f}")
            '''
            print(f"[Epoch {epoch+1}] Train Loss={train_metrics['loss']:.4f} | Val Loss={val_metrics['loss']:.4f}")
    
            # Early Stop
            current_val_loss = val_metrics['loss']
            if not np.isnan(current_val_loss) and current_val_loss < best_val_loss:
                best_val_loss = current_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
    
        window_metrics = {}
        for metric in ['rmse', 'mape', 'smape', 'mase']:
            values = [m[metric] for m in val_metrics_history if not np.isnan(m[metric])]
            window_metrics[metric] = np.mean(values) if values else np.nan
    
        '''
        print(f"\n=== Window {window_id+1} Average ===")
        print(f"RMSE={window_metrics['rmse']:.2f}, MAPE={window_metrics['mape']:.2f}%, "
              f"SMAPE={window_metrics['smape']:.2f}%, MASE={window_metrics['mase']:.4f}")
        '''
        all_train_history.append(train_metrics_history)
        all_val_history.append(val_metrics_history)
    
        if window_id == len(rolling_loaders) - 1:
            final_metrics, true_values, pred_values = evaluate_model(model, val_loader, edge_index)
    
    '''
    
    test_metrics, test_true, test_pred = evaluate_model(
        model, test_loader, edge_index, correction_factor=correction_factor
    )
    
    print(f"Test: RMSE={test_metrics['rmse']:.2f}, MAE={test_metrics['mae']:.2f}, " +
          f"MAPE={test_metrics['mape']:.2f}%, SMAPE={test_metrics['smape']:.2f}%, MASE={test_metrics['mase']:.4f}")
    '''
    
    '''
    all_train_loss = [m['loss'] for window in all_train_history for m in window]
    all_val_loss = [m['loss'] for window in all_val_history for m in window]
    
    plt.figure(figsize=(10, 5))
    plt.plot(all_train_loss, label='Train Loss', linestyle='--')
    plt.plot(all_val_loss, label='Val Loss', linestyle='-')
    plt.title("Full Training Timeline: Loss over All Windows")
    plt.xlabel("Training Step (Window × Epoch)")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    '''




## Optimization/ — MOPSO-Based Scheduling Module

This folder contains the notebook implementation of the MRT train scheduling optimization module based on Multi-Objective Particle Swarm Optimization (MOPSO).

### Files and Responsibilities

- `MOPSO.ipynb`  
  The main notebook that performs:
  - Data loading and preprocessing
  - Problem definition with 3 objectives: congestion, cost, waiting time
  - NSGA-II optimization using `pymoo`
  - Pareto front visualization in both design and objective space
  - Simulation-based evaluation on:
    - Normal station
    - Interchange station

- **Expected Input:**
  - `Final_MRT_Weather_Visitors_Holiday.csv`  
    Preprocessed data containing MRT ridership, weather, tourist volume, and holiday indicators

- **Generated Outputs:**
  - Pareto front plots (via `matplotlib`)
  - Printed evaluation metrics: congestion, waiting time, utilization

### How to Run

Open and execute `optimization/MOPSO.ipynb` in JupyterLab or VSCode.  
Ensure the input CSV file is located in the project root or `data/`.

Dependencies required:
```bash
pip install pymoo pandas numpy matplotlib
```

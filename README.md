# CHARM: Causal Hesitation Analysis for Recommendation via Mutual Information

## Introduction
This work was accepted by SIGIR26

## Core Features
*   **Multi-Behavior LightGCN**: Utilizes Graph Neural Networks (GCN) for feature propagation on the user-item bipartite graph to effectively integrate multi-behavior data.
*   **Intent Discovery**: Aggregates and generates the user's potential purchase intent via an Attention mechanism combining auxiliary behavior history.
*   **Causal Hesitation Modeling**: 
    *   Performs Causal feature injection on hesitated items (interested but not bought).
    *   Introduces **Substitution Penalty** to identify and downweight hesitated items that have been "substituted" by purchased items.
*   **InfoNCE Maximization**: Uses Mutual Information estimation to reinforce consistency between intent representations and real interactions.
*   **Meta-Learning Weight Adjustment**: (Optional) Integrates `BGNN` and `MetaWeightNet` to dynamically and adaptively adjust loss weights for multi-tasks based on training status.

## Dependencies
*   Python 3.8+
*   PyTorch 1.7+ (CUDA supported)
*   NumPy
*   SciPy

## File Structure
| Filename | Description |
| :--- | :--- |
| **`main.py`** | **Entry Point**. Responsible for data loading, model construction, training loop, and evaluation. |
| **`CHARM.py`** | **Core Model**. Contains `CHARM` network architecture (Intent Aggregation, Causal Module) and `CHARMLoss` (Tri-level BPR, InfoNCE, Substitution Penalty). |
| `Params.py` | **Configuration**. Defines all command-line arguments, default hyperparameters, and path settings. |
| `DataHandler.py` | Data Handler. Responsible for loading Pickle datasets, building DataLoaders, and supporting multi-behavior data sampling. |
| `BGNN.py` | Auxiliary Model. Implements a multi-behavior GNN, mainly used in conjunction with the meta-learning module. |
| `MV_Net.py` | Meta Network. Implements `MetaWeightNet` for generating dynamic loss weights. |
| `HesitationAwareRecModel.py` | (Legacy/Backup) Another implementation version of the hesitation-aware model; current main logic is in `CHARM.py`. |

## Quick Start

### 1. Data Preparation
Ensure datasets are located in the `datasets/<dataset_name>/` directory.
Supported format is `pickle` files, including training sets (`trn_<behavior>`) and test set files.

### 2. Training
Simplest run (using default Tmall dataset):
```bash
python main.py
```

Specify dataset and target:
```bash
python main.py --dataset Tmall 
```

### 3. Tuning Examples
Based on model convergence, the following parameters are recommended for adjustment (see `Params.py` for details):

**Reduce learning rate to prevent overfitting:**
```bash
python main.py --lr 1e-5 --meta_lr 5e-5
```

**Disable Meta-Learning (Pure CHARM mode):**
```bash
python main.py --disable_meta_learning
```

**Adjust Hesitation and Substitution Penalty weights:**
```bash
python main.py --hesitation_weight 1.0 --substitute_penalty 0.2
```


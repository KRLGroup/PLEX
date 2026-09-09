# PLEX

Official Paper: https://doi.org/10.1007/978-3-032-37664-0_28

A self-explainable Graph Neural Network architecture that uses **Polyadic Graded Modal Logic (PGML)** to support edge-aware reasoning via ternary predicates R(x, y, e). Supports graph classification, node classification, and link prediction tasks.

## Repository Structure

### Models
| File | Description |
|------|-------------|
| `model.py` | Teacher (Black-box) and logic (PLEX) models for **graph classification** |
| `model_node.py` | Models for **node classification** |
| `model_link.py` | Models for **link prediction** |

### Training
| File | Description |
|------|-------------|
| `train_baseline.py` | Train teacher model (graph classification) |
| `train_baseline_node.py` | Train teacher model (node classification) |
| `train_baseline_link.py` | Train teacher model (link prediction) |
| `train_logic.py` | Train PLEX (graph classification) |
| `train_logic_node.py` | Train PLEX (node classification) |
| `train_logic_link.py` | Train PLEX (link prediction) |

### Hyperparameter Optimization
| File | Description |
|------|-------------|
| `optimize_baseline.py` | Hyperparameter search for teacher models on **graph classification**|
| `optimize_baseline_node.py` | Hyperparameter search for teacher models on **node classification**|
| `optimize_baseline_link.py` | Hyperparameter search for teacher models on **link prediction**|
| `optimize_logic.py` | Hyperparameter search for PLEX on **graph classififcation**|
| `optimize_logic_node.py` | Hyperparameter search for PLEX on **node classification**|
| `optimize_logic_link.py` | Hyperparameter search for PLEX on **link prediction**|

### Explainability & Analysis
| File | Description |
|------|-------------|
| `LayerWiseRules(GC).ipynb` | Extract logic rules from trained model (graph classification) |
| `LayerWiseRules(NC).ipynb` | Extract logic rules (node classification) |
| `LayerWiseRules(LP).ipynb` | Extract logic rules (link prediction) |
| `RuleActivations.py` | Visualize rule activations |
| `HoyerRuleStatistics.py` | Hoyer sparsity statistics and plots |

For the dataset Aromatic-Carbon there are different files for training, optimization and explainability.

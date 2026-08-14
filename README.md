# Notes on Topological Deep Learning

In the Jupyter notebooks included in this repository, we implement neural network models for classifying random graphs and hypergraphs based on their underlying generative models.

## Contents

| Notebook | Task |
|---|---|
| [`random_graph_gnn_classification.ipynb`](random_graph_gnn_classification.ipynb) | Classify which random **graph** model generated a graph. |
| [`hypergraph_hgnn_classification.ipynb`](hypergraph_hgnn_classification.ipynb) | Classify which random **hypergraph** model generated a hypergraph. |

### 1. GNN classification of random graph models

Generates 600 graphs of 20-40 nodes from four models (Erdős–Rényi, Barabási–Albert, Watts–Strogatz, and random geometric) and trains a two-layer GCN
(node features -> GCNConv -> global mean pooling -> MLP) to recover the generating model. Node features are degree and local clustering coefficient only.

# Notes on Topological Deep Learning

In the Jupyter notebooks included in this repository, we implement neural network models for classifying random graphs and hypergraphs based on their underlying generative models. In both cases, we implemented the neural networks using the [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/) library.

## Contents

| Notebook | Task |
|---|---|
| [`random_graph_gnn_classification.ipynb`](random_graph_gnn_classification.ipynb) | Classify which random **graph** model generated a graph. |
| [`random_hypergraph_hgnn_classification.ipynb`](random_hypergraph_hgnn_classification.ipynb) | Classify which random **hypergraph** model generated a hypergraph. |

### 1. Graph Neural Networks for Random Graph Model Classification

In this notebook, we generate 600 graphs of 20-40 nodes from four models (Erdős–Rényi, Barabási–Albert, Watts–Strogatz, and random geometric) and train a two-layer GCN (`GCNConv`) (node features -> GCNConv -> global mean pooling -> MLP) to recover the generating model. Node features are degree and local clustering coefficient only.


### 2. Hypergraph Neural Networks for Random Hypergraph Model Classification

In this notebook, we generate 600 hypergraphs of 20–40 nodes from four models (uniform ER, hypergraph preferential attachment (HPA), hypergraph SBM, and random geometric hypergraphs (HGEO)). All four use the same hyperedge-count rule and the same hyperedge-size distribution, so the models are indistinguishable from size statistics alone, and the classifier must use genuine incidence structure. The model is a two-layer `HypergraphConv` network, on five local node features.

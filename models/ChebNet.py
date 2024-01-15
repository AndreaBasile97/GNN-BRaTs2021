import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GATConv, GraphConv
from dgl.nn.pytorch.conv import SAGEConv
from dgl.nn.pytorch.conv import ChebConv


class ChebNet(nn.Module):
    def __init__(self, in_feats, layer_sizes, n_classes, k, dropout):
        super().__init__()
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p=dropout)

        # Input layer
        self.layers.append(ChebConv(in_feats, layer_sizes[0], k))

        # Hidden layers
        for i in range(1, len(layer_sizes)):
            self.layers.append(ChebConv(layer_sizes[i-1], layer_sizes[i], k))

        # Output layer
        self.layers.append(ChebConv(layer_sizes[-1], n_classes, k))

    def forward(self, g, inputs):
        h = inputs
        for i, layer in enumerate(self.layers):
            h = layer(g, h)
            if i != len(self.layers) - 1: # No activation and dropout on the output layer
                h = F.relu(h)
                h = self.dropout(h)
        return h
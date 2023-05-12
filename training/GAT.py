import torch
import torch.nn as nn
import torch.optim as optim
import dgl.nn as dglnn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(GATLayer, self).__init__()
        self.gat = dglnn.GATConv(in_feats=in_features, out_feats=out_features, num_heads=num_heads, allow_zero_in_degree=True)

    def forward(self, graph, inputs):
        return self.gat(graph, inputs)

class GATDummy(nn.Module):
    def __init__(self, in_features, hidden_features, num_heads, num_classes):
        super(GATDummy, self).__init__()
        self.layer1 = GATLayer(in_features, hidden_features, num_heads)
        self.layer2 = GATLayer(hidden_features * num_heads, hidden_features, 2)
        self.layer3 = GATLayer(hidden_features * 2, hidden_features, 2)
        self.layer4 = GATLayer(hidden_features * 2, hidden_features, 1)
        self.layer5 = GATLayer(hidden_features, num_classes, 1)

    def forward(self, graph, inputs):
        h = self.layer1(graph, inputs)
        h = F.elu(h)
        h = h.view(h.shape[0], -1)
        h = self.layer2(graph, h)
        h = F.elu(h)
        h = h.view(h.shape[0], -1)
        h = self.layer3(graph, h)
        h = F.elu(h)
        h = h.view(h.shape[0], -1)
        h = self.layer4(graph, h)
        h = F.elu(h)
        h = h.view(h.shape[0], -1)
        h = self.layer5(graph, h)
        return h.view(h.shape[0], -1)
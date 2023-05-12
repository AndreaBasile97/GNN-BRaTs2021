import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, num_heads):
        super(GATLayer, self).__init__()
        self.gat = dglnn.GATConv(in_feats=in_features, out_feats=out_features, num_heads=num_heads, allow_zero_in_degree=True)

    def forward(self, graph, inputs):
        return self.gat(graph, inputs)

class GAT(nn.Module):
    def __init__(self, in_features):
        super(GAT, self).__init__()
        self.layer0 = GATLayer(in_features, 1024, 1)
        self.layer1 = GATLayer(1024, 2048, 1)
        self.layer2 = GATLayer(2048, 2048, 1)
        self.layer3 = GATLayer(2048, 1024, 1)
        self.layer4 = GATLayer(1024, 4, 1)


    def forward(self, graph, inputs):
        h = inputs
        h = self.layer0(graph, h)
        h = F.elu(h.view(h.shape[0], -1))
        h = self.layer1(graph, h)
        h = F.elu(h.view(h.shape[0], -1))
        h = self.layer2(graph, h)
        h = F.elu(h.view(h.shape[0], -1))
        h = self.layer3(graph, h)
        h = F.elu(h.view(h.shape[0], -1))
        h = self.layer4(graph, h)
        return h.view(h.shape[0], -1)
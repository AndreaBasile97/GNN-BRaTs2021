from dgl.nn import GNNExplainer
import torch


def explain_graph(model, graph, features):
    features = torch.from_numpy(features).float()
    explainer = GNNExplainer(model, num_hops=1)
    feat_mask, edge_mask = explainer.explain_graph(graph, features)
    return feat_mask, edge_mask
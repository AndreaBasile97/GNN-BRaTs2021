from dgl.nn import GNNExplainer



def explain_graph(model, graph, features):
    explainer = GNNExplainer(model, num_hops=1)
    feat_mask, edge_mask = explainer.explain_graph(graph, features)
    return feat_mask, edge_mask
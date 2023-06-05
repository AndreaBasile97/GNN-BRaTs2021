from dgl.nn import GNNExplainer
import torch
import numpy as np


'''
The output of explanation is:

feat_mask (dict[str, Tensor])  The dictionary that associates the learned node feature importance masks (values) with the respective node types (keys). 
The masks are of shape (Dt) , where Dt is the node feature size for node type t. The values are within range (0,1). The higher, the more important.

edge_mask (dict[Tuple[str], Tensor]) The dictionary that associates the learned edge importance masks (values) with the respective canonical edge types (keys).
The masks are of shape (Et), where Et is the number of edges for canonical edge type t in the graph. The values are within range (0,1). The higher, the more important.

'''

def explain_graph(model, graph, features, labels):
    features = torch.from_numpy(features).float()
    explainer = GNNExplainer(model, num_hops=1)
    class_feat_masks = {}
    class_edge_masks = {}
    classes = np.unique(labels)
    for c in classes:
        indices = (labels == c).nonzero()[0] # Access the first element of the tuple
        class_graph = graph.subgraph(indices.tolist())  # converting to list as DGL requires list
        class_features = features[indices]
        feat_mask, edge_mask = explainer.explain_graph(class_graph, class_features)
        class_feat_masks[c] = feat_mask
        class_edge_masks[c] = edge_mask
    return class_feat_masks, class_edge_masks

'''
The features are take from each scan modality in this exact order
_flair.nii.gz","_t1.nii.gz","_t1ce.nii.gz","_t2.nii.gz
'''
    



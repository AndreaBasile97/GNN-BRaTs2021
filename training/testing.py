import pickle 
import torch
from training.explainer import explain_multiple_graphs


pickle_dataset_path = ''
with open(pickle_dataset_path, 'rb') as f:
    dataset = pickle.load(f)

from models.ChebNet import GraphSage

test_model = GraphSage(in_feats = 20, layer_sizes = [256, 256, 256, 256, 256, 256], n_classes = 4, aggregator_type='pool', dropout=0.2)

test_model.load_state_dict(torch.load('YOUR_SAVED_MODEL.pth'))


# class_node_masks = explain_nodes_by_class(test_model, grafo_esempio, features, labels)
# print(class_node_masks)

class_node_masks_all_graphs = explain_multiple_graphs(test_model, dataset, num_graphs=1, nodes_per_class=50, random_seed=23)
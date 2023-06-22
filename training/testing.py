from training.utilities import generate_tumor_segmentation_from_graph, predict, _3Dplotter
import pickle 
import torch
import nibabel as nib
from training.explainer import explain_nodes_by_class, explain_multiple_graphs
import networkx as nx
import json
import dgl

with open('pickle_dataset/full_dataset_with_id_01.pickle', 'rb') as f:
    dataset = pickle.load(f)


grafo_esempio = dataset[0][0]
features = dataset[0][1]
labels = dataset[0][2]
id = dataset[0][3]

print(labels)
print(id)

from models.GATSage import GraphSage

test_model = GraphSage(in_feats = 20, layer_sizes = [256, 256, 256, 256, 256, 256], n_classes = 4, aggregator_type='pool', dropout=0.2)

test_model.load_state_dict(torch.load('model_epoch_25_graphSage.pth'))


# class_node_masks = explain_nodes_by_class(test_model, grafo_esempio, features, labels)
# print(class_node_masks)

class_node_masks_all_graphs = explain_multiple_graphs(test_model, dataset, num_graphs=1, nodes_per_class=50, random_seed=23)

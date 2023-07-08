from training.utilities import generate_tumor_segmentation_from_graph, predict, _3Dplotter
import pickle 
import torch
import nibabel as nib
from training.explainer import explain_nodes_by_class, explain_multiple_graphs
import networkx as nx
import json
import dgl
import random

with open('pickle_dataset/full_dataset_with_id_01.pickle', 'rb') as f:
    dataset = pickle.load(f)

random.seed(42)  # Set the random seed to ensure reproducibility

# Shuffle the dataset
random.shuffle(dataset)

# Calculate the indices for splitting
train_split = int(len(dataset) * 0.7)
val_split = int(len(dataset) * 0.9)  # This is 90% because we're taking 20% of the remaining data after the training split

# Split the data
train_data = dataset[:train_split]
val_data = dataset[train_split:val_split]
test_data = dataset[val_split:]  # The remaining 10% of the data

from models.GATSage import ChebNet

test_model = ChebNet(in_feats = 20, layer_sizes = [512, 512, 512, 512], n_classes = 4, k=3, dropout=0.2)

print('loading...')
test_model.load_state_dict(torch.load('model_epoch_91.pth'))
print('ok')

print(f'Explaining {len(test_data)} graphs...')
# class_node_masks = explain_nodes_by_class(test_model, grafo_esempio, features, labels)
class_node_masks_all_graphs = explain_multiple_graphs(test_model, test_data, num_graphs=len(test_data), nodes_per_class=50)

    


print('explaining end.')
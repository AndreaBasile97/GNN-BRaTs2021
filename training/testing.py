from training.utilities import generate_tumor_segmentation_from_graph, predict, _3Dplotter
import pickle 
import torch
import nibabel as nib
from training.explainer import explain_graph
import networkx as nx
import json
import dgl

with open('full_dataset_with_id_006.pickle', 'rb') as f:
    dataset = pickle.load(f)


grafo_esempio = dataset[1][0]
features = dataset[1][1]
id = dataset[1][3]

print(id)

from models.GATSage import GraphSage

test_model = GraphSage(in_feats = 20, layer_sizes = [256, 256, 256, 256, 256, 256], n_classes = 4, aggregator_type='pool', dropout=0)

test_model.load_state_dict(torch.load('model_epoch_25.pth'))

# predicted_labels = predict(grafo_esempio, features, test_model)

# slic = nib.load(f'datasets/DGL_graphs/train_006/BraTS2021_{id}/BraTS2021_{id}_supervoxels.nii.gz').get_fdata()
# labels = nib.load(f'datasets/DGL_graphs/train_006/BraTS2021_{id}/BraTS2021_{id}_label.nii.gz').get_fdata()


# tumor = generate_tumor_segmentation_from_graph(predicted_labels, slic)

# _3Dplotter(tumor)
# _3Dplotter(labels)



# Explainer

features_mask, edge_mask = explain_graph(test_model, grafo_esempio, features)
print(max(edge_mask))
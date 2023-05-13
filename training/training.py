import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.data
from dgl.nn import GATConv
import networkx as nx
import nibabel as nib
import pickle
import matplotlib.pyplot as plt
from training.utilities import load_dgl_graphs_from_bin, prune_graphs, compute_average_weights, compute_metrics, standardize_features, minmax
import torch.optim as optim
import os
import warnings

# Ignore UserWarning related to TypedStorage deprecation
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
os.environ["DGLBACKEND"] = "pytorch"

dgl_train_graphs, t_ids = load_dgl_graphs_from_bin('graphs_module/DGL_graphs/FIXED_TRAIN_dgl_graphs_fix.bin', 'graphs_module/DGL_graphs/FIXED_TRAIN_patient_ids_fix.pkl')
# dgl_validation_graphs, v_ids = load_dgl_graphs_from_bin('../graphs_module/DGL_graphs/val_dgl_graphs_fix.bin', '../graphs_module/DGL_graphs/val_patient_ids_fix.pkl')
# dgl_test_graphs, test_ids = load_dgl_graphs_from_bin('../graphs_module/DGL_graphs/test_dgl_graphs_fix.bin', '../graphs_module/DGL_graphs/test_patient_ids_fix.pkl')

from evaluations.compute_metrics import calculate_node_dices
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np

def train(dgl_train_graphs, dgl_validation_graphs, model, loss_w):

    lr0 = 0.001  # Initial learning rate
    weight_decay = 0.0001  # Weight decay for AdamW
    lambda_ = 0.98  # Decay factor for learning rate

    # Define the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=lr0, weight_decay=weight_decay)

    # Define the learning rate scheduler
    scheduler = ExponentialLR(optimizer, gamma=lambda_)

    # optimizer = optim.AdamW(model.parameters(), lr=0.0001) 
    # scheduler = ExponentialLR(optimizer, gamma=0.0001)

    print('Training started...')

    for e in range(100):
        model.train()

        total_loss = 0
        total_recall = 0
        total_precision = 0
        total_train_f1 = 0

        for g in dgl_train_graphs:

            # Get the features, labels, and masks for the current graph
            features = g.ndata["feat"].float()

            labels = g.ndata["label"]
            labels = labels - 1

            # Forward pass
            logits = model(g, features)

            # Compute prediction
            pred = logits.argmax(1)

            # Compute loss with class weights
            loss = F.cross_entropy(logits, labels, weight=loss_w)

            pred = pred + 1
            labels = labels + 1

            recall, precision, f1 = compute_metrics(pred, labels)

            # Accumulate loss and accuracy values for this graph
            total_loss += loss.item()
            total_recall += recall.item()
            total_precision += precision.item()
            total_train_f1 += f1.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
        # Apply the learning rate scheduler
        scheduler.step()

        # Compute average loss and accuracy values across all graphs
        avg_loss = total_loss / len(dgl_train_graphs)
        avg_train_precision = total_precision / len(dgl_train_graphs)
        avg_train_f1 = total_train_f1 / len(dgl_train_graphs)

        # # Validation loop
        # total_val_recall = 0
        # total_val_precision = 0
        # total_val_f1 = 0
        # total_val_loss = 0

        # model.eval()
        # with torch.no_grad():
        #     for g in dgl_validation_graphs:
        #         # Get the features, labels, and masks for the current graph
        #         features = g.ndata["feat"].float()

        #         val_labels = g.ndata["label"]
        #         val_labels = val_labels -1

        #         # Forward pass
        #         logits = model(g, features)

        #         # Compute prediction
        #         val_pred = logits.argmax(1)

        #         # Compute loss with class weights
        #         val_loss = F.cross_entropy(logits, val_labels)  

        #         val_pred = val_pred + 1
        #         val_labels = val_labels + 1
        #         val_recall, val_precision, val_f1_score = compute_metrics(val_pred, val_labels)

        #         # Accumulate metrics values for this validation graph
        #         total_val_loss += val_loss.item()
        #         total_val_recall += val_recall.item()
        #         total_val_precision += val_precision.item()
        #         total_val_f1 += val_f1_score.item()

        # avg_val_loss = total_val_loss / len(dgl_train_graphs)
        # avg_val_precision = total_val_precision / len(dgl_validation_graphs)
        # avg_val_f1 = total_val_f1 / len(dgl_validation_graphs)

        # if e % 5 == 0:
        print(f"EPOCH {e} | loss: {avg_loss:.3f} | precision train WT: {avg_train_precision:.3f} | f1-score train WT: {avg_train_f1:.3f}|")   
            #   |val_loss:{avg_val_loss:.3f} | precision val WT: {avg_val_precision:.3f} | f1-score WT: {avg_val_f1:.3f} ")



# from GAT import GATDummy
from models.GATSage import GATSage
# from GCN import GCN
# from GATPaper import GAT



avg_weights = compute_average_weights(dgl_train_graphs)
pruned_train_graphs = prune_graphs(dgl_train_graphs)

print(f'CrossEntropyLoss weights: {avg_weights}')


# # FEATURES - HIDDEN LAYERS DIM - HEADS - CLASSES
# model = GATDummy(20, 256, 4, 4)

# # modelGCN = GCN(20, 2048, 4)

# trained_model = train(pruned_train_graphs[:1], pruned_train_graphs[:1], model, avg_weights)


# Define GAT parameters
in_feats = 20
layer_sizes = [256, 256, 256]
n_classes = 4
heads = [8, 8, 8]
residuals = [True, True, True]

# Create GAT model
model = GATSage(in_feats, layer_sizes, n_classes, heads, residuals)
trained_model = train(dgl_train_graphs[:1], dgl_train_graphs[:1], model, avg_weights)


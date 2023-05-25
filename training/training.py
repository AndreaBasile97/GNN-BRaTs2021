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
import pandas as pd
# Ignore UserWarning related to TypedStorage deprecation
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
os.environ["DGLBACKEND"] = "pytorch"

dgl_train_graphs, t_ids = load_dgl_graphs_from_bin('training/DGL_graphs/TRAIN_dgl_graphs_fix2.bin', 'training/DGL_graphs/TRAIN_patient_ids_fix2.pkl')
# dgl_validation_graphs, v_ids = load_dgl_graphs_from_bin('../graphs_module/DGL_graphs/val_dgl_graphs_fix.bin', '../graphs_module/DGL_graphs/val_patient_ids_fix.pkl')
# dgl_test_graphs, test_ids = load_dgl_graphs_from_bin('../graphs_module/DGL_graphs/test_dgl_graphs_fix.bin', '../graphs_module/DGL_graphs/test_patient_ids_fix.pkl')

from evaluations.compute_metrics import calculate_node_dices
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np

def train(dgl_train_graphs, dgl_validation_graphs, model, loss_w):

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)


    print('Training started...')

    metrics = []

    for e in range(500):
        model.train()

        total_loss = 0
        total_recall = 0
        total_precision = 0
        total_train_f1 = 0

        for g in dgl_train_graphs:

            # Get the features, labels, and masks for the current graph
            features = g.ndata["feat"]

            labels = g.ndata["label"]

            # 1,2,3,4 -> 0,1,2,3
            labels = labels - 1

            # Forward pass
            logits = model(g, features)

            # Compute prediction
            pred = logits.argmax(1)

            # Compute loss with class weights
            loss = F.cross_entropy(logits, labels, weight=loss_w)

            # 0,1,2,3 -> 1,2,3,4
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

        scheduler.step()

        # Compute average loss and accuracy values across all graphs
        avg_loss = total_loss / len(dgl_train_graphs)
        avg_train_precision = total_precision / len(dgl_train_graphs)
        avg_train_f1 = total_train_f1 / len(dgl_train_graphs)

        # Validation loop
        total_val_recall = 0
        total_val_precision = 0
        total_val_f1 = 0
        total_val_loss = 0

        model.eval()
        with torch.no_grad():
            for g in dgl_validation_graphs:
                # Get the features, labels, and masks for the current graph
                features = g.ndata["feat"].float()

                val_labels = g.ndata["label"]
                val_labels = val_labels -1

                # Forward pass
                logits = model(g, features)

                # Compute prediction
                val_pred = logits.argmax(1)

                # Compute loss with class weights
                val_loss = F.cross_entropy(logits, val_labels)  

                val_pred = val_pred + 1
                val_labels = val_labels + 1
                val_recall, val_precision, val_f1_score = compute_metrics(val_pred, val_labels)

                # Accumulate metrics values for this validation graph
                total_val_loss += val_loss.item()
                total_val_recall += val_recall.item()
                total_val_precision += val_precision.item()
                total_val_f1 += val_f1_score.item()

        avg_val_loss = total_val_loss / len(dgl_train_graphs)
        avg_val_precision = total_val_precision / len(dgl_validation_graphs)
        avg_val_f1 = total_val_f1 / len(dgl_validation_graphs)


        metrics.append({
                    'epoch': e,
                    'loss': avg_loss,
                    'precision_train_WT': avg_train_precision,
                    'f1_score_train_WT': avg_train_f1,
                    'val_loss': avg_val_loss,
                    'precision_val_WT': avg_val_precision,
                    'f1_score_val_WT': avg_val_f1
                })


        # if e % 5 == 0:
        print(f"EPOCH {e} | loss: {avg_loss:.3f} | precision train WT: {avg_train_precision:.3f} | f1-score train WT: {avg_train_f1:.3f}||val_loss:{avg_val_loss:.3f} | precision val WT: {avg_val_precision:.3f} | f1-score WT: {avg_val_f1:.3f} ")

    # Save metrics to a CSV file
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv('training_metrics.csv', index=False)



from models.GATSage import GATSage

avg_weights = compute_average_weights(dgl_train_graphs)

print(f'CrossEntropyLoss weights: {avg_weights}')

# Define GAT parameters
in_feats = 20
layer_sizes = [256, 256, 256, 256, 1024]
n_classes = 4
heads = [2, 2, 2, 2, 2]
residuals = [True, True, True, True, True]

# Create GAT model
model = GATSage(in_feats, layer_sizes, n_classes, heads, residuals)
trained_model = train(dgl_train_graphs, dgl_train_graphs, model, avg_weights)


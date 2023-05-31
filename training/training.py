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

dgl_train_graphs, t_ids = load_dgl_graphs_from_bin('training/DGL_graphs/train_dgl_graphs.bin', 'training/DGL_graphs/train_patient_ids.pkl')
dgl_validation_graphs, v_ids = load_dgl_graphs_from_bin('training/DGL_graphs/val_dgl_graphs.bin', 'training/DGL_graphs/val_patient_ids.pkl')
# dgl_test_graphs, test_ids = load_dgl_graphs_from_bin('../graphs_module/DGL_graphs/test_dgl_graphs_fix.bin', '../graphs_module/DGL_graphs/test_patient_ids_fix.pkl')

def create_batches(graph_list, patient_ids, batch_size=6):
    print('creating batches...')
    batched_graphs = []
    batched_patient_ids = []  # Add a list to store the batched patient IDs
    num_graphs = len(graph_list)

    for idx in range(0, num_graphs, batch_size):
        start = idx
        end = min(idx + batch_size, num_graphs)
        batch = graph_list[start:end]
        batched_graph = dgl.batch(batch)
        batched_graphs.append(batched_graph)
        batched_patient_ids.append(patient_ids[start:end])


    return batched_graphs, batched_patient_ids

dgl_train_batch_graphs, train_batched_ids = create_batches(dgl_train_graphs, t_ids)
dgl_val_batch_graphs, val_batched_ids = create_batches(dgl_validation_graphs, v_ids)


from evaluations.compute_metrics import calculate_node_dices
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np

# def train(dgl_train_graphs, dgl_validation_graphs, model, loss_w):

#     # Define the optimizer
#     optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
#     scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)


#     print('Training started...')

#     metrics = []

#     for e in range(500):
#         model.train()

#         total_loss = 0
#         total_recall = 0
#         total_precision = 0
#         total_train_f1 = 0

#         for g in dgl_train_graphs:

#             # Get the features, labels, and masks for the current graph
#             features = g.ndata["feat"]

#             labels = g.ndata["label"]

#             # 1,2,3,4 -> 0,1,2,3
#             labels = labels - 1

#             # Forward pass
#             logits = model(g, features)

#             # Compute prediction
#             pred = logits.argmax(1)

#             # Compute loss with class weights
#             loss = F.cross_entropy(logits, labels, weight=loss_w)

#             # 0,1,2,3 -> 1,2,3,4
#             pred = pred + 1
#             labels = labels + 1
 
#             recall, precision, f1 = compute_metrics(pred, labels)

#             # Accumulate loss and accuracy values for this graph
#             total_loss += loss.item()
#             total_recall += recall.item()
#             total_precision += precision.item()
#             total_train_f1 += f1.item()
            
#             # Backward pass
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#         scheduler.step()

#         # Compute average loss and accuracy values across all graphs
#         avg_loss = total_loss / len(dgl_train_graphs)
#         avg_train_precision = total_precision / len(dgl_train_graphs)
#         avg_train_f1 = total_train_f1 / len(dgl_train_graphs)

#         # Validation loop
#         total_val_recall = 0
#         total_val_precision = 0
#         total_val_f1 = 0
#         total_val_loss = 0

#         model.eval()
#         with torch.no_grad():
#             for g in dgl_validation_graphs:
#                 # Get the features, labels, and masks for the current graph
#                 features = g.ndata["feat"].float()

#                 val_labels = g.ndata["label"]
#                 val_labels = val_labels -1

#                 # Forward pass
#                 logits = model(g, features)

#                 # Compute prediction
#                 val_pred = logits.argmax(1)

#                 # Compute loss with class weights
#                 val_loss = F.cross_entropy(logits, val_labels)  

#                 val_pred = val_pred + 1
#                 val_labels = val_labels + 1
#                 val_recall, val_precision, val_f1_score = compute_metrics(val_pred, val_labels)

#                 # Accumulate metrics values for this validation graph
#                 total_val_loss += val_loss.item()
#                 total_val_recall += val_recall.item()
#                 total_val_precision += val_precision.item()
#                 total_val_f1 += val_f1_score.item()

#         avg_val_loss = total_val_loss / len(dgl_train_graphs)
#         avg_val_precision = total_val_precision / len(dgl_validation_graphs)
#         avg_val_f1 = total_val_f1 / len(dgl_validation_graphs)


#         metrics.append({
#                     'epoch': e,
#                     'loss': avg_loss,
#                     'precision_train_WT': avg_train_precision,
#                     'f1_score_train_WT': avg_train_f1,
#                     'val_loss': avg_val_loss,
#                     'precision_val_WT': avg_val_precision,
#                     'f1_score_val_WT': avg_val_f1
#                 })


#         # if e % 5 == 0:
#         print(f"EPOCH {e} | loss: {avg_loss:.3f} | precision train WT: {avg_train_precision:.3f} | f1-score train WT: {avg_train_f1:.3f}||val_loss:{avg_val_loss:.3f} | precision val WT: {avg_val_precision:.3f} | f1-score WT: {avg_val_f1:.3f} ")

#     # Save metrics to a CSV file
#     df_metrics = pd.DataFrame(metrics)
#     df_metrics.to_csv('training_metrics.csv', index=False)



from tqdm import tqdm

def train_batches(dgl_train_graphs, dgl_validation_graphs, model, loss_w):
    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    print('Training started...')
    metrics = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    patience = 3  # number of epochs to wait for improvement before stopping
    best_val_loss = float('inf')
    wait = 0

    for e in range(300):
        model.train()

        total_loss = 0
        total_f1_wt = 0
        total_f1_ct = 0
        total_f1_et = 0

        for batched_graph in tqdm(dgl_train_graphs, desc=f"Training epoch {e}"):
            batched_graph = batched_graph.to(device)
            features = batched_graph.ndata["feat"].to(device)
            labels = (batched_graph.ndata["label"] - 1).to(device)  # 1,2,3,4 -> 0,1,2,3

            logits = model(batched_graph, features)
            pred = logits.argmax(1)

            loss = F.cross_entropy(logits, labels, weight=loss_w.to(device))

            pred = pred + 1  # 0,1,2,3 -> 1,2,3,4
            labels = labels + 1

            f1_wt, f1_ct, f1_et = calculate_node_dices(pred, labels)
            total_loss += loss.item()
            total_f1_wt += f1_wt
            total_f1_ct += f1_ct
            total_f1_et += f1_et

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        avg_loss = total_loss / len(dgl_train_graphs)
        avg_train_f1_wt = total_f1_wt / len(dgl_train_graphs)
        avg_train_f1_ct = total_f1_ct / len(dgl_train_graphs)
        avg_train_f1_et = total_f1_et / len(dgl_train_graphs)

        total_val_f1_wt = 0
        total_val_f1_ct = 0
        total_val_f1_et = 0
        total_val_loss = 0

        model.eval()
        with torch.no_grad():
            for batched_graph in tqdm(dgl_validation_graphs, desc=f"Validation epoch {e}"):
                batched_graph = batched_graph.to(device)
                features = batched_graph.ndata["feat"].to(device)
                val_labels = (batched_graph.ndata["label"] - 1).to(device) 

                logits = model(batched_graph, features)
                val_pred = logits.argmax(1)

                val_loss = F.cross_entropy(logits, val_labels) 
                val_pred = val_pred + 1  # 0,1,2,3 -> 1,2,3,4
                val_labels = val_labels + 1

                val_f1_wt, val_f1_ct, val_f1_et = calculate_node_dices(val_pred, val_labels)

                total_val_loss += val_loss.item()
                total_val_f1_wt += val_f1_wt
                total_val_f1_ct += val_f1_ct
                total_val_f1_et += val_f1_et

        avg_val_loss = total_val_loss / len(dgl_validation_graphs)
        avg_val_f1_wt = total_val_f1_wt / len(dgl_validation_graphs)
        avg_val_f1_ct = total_val_f1_ct / len(dgl_validation_graphs)
        avg_val_f1_et = total_val_f1_et / len(dgl_validation_graphs)

        metrics.append({
            'epoch': e,
            'loss': avg_loss,
            'f1_score_train_WT': avg_train_f1_wt,
            'f1_score_train_CT': avg_train_f1_ct,
            'f1_score_train_ET': avg_train_f1_et,
            'val_loss': avg_val_loss,
            'f1_score_val_WT': avg_val_f1_wt,
            'f1_score_val_CT': avg_val_f1_ct,
            'f1_score_val_ET': avg_val_f1_et
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping...")
                break

        print(f"EPOCH {e} | loss: {avg_loss:.3f} | f1-score train WT: {avg_train_f1_wt:.3f} | f1-score train CT: {avg_train_f1_ct:.3f} | f1-score train ET: {avg_train_f1_et:.3f} || val_loss:{avg_val_loss:.3f} | f1-score val WT: {avg_val_f1_wt:.3f} | f1-score val CT: {avg_val_f1_ct:.3f} | f1-score val ET: {avg_val_f1_et:.3f} ")

    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv('training_metrics.csv', index=False)

from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def compute_metrics_batch(pred, labels):
    """
    Compute metrics for batch.
    Args:
        pred (torch.Tensor): The model's predictions.
        labels (torch.Tensor): The ground truth labels.
    Returns:
        recall (float): The recall score.
        precision (float): The precision score.
        f1 (float): The f1 score.
    """
    # Convert tensors to numpy arrays
    pred = pred.cpu().numpy()
    labels = labels.cpu().numpy()

    # Convert all the values 1, 2, or 4 to 1 in both pred and labels
    wt_pred = np.where((pred == 1) | (pred == 2) | (pred == 4), 1, 0)
    wt_labels = np.where((labels == 1) | (labels == 2) | (labels == 4), 1, 0)


    # Convert all the values 1, or 4 to 1 in both pred and labels
    ct_pred = np.where((pred == 1) | (pred == 4), 1, 0)
    ct_labels = np.where((labels == 1) |(labels == 4), 1, 0)


    # Convert all the values 1, or 4 to 1 in both pred and labels
    et_pred = np.where((pred == 4), 1, 0)
    et_labels = np.where((labels == 4), 1, 0)


    # Filter out the 0s
    wt_pred = wt_pred[wt_labels!=0]
    wt_labels = wt_labels[wt_labels!=0]

    ct_pred = ct_pred[ct_labels!=0]
    ct_labels = ct_labels[ct_labels!=0]

    et_pred = et_pred[et_labels!=0]
    et_labels = et_labels[et_labels!=0]



    # Compute F1 scores for each class
    f1_wt = f1_score(wt_labels, wt_pred, average='macro', zero_division=0)
    f1_ct = f1_score(ct_labels, ct_pred, average='macro', zero_division=0)
    f1_et = f1_score(et_labels, et_pred, average='macro', zero_division=0)

    return torch.tensor(f1_wt), torch.tensor(f1_ct), torch.tensor(f1_et)



HEALTHY = 3
EDEMA = 4
NET = 1
ET = 2

# Calculate nodewise Dice score for WT, CT, and ET for a single brain.
# Expects two 1D vectors of integers.
def calculate_node_dices(preds, labels):
    p, l = preds, labels

    wt_preds = np.where(p == HEALTHY, 0, 1)
    wt_labs = np.where(l == HEALTHY, 0, 1)
    wt_dice = calculate_dice_from_logical_array(wt_preds, wt_labs)

    ct_preds = np.isin(p, [NET, ET]).astype(int)
    ct_labs = np.isin(l, [NET, ET]).astype(int)
    ct_dice = calculate_dice_from_logical_array(ct_preds, ct_labs)

    at_preds = np.where(p == ET, 1, 0)
    at_labs = np.where(l == ET, 1, 0)
    at_dice = calculate_dice_from_logical_array(at_preds, at_labs)

    return wt_dice, ct_dice, at_dice

# Each tumor region (WT, CT, ET) is binarized for both the prediction and ground truth 
# and then the overlapping volume is calculated.
def calculate_dice_from_logical_array(binary_predictions, binary_ground_truth):
    true_positives = np.logical_and(binary_predictions == 1, binary_ground_truth == 1)
    false_positives = np.logical_and(binary_predictions == 1, binary_ground_truth == 0)
    false_negatives = np.logical_and(binary_predictions == 0, binary_ground_truth == 1)
    tp, fp, fn = np.count_nonzero(true_positives), np.count_nonzero(false_positives), np.count_nonzero(false_negatives)
    # The case where no such labels exist (only really relevant for ET case).
    if (tp + fp + fn) == 0:
        return 1
    return (2 * tp) / (2 * tp + fp + fn)



from models.GATSage import GATSage

avg_weights = compute_average_weights(dgl_train_graphs)

print(f'CrossEntropyLoss weights: {avg_weights}')

# Define GAT parameters
in_feats = 20
layer_sizes = [256, 256, 256, 256, 256, 256, 256]
n_classes = 4
heads = [4, 4, 4, 4, 4, 4, 4]
residuals = [True, True, True, True, True, True, True]

# Create GAT model
model = GATSage(in_feats, layer_sizes, n_classes, heads, residuals)
trained_model = train_batches(dgl_train_batch_graphs, dgl_val_batch_graphs, model, avg_weights)


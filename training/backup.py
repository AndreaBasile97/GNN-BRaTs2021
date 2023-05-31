import torch
import torch.nn.functional as F
from training.utilities import compute_average_weights, calculate_node_dices, generate_dgl_dataset, create_batches, batch_dataset
import torch.optim as optim
import os
import warnings
import pandas as pd
import torch
import numpy as np
import pickle
import random
from tqdm import tqdm
from evaluations.compute_metrics import calculate_node_dices
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np
# Ignore UserWarning related to TypedStorage deprecation
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
os.environ["DGLBACKEND"] = "pytorch"




####### LOAD THE DATASET  AND SPLIT TRAIN - TEST - VAL ########
with open('full_dataset.pickle', 'rb') as f:
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

print(f"Train data size: {len(train_data)}")
print(f"Validation data size: {len(val_data)}")
print(f"Test data size: {len(test_data)}")
# dataset = generate_dgl_dataset('training/DGL_graphs/train/')

import itertools

def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

# Create your batches
train_batches = list(grouper(train_data, 6))
val_batches = list(grouper(val_data, 6))
test_batches = list(grouper(test_data, 6))



def train(dgl_train_graphs, dgl_validation_graphs, model, loss_w):

    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)


    print('Training started...')

    metrics = []
    patience = 10  # number of epochs to wait for improvement before stopping
    best_val_loss = float('inf')
    wait = 0

    for e in range(500):
        model.train()

        total_loss = 0
        total_f1_wt = 0
        total_f1_ct = 0
        total_f1_et = 0

        for g, feature, label in tqdm(dgl_train_graphs, desc=f"Training epoch {e}"):

            # Get the features, labels, and masks for the current graph
            features = torch.tensor(feature).float()

            labels = torch.tensor(label).long()

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
 
            f1_wt, f1_ct, f1_et = calculate_node_dices(pred, labels)
            total_loss += loss.item()
            total_f1_wt += f1_wt
            total_f1_ct += f1_ct
            total_f1_et += f1_et
            
            # Backward pass
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
            for g, feature, label in tqdm(dgl_validation_graphs, desc=f"Training epoch {e}"):
                # Get the features, labels, and masks for the current graph
                features = torch.tensor(feature).float()

                val_labels = torch.tensor(label).long()
                val_labels = val_labels -1

                # Forward pass
                logits = model(g, features)

                # Compute prediction
                val_pred = logits.argmax(1)

                # Compute loss with class weights
                val_loss = F.cross_entropy(logits, val_labels)  

                val_pred = val_pred + 1
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

        # if e % 5 == 0:
        print(f"EPOCH {e} | loss: {avg_loss:.3f} | f1-score train WT: {avg_train_f1_wt:.3f} | f1-score train CT: {avg_train_f1_ct:.3f} | f1-score train ET: {avg_train_f1_et:.3f} || val_loss:{avg_val_loss:.3f} | f1-score val WT: {avg_val_f1_wt:.3f} | f1-score val CT: {avg_val_f1_ct:.3f} | f1-score val ET: {avg_val_f1_et:.3f} ")

    # Save metrics to a CSV file
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv('training_metrics.csv', index=False)

    torch.save(model.state_dict(), f'model_epoch_{e}.pth')

import dgl
def train_batch(dgl_train_graphs, dgl_validation_graphs, model, loss_w):
    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    print('Training started...')

    metrics = []
    patience = 10  # number of epochs to wait for improvement before stopping
    best_val_loss = float('inf')
    wait = 0

    # Filter out None triplets
    dgl_train_graphs = [triplet for triplet in dgl_train_graphs if None not in triplet]
    dgl_validation_graphs = [triplet for triplet in dgl_validation_graphs if None not in triplet]

    for e in range(500):
        model.train()

        total_loss = 0
        total_f1_wt = 0
        total_f1_ct = 0
        total_f1_et = 0

        for batch in tqdm(dgl_train_graphs, desc=f"Training epoch {e}"):

            bg = dgl.batch([data[0] for data in batch])  # batched graph
            features = torch.cat([torch.tensor(data[1]).float() for data in batch], dim=0)  # concatenate features
            labels = torch.cat([torch.tensor(data[2]).long() - 1 for data in batch], dim=0)  # Offset the labels and concatenate

            # Forward pass
            logits = model(bg, features)

            # Compute prediction
            pred = logits.argmax(1)

            # Compute loss with class weights
            loss = F.cross_entropy(logits, labels, weight=loss_w)

            # 0,1,2,3 -> 1,2,3,4
            pred = pred + 1
            labels = labels + 1

            f1_wt, f1_ct, f1_et = calculate_node_dices(pred, labels)
            total_loss += loss.item()
            total_f1_wt += f1_wt
            total_f1_ct += f1_ct
            total_f1_et += f1_et

            # Backward pass
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
            for batch in tqdm(dgl_validation_graphs, desc=f"Validation epoch {e}"):

                bg = dgl.batch([data[0] for data in batch])  # batched graph
                features = torch.cat([torch.tensor(data[1]).float() for data in batch], dim=0)  # concatenate features
                labels = torch.cat([torch.tensor(data[2]).long() - 1 for data in batch], dim=0)  # Offset the labels and concatenate        

                # Forward pass
                logits = model(bg, features)

                # Compute prediction
                pred = logits.argmax(1)

                # Compute loss with class weights
                loss = F.cross_entropy(logits, labels, weight=loss_w)

                # 0,1,2,3 -> 1,2,3,4
                pred = pred + 1
                labels = labels + 1

                f1_wt, f1_ct, f1_et = calculate_node_dices(pred, labels)
                total_val_loss += loss.item()
                total_val_f1_wt += f1_wt
                total_val_f1_ct += f1_ct
                total_val_f1_et += f1_et

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

    # Save metrics to a CSV file
    df_metrics = pd.DataFrame(metrics)
    df_metrics.to_csv('training_metrics.csv', index=False)

    torch.save(model.state_dict(), f'model_epoch_{e}.pth')





from models.GATSage import GATSage

avg_weights = compute_average_weights(val_data)

print(f'CrossEntropyLoss weights: {avg_weights}')

# Define GAT parameters
in_feats = 20
layer_sizes = [256, 256, 256, 256, 256, 256]
n_classes = 4
heads = [4, 4, 4, 4, 4, 4]
residuals = [True, True, True, True, True, True]

# Create GAT model
model = GATSage(in_feats, layer_sizes, n_classes, heads, residuals)
trained_model = train_batch(train_batches, val_batches, model, avg_weights)
# trained_model = train(train_data, val_data, model, avg_weights)


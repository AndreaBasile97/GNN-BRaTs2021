import torch
import torch.nn.functional as F
from training.utilities import compute_average_weights, save_settings
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
from dotenv import load_dotenv
import datetime
import argparse

# Create the parser 
parser = argparse.ArgumentParser() 
# Add an argument for the model 
parser.add_argument('--model', type=str, required=True, help="Name of the model") 
# Parse the arguments 
args = parser.parse_args() 
# You can now access the model argument with args.model 
print(f'Training with model: {args.model}')


timestamp = datetime.datetime.now()

# Ignore UserWarning related to TypedStorage deprecation
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
os.environ["DGLBACKEND"] = "pytorch"


load_dotenv()
dataset_pickle_path = ''
model_metrics_path = f'metrics.csv'
model_backup_path = f'model.pth'

####### LOAD THE DATASET  AND SPLIT TRAIN - TEST - VAL ########
with open(dataset_pickle_path, 'rb') as f:
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


import dgl
def train_batch(timestamp, dgl_train_graphs, dgl_validation_graphs, model, loss_w, patience, val_lr, val_weight_decay, val_gamma):
    # Define the optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr = val_lr, weight_decay = val_weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = val_gamma)

    # timestamp = datetime.datetime.now()
    print(f'Training started at: {timestamp}')

    metrics = []
    best_val_loss = float('inf')
    wait = 0

    # Filter out None triplets
    dgl_train_graphs = [triplet for triplet in dgl_train_graphs if None not in triplet]
    dgl_validation_graphs = [triplet for triplet in dgl_validation_graphs if None not in triplet]

    for e in range(500):
        model.train()

        total_loss = 0
        total_dice_wt = 0
        total_dice_ct = 0
        total_dice_et = 0

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

            dice_wt, dice_ct, dice_et = calculate_node_dices(pred, labels)
            total_loss += loss.item()
            total_dice_wt += dice_wt
            total_dice_ct += dice_ct
            total_dice_et += dice_et

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        scheduler.step()

        avg_loss = total_loss / len(dgl_train_graphs)
        avg_train_dice_wt = total_dice_wt / len(dgl_train_graphs)
        avg_train_dice_ct = total_dice_ct / len(dgl_train_graphs)
        avg_train_dice_et = total_dice_et / len(dgl_train_graphs)

        total_val_dice_wt = 0
        total_val_dice_ct = 0
        total_val_dice_et = 0
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

                dice_wt, dice_ct, dice_et = calculate_node_dices(pred, labels)
                total_val_loss += loss.item()
                total_val_dice_wt += dice_wt
                total_val_dice_ct += dice_ct
                total_val_dice_et += dice_et

        avg_val_loss = total_val_loss / len(dgl_validation_graphs)
        avg_val_dice_wt = total_val_dice_wt / len(dgl_validation_graphs)
        avg_val_dice_ct = total_val_dice_ct / len(dgl_validation_graphs)
        avg_val_dice_et = total_val_dice_et / len(dgl_validation_graphs)

        metrics.append({
            'epoch': e,
            'loss': avg_loss,
            'dice_score_train_WT': avg_train_dice_wt,
            'dice_score_train_CT': avg_train_dice_ct,
            'dice_score_train_ET': avg_train_dice_et,
            'val_loss': avg_val_loss,
            'dice_score_val_WT': avg_val_dice_wt,
            'dice_score_val_CT': avg_val_dice_ct,
            'dice_score_val_ET': avg_val_dice_et
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print("Early stopping...")
                break

        print(f"EPOCH {e} | loss: {avg_loss:.3f} | dice-score train WT: {avg_train_dice_wt:.3f} | dice-score train CT: {avg_train_dice_ct:.3f} | dice-score train ET: {avg_train_dice_et:.3f} || val_loss:{avg_val_loss:.3f} | dice-score val WT: {avg_val_dice_wt:.3f} | dice-score val CT: {avg_val_dice_ct:.3f} | dice-score val ET: {avg_val_dice_et:.3f} ")

        # Save metrics to a CSV file
        df_metrics = pd.DataFrame(metrics)
        string_timestamp = timestamp.strftime("%Y%m%d-%H%M%S")
        df_metrics.to_csv(model_metrics_path, index=False)

    torch.save(model.state_dict(), model_backup_path)



from models.ChebNet import ChebNet

avg_weights = compute_average_weights(val_data)

print(f'CrossEntropyLoss weights: {avg_weights}')

# Define parameters
in_feats = 20
layer_sizes = [512, 512, 512]
n_classes = 4
heads = [8, 8, 8, 8, 8, 8]
residuals = [False, True, True, False, True, True]

patience = 10 # number of epochs to wait for improvement before stopping
lr = 0.0005
weight_decay = 0.0001 
gamma = 0.98

val_dropout = 0.2
val_feat_drop = 0.2
val_attn_drop = 0.2

# Create model
if args.model == 'Cheb':
    model = ChebNet(in_feats, layer_sizes, n_classes, k = 3, dropout = val_dropout)

save_settings(timestamp, model, patience, lr, weight_decay, gamma, args.model, heads, residuals, \
              val_dropout, layer_sizes, in_feats, n_classes, val_feat_drop, val_attn_drop, dataset_pickle_path, model_path = None)


trained_model = train_batch(timestamp, train_batches, val_batches, model, avg_weights, patience, lr, weight_decay, gamma)
# trained_model = train(train_data, val_data, model, avg_weights)

trained_model = train_batch(timestamp, train_batches, val_batches, model, avg_weights, patience, lr, weight_decay, gamma)
# trained_model = train(train_data, val_data, model, avg_weights)

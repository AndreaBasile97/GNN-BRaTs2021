import dgl
import dgl.data
import torch
from GAT import GATDummy
import torch.optim as optim
import torch.nn.functional as F
import warnings

# Ignore UserWarning related to TypedStorage deprecation
warnings.filterwarnings("ignore", category=UserWarning, message="TypedStorage is deprecated")
# Load the CORA dataset
dataset = dgl.data.CoraGraphDataset()

# Get the graph
graph = dataset[0]

# Prepare the training and validation sets
def prepare_data(graph, split_ratio=0.8):
    node_count = graph.number_of_nodes()
    train_node_count = int(node_count * split_ratio)

    train_mask = torch.zeros(node_count, dtype=torch.bool)
    train_mask[:train_node_count] = 1

    val_mask = torch.zeros(node_count, dtype=torch.bool)
    val_mask[train_node_count:] = 1

    graph.ndata['train_mask'] = train_mask
    graph.ndata['val_mask'] = val_mask

    return graph

# Prepare the data with an 80-20 split
graph = prepare_data(graph)

# Create the training and validation graph lists
train_graphs = [graph.subgraph(graph.ndata['train_mask'])]
val_graphs = [graph.subgraph(graph.ndata['val_mask'])]



import dgl
import dgl.data
import numpy as np
from sklearn.metrics import f1_score

def compute_metrics(pred, labels, mask):
    masked_pred = pred[mask]
    masked_labels = labels[mask]
    f1 = f1_score(masked_labels, masked_pred, average="micro")
    return f1

def train_cora(graph, model, epochs=100, lr=0.01):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for e in range(epochs):
        model.train()

        # Forward pass
        features = graph.ndata["feat"]
        logits = model(graph, features)

        # Compute loss
        labels = graph.ndata["label"]
        train_mask = graph.ndata["train_mask"]

        loss = F.cross_entropy(logits[train_mask], labels[train_mask])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Compute training metrics
        train_pred = logits.argmax(1)
        train_f1 = compute_metrics(train_pred, labels, train_mask)

        # Evaluate on the validation set
        model.eval()
        with torch.no_grad():
            val_mask = graph.ndata["val_mask"]
            val_pred = logits.argmax(1)
            val_f1 = compute_metrics(val_pred, labels, val_mask)

        print(f"EPOCH {e} | loss: {loss:.3f} | train_f1: {train_f1:.3f} | val_f1: {val_f1:.3f}")

# Load the CORA dataset
dataset = dgl.data.CoraGraphDataset()
graph = dataset[0]
graph = prepare_data(graph)

# Initialize the GAT model
in_features = graph.ndata["feat"].shape[1]
hidden_features = 8
num_heads = 8
num_classes = dataset.num_classes

model = GATDummy(in_features, hidden_features, num_heads, num_classes)

# Train the model
train_cora(graph, model)

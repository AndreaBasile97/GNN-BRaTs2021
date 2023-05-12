import numpy as np
import networkx as nx
import nibabel as nib
import pickle
import matplotlib.pyplot as plt
import torch
import dgl
import dgl.data
import plotly.graph_objs as go
import sklearn.metrics as metrics

def get_coordinates(tumor_seg, values=[1, 2, 4]):

    coordinates = {}
    
    for value in values:
        # Get the indices where the value is present in the image
        indices = np.where(tumor_seg == value)

        # Convert the indices to a list of tuples representing coordinates
        coords = list(zip(*indices))

        # Add the coordinates to the dictionary
        coordinates[value] = coords

    return coordinates


def get_patient_ids(paths):
    ids = []
    for path in paths:
        splitted_path = path.split("/")
        ids.append(splitted_path[-1].split("_")[1])
    if all(elem == ids[0] for elem in ids):
        return ids, True
    else:
        return ids, False
    

def get_supervoxel_values(slic_image, coordinates_dict):

    supervoxel_values = {}

    for value, coordinates in coordinates_dict.items():
        value_list = []
        for coord in coordinates:
            # Get the value of the supervoxel at the given coordinate
            supervoxel_value = slic_image[coord]

            # Add the value to the list
            value_list.append(supervoxel_value)
        
        # Add the list of supervoxel values to the dictionary
        supervoxel_values[value] = list(np.unique(value_list))

    return supervoxel_values


def assign_labels_to_graph(tumor_seg, slic_image, graph):

    coords = get_coordinates(tumor_seg)
    labels_supervoxel_dict = get_supervoxel_values(slic_image, coords)

    for label, supervoxel_list in labels_supervoxel_dict.items():
        for supervoxel in supervoxel_list:
            graph.nodes[str(int(supervoxel))]["label"] = label

    for n in graph.nodes():
        try:
            graph.nodes[n]["label"]
        except:
            graph.nodes[n]["label"] = 3
    
    return graph    


def generate_tumor_segmentation_from_graph(segmented_image, graph):
    # Create a dictionary to map node IDs to their labels
    node_label_map = {int(n): graph.nodes[n]['label'] for n in graph.nodes()}
    
    # Create a mask for labels to keep (1, 2, and 4)
    labels_to_keep = np.array([1, 2, 4])
    
    # Replace node IDs in segmented_image with their corresponding labels
    label_image = np.vectorize(node_label_map.get)(segmented_image)
    
    # Use np.isin to create a boolean mask for the labels we want to keep
    mask = np.isin(label_image, labels_to_keep)
    
    # Use np.where to create the segmented tumor image
    segmented_tumor = np.where(mask, label_image, 0)
    
    return segmented_tumor
    

def tensor_labels(segmented_image, labels_generated, empty_RAG, id_patient, save=False):
    R = assign_labels_to_graph(labels_generated, segmented_image, empty_RAG)
    tl = torch.tensor(list(nx.get_node_attributes(R, 'label').values()))
    if save==True:
       torch.save(tl, f'/content/drive/MyDrive/Tesi Progetto/tensor_labels/tensor_label_{id_patient}')
    return tl


def _3Dplotter(numpy_image):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    ax1.imshow(numpy_image[numpy_image.shape[0] // 2, :, :])
    ax1.set_title("Sagittale")
    ax2.imshow(numpy_image[:, numpy_image.shape[1] // 2, :])
    ax2.set_title("Coronale")
    ax3.imshow(numpy_image[:, :, numpy_image.shape[2] // 2])
    ax3.set_title("Assiale")
    plt.show()


def load_and_assign_tensor_labels(graph_ids, tensor_labels): 
    labeled_graphs = [] 

    for graph_id, tensor_label in zip(graph_ids, tensor_labels): 
        # Load the graph associated with the graph_id 
        file_name = f"/content/drive/MyDrive/Tesi Progetto/graphs/brain_graph_{graph_id}.graphml" 
        graph = nx.read_graphml(file_name) 

        tensor_label = tensor_label.cpu().numpy()
        i = 0
        
        for node in graph.nodes():
            if node == '0':
                graph.nodes[node]['label'] = 0
            else:
                graph.nodes[node]['label'] = tensor_label[i]
            i += 1

        labeled_graphs.append(graph)
    
    j = 0
    predicted_tumor_images = []
    for labeled_graph in labeled_graphs:
        img = nib.load(f"/content/drive/MyDrive/Tesi Progetto/dataset/BraTS2021_{graph_ids[j]}/BraTS2021_{graph_ids[j]}_SLIC.nii.gz")
        slic = img.get_fdata()
        im = generate_tumor_segmentation_from_graph(slic, labeled_graph)
        predicted_tumor_images.append(im)
        j += 1

    return predicted_tumor_images


def load_dgl_graphs_from_bin(file_path, ids_path):
    dgl_graph_list, _ = dgl.load_graphs(file_path)
    with open(f'{ids_path}', 'rb') as file:
        ids = pickle.load(file)
    return dgl_graph_list, ids


def prune_graphs(dgl_train_graphs):
    pruned_graphs = []

    for g in dgl_train_graphs:
        # Get the indices of nodes with label 3 and not 3
        labels = g.ndata["label"]
        nodes_label_3 = (labels == 3).nonzero(as_tuple=True)[0].tolist()
        nodes_not_3 = (labels != 3).nonzero(as_tuple=True)[0].tolist()

        # Calculate the number of nodes to remove
        total_nodes = len(labels)
        nodes_to_remove_count = total_nodes - 3 * len(nodes_not_3)

        # Remove 'nodes_to_remove_count' nodes with label 3
        nodes_to_remove = nodes_label_3[:nodes_to_remove_count]

        # Remove nodes from the graph
        pruned_g = dgl.remove_nodes(g, nodes_to_remove)

        # Add the pruned graph to the new list
        pruned_graphs.append(pruned_g)

    return pruned_graphs


def count_labels(graphs):
    label_counts = {}
    for graph in graphs:
        labels = graph.ndata['label'].tolist()
        for label in labels:
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts[label] = 1
    return label_counts


def class_weights_tensor(label_weights):
    num_classes = max(label_weights.keys())
    weight_tensor = torch.zeros(num_classes, dtype=torch.float32)

    # Sort the dictionary by keys (labels)
    sorted_label_weights = sorted(label_weights.items(), key=lambda x: x[0])

    for label, weight in sorted_label_weights:
        weight_tensor[label - 1] = weight  # Subtract 1 if your labels start from 1
    return weight_tensor


def compute_average_weights(graphs):
    label_counts = count_labels(graphs)
    total_count = sum(label_counts.values())
    class_weights = {label: total_count / count for label, count in label_counts.items()}
    weight_tensor = class_weights_tensor(class_weights)
    return weight_tensor

def compute_metrics(predicted_labels, true_labels):

    # Set values 1, 2, and 4 to 1, and all other values to 0 for both pred and labels
    pred_modified = (predicted_labels == 1) | (predicted_labels == 2) | (predicted_labels == 4)
    labels_modified = (true_labels == 1) | (true_labels == 2) | (true_labels == 4)

    # Convert the boolean tensors to integer tensors
    pred_modified = pred_modified.to(torch.int)
    labels_modified = labels_modified.to(torch.int)

    # Convert the tensors to NumPy arrays
    pred_modified_np = pred_modified.numpy()
    labels_modified_np = labels_modified.numpy()

    # Calculate F1-score, precision, and recall
    f1 = metrics.f1_score(labels_modified_np, pred_modified_np, zero_division=0)
    precision = metrics.precision_score(labels_modified_np, pred_modified_np, zero_division=0)
    recall = metrics.recall_score(labels_modified_np, pred_modified_np, zero_division=0)

    return recall, precision, f1


def minmax(features):
    # Min-Max scaling
    min_features = torch.min(features, dim=0)[0]
    max_features = torch.max(features, dim=0)[0]
    range_features = max_features - min_features
    normalized_features = (features - min_features) / range_features

    return normalized_features


def standardize_features(features):
    mean_features = torch.mean(features, dim=0)
    std_features = torch.std(features, dim=0)
    standardized_features = (features - mean_features) / (std_features + 1e-8)  # Adding a small value to avoid division by zero
    return standardized_features


import networkx as nx
import matplotlib.pyplot as plt

import networkx as nx
import matplotlib.pyplot as plt

def DGLGraph_plotter(nx_graph):
    # Color Map
    color_map = {3: (0.5, 0.5, 0.5, 0.2), 1: 'blue', 2: 'yellow', 4: 'red'}

    # Assign color to nodes taking their label
    node_colors = [color_map[nx_graph.nodes[node]['label']] for node in nx_graph.nodes]

    # Prepare the list of colors based on the edge labels
    edge_colors = []
    for u, v in nx_graph.edges():
        if nx_graph.nodes[u]['label'] in {1, 2, 4} and nx_graph.nodes[v]['label'] in {1, 2, 4}:
            edge_colors.append('red')
        else:
            edge_colors.append((0, 0, 0, 0.2))  # Black color with 0.2 transparency

    plt.figure(figsize=(15, 11))
    # Draw the graph
    pos = nx.spring_layout(nx_graph)
    nx.draw(nx_graph, pos, node_color=node_colors, edge_color=edge_colors)

    # Create a dictionary with node keys for nodes with labels 1, 2, and 4
    label_dict = {node: node for node in nx_graph.nodes if nx_graph.nodes[node]['label'] in {1, 2, 4}}

    # Draw node labels
    nx.draw_networkx_labels(nx_graph, pos, labels=label_dict, font_size = 8)

    # Create the legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Enhanced', markerfacecolor='red', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Edema', markerfacecolor='yellow', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Necrosis', markerfacecolor='blue', markersize=8),
        plt.Line2D([0], [0], marker='o', color='w', label='Other', markerfacecolor=color_map[3], markersize=8)
    ]

    # Display the legend
    plt.legend(handles=legend_elements, loc='best')
    plt.show()





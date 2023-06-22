from dgl.nn import GNNExplainer
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import networkx as nx

def explain_multiple_graphs(model, dataset, num_graphs=10, nodes_per_class=2, random_seed=42):
    np.random.seed(random_seed)
    random_indices = np.random.choice(len(dataset), num_graphs, replace=False)

    class_node_masks_all_graphs = []

    for idx in random_indices:
        graph, features, labels, _id = dataset[idx]

        # Create directory if it doesn't exist
        dir_path = f"{_id}_explanations"
        try:
            os.makedirs(dir_path, exist_ok=True)
        except Exception as e:
            print(f"Failed to create directory: {dir_path}")
            print(f"Reason: {e}")

        class_node_masks, class_edge_masks = explain_nodes_by_class(model, graph, features, labels, _id, nodes_per_class)
        class_node_masks_all_graphs.append(class_node_masks)

        # Save dictionaries to pickle
        with open(os.path.join(dir_path, f'{_id}_feat_mask.pickle'), 'wb') as handle:
            pickle.dump(class_node_masks, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(os.path.join(dir_path, f'{_id}_edge_mask.pickle'), 'wb') as handle:
            pickle.dump(class_edge_masks, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    return class_node_masks_all_graphs


def explain_nodes_by_class(model, graph, features, labels, _id, nodes_per_class=2):
    features = torch.from_numpy(features).float()
    explainer = GNNExplainer(model, num_hops=3)

    class_node_masks = {}
    class_edge_masks = {}  # new dictionary for edge masks
    classes = np.unique(labels)
    for c in classes:
        indices = (labels == c).nonzero()[0] 
        # Randomly permute the indices before selecting the subset
        indices = indices[torch.randperm(len(indices))][:nodes_per_class]
        node_masks = []
        edge_masks = {}  # Change edge_masks to a dictionary
        for node_id in indices:
            new_node_id, sg, feat_mask, edge_mask = explainer.explain_node(node_id.item(), graph, features)
            node_masks.append(feat_mask)
            edge_masks[node_id.item()] = edge_mask.detach().numpy().tolist()  # Save edge masks with node_id as key
        class_node_masks[c] = node_masks
        class_edge_masks[c] = edge_masks  # collect edge masks per class
    # plot_graph_with_edge_importance(graph, class_edge_masks, _id, labels)
    percentile_importance_score_dict = percentile_importance_score(class_node_masks)
    plot_percentile_importance_score(percentile_importance_score_dict, _id)
    plot_explanation(percentile_importance_score_dict, _id)
    return class_node_masks, class_edge_masks  # return edge masks as well


# Computes the mean percentile's importance score for each label
def percentile_importance_score(class_node_masks_dict):
    mean_dict = {}
    for key in class_node_masks_dict:
        mean_dict[key] = torch.mean(torch.stack(class_node_masks_dict[key]), dim=0)
    return mean_dict

def grouped_percentile_score(percentile_importance_score_mean):
    percentile_means = {}

    # For each key in the original dictionary
    for key in percentile_importance_score_mean:
        tensor = percentile_importance_score_mean[key]
        
        # Compute means for each percentile group
        percentile_10 = torch.mean(tensor[::5])
        percentile_25 = torch.mean(tensor[1::5])
        percentile_50 = torch.mean(tensor[2::5])
        percentile_75 = torch.mean(tensor[3::5])
        percentile_90 = torch.mean(tensor[4::5])

        # Store the means as a tensor in the new dictionary
        percentile_means[key] = torch.tensor([percentile_10, percentile_25, percentile_50, percentile_75, percentile_90])
    return percentile_means

# This function takes 5 elements for each modality FLAIR, T1, T1-Ce, T2 and extract the mean.
# The output will be [flair_importance_avg, t1_importance_avg, t1-ce_importance_avg, T2_importance_avg]
def modality_importance_score(class_node_masks_dict):
    mean_modality_dict = {}
    new_dict = {}
    for key in class_node_masks_dict:
        new_list = []
        for tensor in class_node_masks_dict[key]:
            a = torch.mean(tensor[0:5])
            b = torch.mean(tensor[5:10])
            c = torch.mean(tensor[10:15])
            d = torch.mean(tensor[15:20])
            new_list.append(torch.tensor([a, b, c, d]))
        new_dict[key] = new_list

    for key in class_node_masks_dict:
            mean_modality_dict[key] = torch.mean(torch.stack(new_dict[key]), dim=0)
    
    return mean_modality_dict


def plot_percentile_importance_score(percentile_importance_score_dict, _id):
    modalities = ["FLAIR", "T1", "T1-CE", "T2"]
    num_modalities = len(modalities)
    percentiles = ["10th", "25th", "50th", "75th", "90th"]
    num_percentiles = len(percentiles)

    # Create subplots for each diagram
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.subplots_adjust(hspace=0.4)

    # Iterate over each key in the percentile_importance_score_dict
    for i, (key, tensor_data) in enumerate(percentile_importance_score_dict.items()):
        # Create data for each percentile
        data = [tensor_data[j::num_percentiles].tolist() for j in range(num_percentiles)]

        # Plot the bar chart
        ax = axs[i // 2, i % 2]
        positions = range(1, num_modalities + 1)
        bar_width = 0.15
        for k in range(num_percentiles):
            percentile_positions = [p + k * bar_width for p in positions]
            ax.bar(percentile_positions, data[k], width=bar_width, label=percentiles[k])

        ax.set_xlabel("Modality")
        ax.set_ylabel("Importance")
        ax.set_title("Bar Chart for Label {}".format(key))
        ax.set_xticks([p + bar_width * (num_percentiles - 1) / 2 for p in positions])
        ax.set_xticklabels(modalities)
        ax.legend()

    plt.savefig(f"{_id}_explanations/{_id}_percentile_explanation.png")

    # Show the plot
    plt.tight_layout()
    plt.close(fig)



def plot_explanation(class_feat_masks, _id):
    modalities = ['flair', 't1', 't1ce', 't2']
    class_ids = [1, 2, 3, 4]

    avg_importance = {}

    for class_id in class_ids:
        try:
            avg_importance[class_id] = [class_feat_masks[class_id][i*5:(i+1)*5].mean().item() for i in range(4)]
        except KeyError:
            print(f"Key {class_id} not found in class_feat_masks.")
            avg_importance[class_id] = [0 for i in range(4)]  # o qualsiasi valore di default che desideri

    # Transform the data into a format suitable for matplotlib
    data = np.array(list(avg_importance.values()))

    # Plotting
    bar_width = 0.2
    r1 = np.arange(len(data[0]))  # the label locations
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.bar(r1, data[0], width=bar_width, label='Necrotic')
    ax.bar(r2, data[1], width=bar_width, label='Edema')
    ax.bar(r3, data[2], width=bar_width, label='Healthy')
    ax.bar(r4, data[3], width=bar_width, label='Enhancing Tumor')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Mean Importance')
    ax.set_title('Mean Importance by Modality and Class')
    ax.set_xticks([r + bar_width for r in range(len(modalities))])
    ax.set_xticklabels(modalities)
    ax.legend()

    fig.tight_layout()

    # Save the figure as a png file
    plt.savefig(f"{_id}_explanations/{_id}_feature_explanation.png")

    # Close the figure to free up memory
    plt.close(fig)

import networkx as nx
import matplotlib.pyplot as plt
import dgl
import numpy as np


def plot_graph_with_edge_importance(graph, class_edge_masks, _id, labels):
    # Convert DGL graph to NetworkX graph
    nx_graph = graph.to_networkx().to_undirected()

    # Remove node 0 from the graph
    nx_graph.remove_node(0)

    # Generate the layout of the graph
    pos = nx.spring_layout(nx_graph)

    # Assign colors based on class
    colors = {
        1: 'blue',
        2: 'yellow',
        3: 'green',
        4: 'red'
    }

    # Create a plot with a larger size
    fig, ax = plt.subplots(figsize=(10, 10))

    # Variables to store nodes for drawing
    visible_nodes = set()

    # Iterate over the class_edge_masks dictionary
    for class_name, node_edge_masks in class_edge_masks.items():
        # Get the nodes for the current class, and add their neighbors
        class_nodes = set(node_edge_masks.keys())
        for node in list(class_nodes):
            class_nodes.update(list(nx_graph.neighbors(node)))

        visible_nodes.update(class_nodes)

    # Get the edges for the visible nodes
    visible_edges = [(u, v) for u, v in nx_graph.edges() if u in visible_nodes or v in visible_nodes]

    # Iterate over the visible nodes to draw them
    for node_id in visible_nodes:
        # Get node color from the label
        node_color = colors[labels[node_id]]

        # Draw the node
        nx.draw_networkx_nodes(nx_graph, pos, nodelist=[node_id], node_color=node_color)

        # If node is in class_edge_masks, draw a violet border around it
        if node_id in class_edge_masks:
            nx.draw_networkx_nodes(nx_graph, pos, nodelist=[node_id], node_color=node_color, edgecolors='violet', linewidths=2.0)

        # Draw the edges with thickness based on importance
        for u, v in nx_graph.edges(node_id):
            if (u, v) in visible_edges:
                if u in node_edge_masks:
                    edge_importances = node_edge_masks[u]
                    edge_index = list(nx_graph.edges(u)).index((u, v))
                    edge_width = edge_importances[edge_index] * 5  # Scale edge width
                elif v in node_edge_masks:
                    edge_importances = node_edge_masks[v]
                    edge_index = list(nx_graph.edges(v)).index((v, u))
                    edge_width = edge_importances[edge_index] * 5  # Scale edge width
                else:
                    edge_width = 0.1  # Default edge width

                # Draw the edge
                nx.draw_networkx_edges(nx_graph, pos, edgelist=[(u, v)], width=edge_width)

    # Draw the labels (node IDs) with a smaller font size for visible nodes only
    node_labels = {n: str(n) for n in visible_nodes}
    nx.draw_networkx_labels(nx_graph, pos, labels=node_labels, font_size=8)

    # Display the plot
    plt.axis("off")
    plt.title(f'Graph {_id}')
    plt.show()

import networkx as nx
import pickle
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import ast

def plot_percentiles_mean(labeled_graph):
    percentile_indices = {
        10: [0, 5, 10, 15],
        25: [1, 6, 11, 16],
        50: [2, 7, 12, 17],
        75: [3, 8, 13, 18],
        90: [4, 9, 14, 19]
    }

    # Prepare data for each label
    for label in [1, 2, 3, 4]:
        # Filter nodes with the current label
        nodes = [node for node, data in labeled_graph.nodes(data=True) if data['label'] == label]
        
        # Calculate the mean of the desired percentiles for the filtered nodes
        means = []
        for percentile, indices in percentile_indices.items():
            percentile_values = [np.mean([ast.literal_eval(labeled_graph.nodes[node]['feature'])[i] for i in indices]) for node in nodes]
            mean_value = np.mean(percentile_values)
            means.append(mean_value)
        
        # Plot the results
        plt.plot([10, 25, 50, 75, 90], means, label=f'Label {label}')

    plt.xlabel('Percentile')
    plt.ylabel('Mean Value')
    plt.legend()
    plt.show()



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

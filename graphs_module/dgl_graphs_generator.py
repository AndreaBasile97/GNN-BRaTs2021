import torch
import dgl
import os
import networkx as nx
import nibabel as nib
import pickle
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DglGenerator:
    def __init__(self, dataset_path, graph_path, labels_path, dgl_path, tensor_path):
        self.dataset_path = dataset_path
        self.graph_path = graph_path
        self.labels_path = labels_path
        self.dgl_path = dgl_path
        self.tensor_path = tensor_path


    def dataset_splitting(self):
        graph_filenames = os.listdir(self.graph_path)
        graph_filenames.sort()

        graph_ids = []
        for graph in graph_filenames:
            id = graph.split("_")[2].split(".")[0]
            graph_ids.append(id)

        label_filenames = []
        for id in graph_ids:
            if os.path.isfile(f"{self.labels_path}/labels_{id}.pkl"):
                label_filenames.append(f"labels_{id}.pkl")

        graph_label_pairs = list(zip(graph_filenames, label_filenames))

        random.shuffle(graph_label_pairs)

        train_val_pairs, test_pairs = train_test_split(graph_label_pairs, test_size=0.1, random_state=42)
        train_pairs, val_pairs = train_test_split(train_val_pairs, test_size=0.2, random_state=42)

        train_graphs, train_labels = zip(*train_pairs)
        val_graphs, val_labels = zip(*val_pairs)
        test_graphs, test_labels = zip(*test_pairs)

        train_graphs = [os.path.join(self.graph_path, filename) for filename in train_graphs]
        train_labels = [os.path.join(self.labels_path, filename) for filename in train_labels]
        val_graphs = [os.path.join(self.graph_path, filename) for filename in val_graphs]
        val_labels = [os.path.join(self.labels_path, filename) for filename in val_labels]
        test_graphs = [os.path.join(self.graph_path, filename) for filename in test_graphs]
        test_labels = [os.path.join(self.labels_path, filename) for filename in test_labels]

        print(f"Training set: {len(train_graphs)} samples with {len(train_labels)} labels")
        print(f"Validation set: {len(val_graphs)} samples with {len(val_labels)} labels")
        print(f"Test set: {len(test_graphs)} samples with {len(test_labels)} labels")

        return train_graphs, train_labels, val_graphs, val_labels, test_graphs, test_labels


    def load_graphs(self, graph_files, set_type, save=False):
        graphs = []
        patient_ids = []

        for graph_file in graph_files:
            is_corrupted = False

            id_patient = graph_file.split("_")[2].split(".")[0]
            print(id_patient)

            segmented_image = nib.load(f'{self.dataset_path}/BraTS2021_{id_patient}/BraTS2021_{id_patient}_SLIC.nii.gz')

            with open(f'{self.labels_path}/labels_{id_patient}.pkl', 'rb') as file:
                labels_generated = pickle.load(file)

            graph = nx.read_graphml(graph_file)
            dgl_graph = dgl.from_networkx(graph)

            feature_name = 'feature'
            features = []
            for _, node_data in graph.nodes(data=True):
                feature = eval(node_data[feature_name])
                if not feature:
                    is_corrupted = True
                    break
                features.append(feature)
            if is_corrupted:
                continue
            features = torch.tensor(features)

            dgl_graph.ndata['feat'] = features

            tl = self.tensor_labels(segmented_image.get_fdata(), labels_generated, graph, id_patient, save=False)

            dgl_graph.ndata['label'] = tl

            graphs.append(dgl_graph)
            patient_ids.append(id_patient)

        if save:
            dgl.save_graphs(f'{self.dgl_path}/{set_type}_dgl_graphs.bin', graphs)
            with open(f'{self.dgl_path}/{set_type}_patient_ids.pkl', 'wb') as file:
                pickle.dump(patient_ids, file)

        return graphs, patient_ids


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
import os
import nibabel as nib
import pickle
import numpy as np
from matplotlib import pyplot as plt
from skimage import graph
import networkx as nx
from training.utilities import get_patient_ids, add_labels_to_single_graph


class GraphsModule:
    def __init__(self, dataset_path, graphs_path, features_path, labels_path):
        self.dataset_path = dataset_path
        self.graphs_path = graphs_path
        self.features_path = features_path
        self.labels_path = labels_path


    def extract_RAG(self, segments):
        rag = graph.RAG(segments, connectivity=2)  
        G = nx.Graph()
        G.add_nodes_from(rag.nodes)
        G.add_edges_from(rag.edges)
        return G
    

    def add_features_to_single_graph(self, features_file, graph_file):
        # Load the features dictionary
        print(features_file)
        with open(features_file, 'rb') as f:
            features_dict = pickle.load(f)

        # Load the graph
        G = nx.read_graphml(graph_file)

        # Create a copy of the graph
        new_graph = nx.Graph()
        new_graph.add_nodes_from(G.nodes())
        new_graph.add_edges_from(G.edges())

        # Assign the feature vector to the corresponding node in the copy of the graph
        for node, features in features_dict.items():
            if str(int(node)) in new_graph:
                new_graph.nodes[str(int(node))]['feature'] = str(features)
            else:
                raise Exception(f'{str(int(node))} not found in the graph {new_graph}')

        new_graph.nodes[str(0)]['feature'] = '[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]'

        return new_graph
    

    # Generation of empty graphs for the entire dataset
    def generate_empty_RAG(self):
        subdirectories = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        for subdir in subdirectories:
            subdir_path = os.path.join(self.dataset_path, subdir)
            images_to_combine_path = []
            id, flag = get_patient_ids([f"{self.dataset_path}/{subdir}"])
            for filename in os.listdir(subdir_path):
                try:
                    scan_modality = (filename.split("_")[2]).split(".")[0]
                    if scan_modality in ['SLIC']:
                        scan = nib.load(f"{self.dataset_path}/{subdir}/{filename}").get_fdata()
                        rag = self.extract_RAG(scan)
                        nx.write_graphml(rag, f"{self.graphs_path}/brain_graph_{id[0]}.graphml")
                        print(f"Graph successfully created for {id[0]}")
                except Exception as e:
                    print(f"Error: {e}")


    # Assigning features to all graphs
    def assign_features_to_all_graphs(self):
        # take all the graphs in graphs_path
        graphs = [g for g in os.listdir(self.graphs_path)]
        for graph in graphs:
            try:
                print(f'Processing {graph}')
                id = graph.split("_")[2].split(".")[0]
                features_file_path = f"{self.features_path}/features_{id}.pkl"
                graphs_file_path = f"{self.graphs_path}/{graph}"
                new_graph = self.add_features_to_single_graph(features_file_path, graphs_file_path)
                new_graph.remove_node('0')
                nx.write_graphml(new_graph, f"{self.graphs_path}/brain_graph_{id}.graphml")
            except:
                print(f'Features associated with {graph} not found. Are you sure that you have {self.features_path}/features_{id}.pkl in {self.features_path}')
                pass


    # Assigning labels to all graphs
    def assign_labels_to_all_graphs(self):
        subdirectories = [d for d in os.listdir(self.dataset_path) if os.path.isdir(os.path.join(self.dataset_path, d))]
        for subdir in subdirectories:
            subdir_path = os.path.join(self.dataset_path, subdir)
            images_to_combine_path = []
            id, flag = get_patient_ids([f"{self.dataset_path}/{subdir}"])
            print(f'Processing {id[0]}')
            for filename in os.listdir(subdir_path):
                try:
                    scan_modality = (filename.split("_")[2]).split(".")[0]
                    if scan_modality in ['SLIC']:
                        slic = nib.load(f"{self.dataset_path}/{subdir}/{filename}").get_fdata()
                except Exception as e:
                    print(f'The slic file for patient {id[0]} was not found: {e}')

            with open(f'{self.labels_path}/labels_{id[0]}.pkl', 'rb') as f:
                label = pickle.load(f)

            graph = nx.read_graphml(f'{self.graphs_path}/brain_graph_{id[0]}.graphml')

            try:
                new_graph = add_labels_to_single_graph(label, slic, graph)
                nx.write_graphml(new_graph, f"{self.graphs_path}/brain_graph_{id[0]}.graphml")
                print(f'Labels successfully associated with graph {id[0]}')
            except Exception as e:
                print(f'Error, labels have not been added to the graph of patient {id[0]}: {e}')

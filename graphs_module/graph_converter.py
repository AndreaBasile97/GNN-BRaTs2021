import networkx as nx
import torch
import nibabel as nib
import pickle
import dgl
import numpy as np

def load_dgl_graphs_from_bin(file_path, ids_path):
    dgl_graph_list, _ = dgl.load_graphs(file_path)
    with open(f'{ids_path}', 'rb') as file:
        ids = pickle.load(file)
    return dgl_graph_list, ids


def dgl_to_simple_graph(dgl_graph):
    nx_multigraph = dgl_graph.to_networkx().to_undirected()
    nx_graph = nx.Graph()
    
    # Copy node attributes
    for node, attrs in dgl_graph.ndata.items():
        for u, attr_values in enumerate(attrs):
            if node == 'label':
                attr_values = int(attr_values)
            elif node == 'feat':
                attr_values = attr_values.tolist()
            nx_graph.add_node(u, **{node: attr_values})
    
    # Copy edges (while removing multiple edges)
    for u, v, data in nx_multigraph.edges(data=True):
        if nx_graph.has_edge(u, v):
            continue
        nx_graph.add_edge(u, v, **data)
    
    return nx_graph


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


def tensor_labels(segmented_image, labels_generated, empty_RAG, id_patient, save=False):
    R = assign_labels_to_graph(labels_generated, segmented_image, empty_RAG)
    tl = torch.tensor(list(nx.get_node_attributes(R, 'label').values()))
    if save==True:
       torch.save(tl, f'..labels_module/tensor_labels/tensor_label_{id_patient}')
    return tl


# create a dataset of DGLGraphs using the networkx graphs
def load_graphs(graph_files, set_type:str, save=False):
    
    graphs = []
    patient_ids = []  # Add a list to store patient IDs

    for graph_file in graph_files:

        is_corrupted = False

        id_patient = graph_file.split("_")[2].split(".")[0]
        print(id_patient)

        try:
            # Load segmented brain by SLIC of the current user
            segmented_image = nib.load(f'../datasets/old_dataset/BraTS2021_{id_patient}/BraTS2021_{id_patient}_SLIC.nii.gz')
            
            # Load labels of the current user
            with open(f'../labels_module/labels/labels_{id_patient}.pkl', 'rb') as file:
                labels_generated = pickle.load(file)

            # Load graph of the current user
            graph = nx.read_graphml(f'graphs/{graph_file}')

            graph.remove_node('0')
        
            # Convert to DGL format
            dgl_graph = dgl.from_networkx(graph)


            with open(f'../features_module/new_features/features_{id_patient}.pkl', 'rb') as f:
                feature_pkl = pickle.load(f)
            output_list = [value for value in feature_pkl.values()]
            is_corrupted = any(len(lst) == 0 for lst in output_list) # check if there are empty list
            tensor_pkl = torch.tensor(output_list) # tensor features
            dgl_graph.ndata['feat'] = tensor_pkl

            # Convert labels into torch tensors
            tl = tensor_labels(segmented_image.get_fdata(), labels_generated, graph, id_patient, save=False)

            # Assign to dgl_graph.ndata['label']
            dgl_graph.ndata['label'] = tl

            if not is_corrupted:
                graphs.append(dgl_graph)
                patient_ids.append(id_patient)  # Add the patient ID to the list
        except:
            pass

    if save:
        dgl.save_graphs(f'DGL_graphs/{set_type}_dgl_graphs_fix.bin', graphs)
        # Save patient_ids to a file
        with open(f'DGL_graphs/{set_type}_patient_ids_fix.pkl', 'wb') as file:
            pickle.dump(patient_ids, file)
            
    return graphs, patient_ids

import os 

def generate_DGL_graphs():
    graphs_list = [g for g in os.listdir('./new_graphs')]
    load_graphs(graphs_list, set_type='new_train', save=True)


def generate_tumor_segmentation_from_graph(segmented_image, labeled_graph):
    # Create a dictionary to map node IDs to their labels
    node_label_map = {int(n): labeled_graph.nodes[n]['label'] for n in labeled_graph.nodes()}
    
    # Create a mask for labels to keep (1, 2, and 4)
    labels_to_keep = np.array([1, 2, 4])
    
    # Replace node IDs in segmented_image with their corresponding labels
    label_image = np.vectorize(node_label_map.get)(segmented_image)
    
    # Use np.isin to create a boolean mask for the labels we want to keep
    mask = np.isin(label_image, labels_to_keep)
    
    # Use np.where to create the segmented tumor image
    segmented_tumor = np.where(mask, label_image, 0)
    
    return segmented_tumor


generate_DGL_graphs()
# dgl_train_graphs, t_ids = load_dgl_graphs_from_bin('../graphs_module/DGL_graphs/train_dgl_graphs_fix.bin', '../graphs_module/DGL_graphs/train_patient_ids_fix.pkl')

# print(t_ids[0])
# print(dgl_train_graphs[0].ndata['feat'])

# graph = nx.read_graphml(f'graphs/brain_graph_{t_ids[0]}.graphml')
# with open(f'../features_module/new_features/features_{t_ids[0]}.pkl', 'rb') as f:
#     feature_pkl = pickle.load(f)
# output_list = [value for value in feature_pkl.values()]
# tensor_pkl = torch.tensor(output_list)
# print(tensor_pkl)
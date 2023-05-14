import networkx as nx
import torch
import nibabel as nib
import pickle
import dgl
import numpy as np
import os 


# Load DGL graphs from bin
def load_dgl_graphs_from_bin(file_path, ids_path):
    dgl_graph_list, _ = dgl.load_graphs(file_path)
    with open(f'{ids_path}', 'rb') as file:
        ids = pickle.load(file)
    return dgl_graph_list, ids




# Convert DGL to Networkx
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




# Utilities for assign labels to a NX Graph
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




# Transform labels to a tensor for a given patient
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
            graph = nx.read_graphml(f'new_graphs/{graph_file}')
        
            # Convert to DGL format
            dgl_graph = dgl.from_networkx(graph)


            with open(f'../features_module/new_features/features_{id_patient}.pkl', 'rb') as f:
                feature_pkl = pickle.load(f)
                del feature_pkl[0]

            output_list = [value for value in feature_pkl.values()]
            is_corrupted = any(len(lst) == 0 for lst in output_list) # check if there are empty list
            tensor_pkl = torch.tensor(output_list, dtype=torch.float32) # tensor features

            dgl_graph.ndata['feat'] = tensor_pkl

            # Convert labels into torch tensors
            tl = tensor_labels(segmented_image.get_fdata(), labels_generated, graph, id_patient, save=False)

            # Assign to dgl_graph.ndata['label']
            dgl_graph.ndata['label'] = tl

            if not is_corrupted:
                graphs.append(dgl_graph)
                patient_ids.append(id_patient)  # Add the patient ID to the list
        except Exception as e:
            print(e)

    if save:
        dgl.save_graphs(f'DGL_graphs/{set_type}_dgl_graphs_fix.bin', graphs)
        # Save patient_ids to a file
        with open(f'DGL_graphs/{set_type}_patient_ids_fix.pkl', 'wb') as file:
            pickle.dump(patient_ids, file)
            
    return graphs, patient_ids



# Transform a feature matrix of 20-dimensional vector into a 5-dimensional vector through the mean

# def transform_feature_vector(feature_vector):
#     # Define your percentile indices
#     percentile_indices = {
#         10: [0, 5, 10, 15],
#         25: [1, 6, 11, 16],
#         50: [2, 7, 12, 17],
#         75: [3, 8, 13, 18],
#         90: [4, 9, 14, 19]
#     }

#     # Create an empty array of 5 elements
#     new_feature_vector = np.empty(5)

#     # Compute the mean for each percentile and store it in the new feature vector
#     for i, indices in enumerate(percentile_indices.values()):
#         new_feature_vector[i] = np.mean([feature_vector[idx] for idx in indices])

#     return new_feature_vector

# def transform_feature_matrix(feature_matrix):
#     # Convert PyTorch tensor to numpy array
#     feature_matrix_np = feature_matrix.numpy()

#     # Create an empty matrix with n rows and 5 columns
#     new_feature_matrix = np.empty((feature_matrix_np.shape[0], 5))

#     # Transform each feature vector
#     for i in range(feature_matrix_np.shape[0]):
#         new_feature_matrix[i, :] = transform_feature_vector(feature_matrix_np[i, :])

#     # Convert the resulting numpy array back to PyTorch tensor
#     new_feature_matrix_torch = torch.from_numpy(new_feature_matrix)
    
#     return new_feature_matrix_torch



# Generation of DGL bin using 'load_graphs'
def generate_DGL_graphs():
    graphs_list = [g for g in os.listdir('./new_graphs')]
    load_graphs(graphs_list, set_type='FIXED_TRAIN', save=True)



generate_DGL_graphs()

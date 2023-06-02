import os
import pickle
from tqdm import tqdm
import torch
import numpy as np
import json
import networkx as nx
import dgl


def get_patient_ids(paths):
    ids = []
    for path in paths:
        splitted_path = path.split("/")
        ids.append(splitted_path[-1].split("_")[1])
    if all(elem == ids[0] for elem in ids):
        return ids, True
    else:
        return ids, False
    

def load_networkx_graph(fp): 
    with open(fp,'r') as f: 
        json_graph = json.loads(f.read()) 
        return nx.readwrite.json_graph.node_link_graph(json_graph) 
    
    
def get_graph(graph_path, id): 
    nx_graph = load_networkx_graph(graph_path) 
    features = np.array([nx_graph.nodes[n]['features'] for n in nx_graph.nodes]) 
    labels = np.array([nx_graph.nodes[n]['label'] for n in nx_graph.nodes]) 
    
    # Mappatura delle etichette 
    label_mapping = {0: 3, 1: 2, 2: 1, 3: 4} 
    labels = np.vectorize(label_mapping.get)(labels) 
    
    G = dgl.from_networkx(nx_graph) 
    n_edges = G.number_of_edges() 
    # normalization 
    degs = G.in_degrees().float() 
    norm = torch.pow(degs, -0.5) 
    norm[torch.isinf(norm)] = 0 
    G.ndata['norm'] = norm.unsqueeze(1) 
    #G.ndata['feat'] = features 
    return (G, features, labels, id) 


def generate_dgl_dataset(dataset_path):
    print(f'generating the dataset form {dataset_path}')
    subdirectories = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    new_graphs = [] 
    for subdir in tqdm(subdirectories):
        subdir_path = os.path.join(dataset_path, subdir)
        try:
            id, flag = get_patient_ids([f"{dataset_path}/{subdir}"])
            for filename in os.listdir(subdir_path):
                file_type = (filename.split("_")[2]).split(".")[0]
                if file_type in ['nxgraph']:
                    quadruple = get_graph(f'{dataset_path}/{subdir}/BraTS2021_{id[0]}_nxgraph.json', id[0]) 
                    new_graphs.append(quadruple)

        except Exception as e:
            print(f'Error: {e}')
    print('Success! The dataset has been generated.')
    with open('full_dataset_with_id.pickle', 'wb') as f:
        pickle.dump(new_graphs, f)
    return new_graphs


generate_dgl_dataset('/ext/train_006')
# generate_dgl_dataset('/Users/daniela/Desktop/Nuovo Progetto/preprocessed')
import os
import nibabel as nib
import pickle
import numpy as np
from matplotlib import pyplot as plt
from skimage import graph
import networkx as nx


def get_patient_ids(paths):
    ids = []
    for path in paths:
        splitted_path = path.split("/")
        ids.append(splitted_path[-1].split("_")[1])
    if all(elem == ids[0] for elem in ids):
        return ids, True
    else:
        return ids, False


def extract_RAG(segments):
    rag = graph.RAG(segments, connectivity=2)  
    G = nx.Graph()
    G.add_nodes_from(rag.nodes)
    G.add_edges_from(rag.edges)
    return G


#########################
dataset_path = '/ext/PREPROCESSED_DATASET_NEW/'
root = '/ext/tesi_BraTS2021/graphs/'


def generate_empty_RAG(brain_images_path):
    subdirectories = [d for d in os.listdir(brain_images_path) if os.path.isdir(os.path.join(brain_images_path, d))]
    for subdir in subdirectories:
        subdir_path = os.path.join(brain_images_path, subdir)
        images_to_combine_path = []
        id, flag = get_patient_ids([f"{dataset_path}{subdir}"])
        for filename in os.listdir(subdir_path):
            try:
                scan_modality = (filename.split("_")[2]).split(".")[0]
                if scan_modality in ['SLIC']:
                    scan = nib.load(f"{brain_images_path}/{subdir}/{filename}").get_fdata()
                    rag = extract_RAG(scan)
                    nx.write_graphml(rag, f"{root}brain_graph_{id[0]}.graphml")
                    print(f"Graph sucessfully created for {id[0]}")
            except Exception as e:
                print(f"Error: {e}")

generate_empty_RAG(dataset_path)
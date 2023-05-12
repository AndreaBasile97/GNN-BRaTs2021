import pickle
from utilities import load_dgl_graphs_from_bin
import torch

dgl_train_graphs, t_ids = load_dgl_graphs_from_bin('dgl/train_dgl_graphs.bin', 'dgl/train_patient_ids.pkl')
dgl_validation_graphs, v_ids = load_dgl_graphs_from_bin('dgl/val_dgl_graphs.bin', 'dgl/val_patient_ids.pkl')
dgl_test_graphs, test_ids = load_dgl_graphs_from_bin('dgl/test_dgl_graphs.bin', 'dgl/test_patient_ids.pkl')


def check_features(dgl_graph, id):
    
    with open(f'../features/features_{id}.pkl', 'rb') as f:
        feature_pkl = pickle.load(f)

    output_list = [value for value in feature_pkl.values()]
    tensor_pkl = torch.tensor(output_list)

    print(tensor_pkl.shape)
    print(dgl_graph.ndata['feat'].shape)

check_features(dgl_train_graphs[0], t_ids[0])




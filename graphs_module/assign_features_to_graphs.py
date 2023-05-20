import pickle
import networkx as nx
import os

# Graphs are filled with 20-dimensional feature vector for each node.
def add_features_to_graph(features_file, graph_file):
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

#############################
graphs_path = '/ext/tesi_BraTS2021/graphs'
features_path = '/ext/tesi_BraTS2021/features'
save_path = '/ext/tesi_BraTS2021/graphs'


def assing_features_to_graphs(graphs_path, features_path, save_path):
    # take all the graphs in graphs_path
    graphs = [g for g in os.listdir(graphs_path)]
    for graph in graphs:
        try:
            print(f'processing {graph}')
            id = graph.split("_")[2].split(".")[0]
            features_file_path = f"{features_path}/features_{id}.pkl"  # Use a new variable here
            graphs_file_path = f"{graphs_path}/{graph}"
            new_graph = add_features_to_graph(features_file_path, graphs_file_path)  # Use the new variable here
            new_graph.remove_node('0')
            nx.write_graphml(new_graph, f"{save_path}/brain_graph_{id}.graphml")
        except:
            print(f'features associated with {graph} not found. Are you sure that you have {features_path}/features_{id}.pkl in {features_path}')
            pass


assing_features_to_graphs(graphs_path, features_path, save_path)
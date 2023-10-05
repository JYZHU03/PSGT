import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

import networkx as nx

def edge_index_to_adj_matrix(data):
    edge_index = data.edge_index
    batch = data.batch
    batch_size = batch[-1].item() + 1

    node_counts = torch.bincount(batch)

    max_node_num = node_counts.max().item()
    adj_matrices = torch.zeros((batch_size, max_node_num, max_node_num), dtype=torch.float)
    for b in range(batch_size):
        mask = (batch[edge_index[0]] == b)
        edges_of_graph = edge_index[:, mask]
        start_node = (batch == b).nonzero(as_tuple=True)[0][0].item()
        adjusted_edges = edges_of_graph - start_node
        adj_matrices[b, adjusted_edges[0], adjusted_edges[1]] = 1

    adj_matrices = 1 - adj_matrices

    no_restrict_mask = torch.zeros_like(adj_matrices)

    return adj_matrices, no_restrict_mask



def construct_single_graph_relation_matrix(adj):
    G = nx.from_numpy_array(adj.cpu().numpy())

    length = dict(nx.all_pairs_shortest_path_length(G))
    max_nodes = adj.shape[0]

    dist_matrix = torch.full((max_nodes, max_nodes), float('inf'))
    for i in range(max_nodes):
        for j, l in length[i].items():
            dist_matrix[i, j] = l

    dist_matrix[dist_matrix == float('inf')] = 0

    depth_vector = dist_matrix[0]

    beta = 0.5
    decay = lambda d: torch.exp(-beta * d)

    S = 1 / (dist_matrix + 1) * decay(depth_vector).unsqueeze(0) * decay(depth_vector).unsqueeze(1)

    return S


def construct_relation_matrix(data):
    edge_index = data.edge_index
    batch = data.batch
    batch_size = batch[-1].item() + 1

    node_counts = torch.bincount(batch)

    max_node_num = node_counts.max().item()

    S_matrices = torch.zeros((batch_size, max_node_num, max_node_num), dtype=torch.float, device=edge_index.device)

    for b in range(batch_size):
        mask = (batch[edge_index[0]] == b)
        edges_of_graph = edge_index[:, mask]
        start_node = (batch == b).nonzero(as_tuple=True)[0][0].item()
        adjusted_edges = edges_of_graph - start_node

        adj_matrix = torch.zeros((max_node_num, max_node_num), device=edge_index.device)
        adj_matrix[adjusted_edges[0], adjusted_edges[1]] = 1

        S_subgraph = construct_single_graph_relation_matrix(adj_matrix)

        S_matrices[b] = S_subgraph
    return S_matrices

def add_second_order_edges(data):
    edge_index = data['edge_index']
    batch = data['batch']
    batch_size = batch[-1] + 1

    graph_ptrs = batch.bincount().cumsum(dim=0)
    graph_ptrs = torch.cat([torch.tensor([0]), graph_ptrs.cpu()])

    max_nodes = max(graph_ptrs[1:] - graph_ptrs[:-1])

    adjacency_matrices = []

    for idx in range(batch_size):
        start_ptr, end_ptr = graph_ptrs[idx], graph_ptrs[idx + 1]
        graph_edges = edge_index[:, (batch[edge_index[0]] == idx) & (batch[edge_index[1]] == idx)]

        adj_matrix = torch.zeros((max_nodes, max_nodes))

        adj_matrix[graph_edges[0] - start_ptr, graph_edges[1] - start_ptr] = 1
        adj_matrix[graph_edges[1] - start_ptr, graph_edges[0] - start_ptr] = 1

        adj_square = torch.mm(adj_matrix, adj_matrix)
        adj_square[adj_square > 0] = 1

        adj_square[torch.arange(max_nodes), torch.arange(max_nodes)] = 0

        adj_matrix = adj_matrix + adj_square
        adj_matrix[adj_matrix > 0] = 1



        adjacency_matrices.append(adj_matrix)

    adjacency_tensor = torch.stack(adjacency_matrices)
    adjacency_tensor = 1 - adjacency_tensor
    return adjacency_tensor



def compute_decay_distance(A, beta=0.1):
    A_np = A.numpy()

    A_csr = csr_matrix(A_np)

    distances_from_source, _ = dijkstra(A_csr, return_predecessors=True, indices=0)

    decay_np = np.exp(-beta * distances_from_source)

    decay_torch = torch.from_numpy(decay_np).float()

    return decay_torch

def add_relation_edges(data):
    edge_index = data['edge_index']
    batch = data['batch']
    batch_size = batch[-1] + 1

    graph_ptrs = batch.bincount().cumsum(dim=0)
    graph_ptrs = torch.cat([torch.tensor([0]), graph_ptrs.cpu()])

    max_nodes = max(graph_ptrs[1:] - graph_ptrs[:-1])

    adjacency_matrices = []

    for idx in range(batch_size):
        start_ptr, end_ptr = graph_ptrs[idx], graph_ptrs[idx + 1]
        graph_edges = edge_index[:, (batch[edge_index[0]] == idx) & (batch[edge_index[1]] == idx)]

        adj_matrix = torch.zeros((max_nodes, max_nodes))

        adj_matrix[graph_edges[0] - start_ptr, graph_edges[1] - start_ptr] = 1
        adj_matrix[graph_edges[1] - start_ptr, graph_edges[0] - start_ptr] = 1

        adj_square = torch.mm(adj_matrix, adj_matrix)
        S_2 = (adj_square > 0) & (adj_matrix == 0)
        S_2 = S_2.float() * 2.0

        decay = compute_decay_distance(adj_matrix, beta=0.5)
        adj_matrix = adj_matrix + S_2

        a = 1
        P = torch.zeros_like(adj_matrix)
        non_zero_indices = adj_matrix != 0

        P[non_zero_indices] = torch.exp(a / adj_matrix[non_zero_indices])

        adj_matrix = P * decay.view(-1, 1) * decay.view(1, -1)


        adjacency_matrices.append(adj_matrix)

    adjacency_tensor = torch.stack(adjacency_matrices)
    return adjacency_tensor


def get_attention_mask(data, edge_att):

    head_list = []
    relation_graph = add_relation_edges(data) * 1000.1
    ori_adj, no_restrict_mask = edge_index_to_adj_matrix(data)
    no_restrict_mask = no_restrict_mask * 1000.1
    edge_att = edge_att * 1000.1


    head_list.append(edge_att)
    head_list.append(relation_graph)
    head_list.append(no_restrict_mask)
    head_list.append(no_restrict_mask)

    output_tensor = torch.cat(head_list, dim=0)

    return output_tensor






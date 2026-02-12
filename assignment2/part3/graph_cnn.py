import torch.nn as nn
import torch

import torch.nn.functional as F
from torch_geometric.utils import add_self_loops


class MatrixGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(MatrixGraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.B)

    def make_adjacency_matrix(self, edge_index, num_nodes):
        """
        Creates adjacency matrix from edge index.

        :param edge_index: [source, destination] pairs defining directed edges nodes. dims: [2, num_edges]
        :param num_nodes: number of nodes in the graph.
        :return: adjacency matrix with shape [num_nodes, num_nodes]

        Hint: A[i,j] -> there is an edge from node j to node i
        """
        adjacency_matrix = torch.zeros((num_nodes, num_nodes), dtype=torch.float32, device=edge_index.device)
        sources, destinations = edge_index
        adjacency_matrix[destinations, sources] = 1.0
        return adjacency_matrix

    def make_inverted_degree_matrix(self, edge_index, num_nodes):
        """
        Creates inverted degree matrix from edge index.

        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :param num_nodes: number of nodes in the graph.
        :return: inverted degree matrix with shape [num_nodes, num_nodes]. Set degree of nodes without an edge to 1.
        """
        sources, destinations = edge_index
        degree_vector = torch.zeros(num_nodes, dtype=torch.float32, device=edge_index.device)
        ones = torch.ones(destinations.size(0), dtype=torch.float32, device=edge_index.device)
        degree_vector = degree_vector.index_add(0, destinations, ones)

        degree_vector = torch.where(degree_vector == 0, torch.ones_like(degree_vector), degree_vector)
        inverted_degree_vector = 1.0 / degree_vector
        inverted_degree_matrix = torch.diag(inverted_degree_vector)
        return inverted_degree_matrix

    def forward(self, x, edge_index):
        """
        Forward propagation for GCNs using efficient matrix multiplication.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :return: activations for the GCN
        """
        A = self.make_adjacency_matrix(edge_index, x.size(0))
        D_inv = self.make_inverted_degree_matrix(edge_index, x.size(0))

        neighbor_term = D_inv @ A @ x @ self.W.T
        self_term = x @ self.B.T
        out = neighbor_term + self_term
        return out

class MessageGraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(MessageGraphConvolution, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.B = nn.Parameter(torch.Tensor(out_features, in_features))

        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.B)

    @staticmethod
    def message(x, edge_index):
        """
        message step of the message passing algorithm for GCNs.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :return: message vector with shape [num_nodes, num_in_features]. Messages correspond to the old node values.

        Hint: check out torch.Tensor.index_add function
        """
        num_nodes = x.size(0)
        sources, destinations = edge_index

        messages = x[sources]
        aggregated_messages = torch.zeros((num_nodes, x.size(1)), dtype=x.dtype, device=x.device)
        aggregated_messages = aggregated_messages.index_add(0, destinations, messages)
        
        sum_weight = torch.zeros(num_nodes, 1, dtype=x.dtype, device=x.device)
        ones = torch.ones(messages.size(0), 1, dtype=x.dtype, device=x.device)
        
        sum_weight = sum_weight.index_add(0, destinations, ones)
        sum_weight = torch.where(sum_weight == 0, torch.ones_like(sum_weight), sum_weight)

        aggregated_messages = aggregated_messages / sum_weight
        return aggregated_messages

    def update(self, x, messages):
        """
        update step of the message passing algorithm for GCNs.

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param messages: messages vector with shape [num_nodes, num_in_features]
        :return: updated values of nodes. shape: [num_nodes, num_out_features]
        """
        neighbor_term = messages @ self.W.T
        self_term = x @ self.B.T
        x = neighbor_term + self_term
        return x

    def forward(self, x, edge_index):
        message = self.message(x, edge_index)
        x = self.update(x, message)
        return x


class GraphAttention(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphAttention, self).__init__()
        self.W = nn.Parameter(torch.Tensor(out_features, in_features))
        self.a = nn.Parameter(torch.Tensor(out_features * 2))

        nn.init.xavier_uniform_(self.W)
        nn.init.uniform_(self.a, 0, 1)

    def forward(self, x, edge_index, debug=False):
        """
        Forward propagation for GATs.
        Follow the implementation of Graph attention networks (Veličković et al. 2018).

        :param x: values of nodes. shape: [num_nodes, num_features]
        :param edge_index: [source, destination] pairs defining directed edges nodes. shape: [2, num_edges]
        :param debug: used for tests
        :return: updated values of nodes. shape: [num_nodes, num_out_features]
        :return: debug data for tests:
                 messages -> messages vector with shape [num_nodes, num_out_features], i.e. Wh from Veličković et al.
                 edge_weights_numerator -> unnormalized edge weightsm i.e. exp(e_ij) from Veličković et al.
                 softmax_denominator -> per destination softmax normalizer

        Hint: the GAT implementation uses only 1 parameter vector and edge index with self loops
        Hint: It is easier to use/calculate only the numerator of the softmax
              and weight with the denominator at the end.

        Hint: check out torch.Tensor.index_add function
        """
        edge_index, _ = add_self_loops(edge_index)

        sources, destinations = edge_index
        messages = x @ self.W.T

        h_src = messages[sources]
        h_dst = messages[destinations]
        attention_inputs = torch.cat([h_dst, h_src], dim=1)

        e = (attention_inputs * self.a.unsqueeze(0)).sum(dim=1)
        e = F.leaky_relu(e)
        edge_weights_numerator = torch.exp(e)

        weighted_messages = h_src * edge_weights_numerator.unsqueeze(1)

        softmax_denominator = torch.zeros(x.size(0), dtype=torch.float32, device=edge_index.device)
        softmax_denominator = softmax_denominator.index_add(0, destinations, edge_weights_numerator)

        aggregated_messages = torch.zeros_like(messages)
        aggregated_messages = aggregated_messages.index_add(0, destinations, weighted_messages)

        denom = softmax_denominator.clamp(min=1e-6).unsqueeze(1)
        aggregated_messages = aggregated_messages / denom
        
        if debug:
            return aggregated_messages, {'edge_weights': edge_weights_numerator, 'softmax_weights': softmax_denominator,
                                         'messages': h_src}
        return aggregated_messages

import torch, math, numpy as np, scipy.sparse as sp
import torch.nn as nn, torch.nn.functional as F, torch.nn.init as init

from torch.autograd import Variable
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import timeit
import itertools
from copy import deepcopy 

import pickle
from collections import defaultdict


class HyperMSGConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, a, b, cuda=True):
        super(HyperMSGConvolution, self).__init__()
        self.a, self.b = a, b
        self.cuda = cuda
        self.W = Parameter(torch.FloatTensor(a, b))
        self.W2 = Weight_net()
        self.bias = Parameter(torch.FloatTensor(b))
        #self.edge_count = edge_count
        self.reset_parameters()

        
    def reset_parameters(self):
        std = 1. / math.sqrt(self.W.size(1))
        self.W.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, structure, H, input_weight):
        #self.edge_count = self.edge_count.cuda()
        W, b = self.W, self.bias
        W2 = self.W2(input_weight)
        AH = signal_shift_hypergraph_(structure,H, W2) 
        AHW = torch.mm(AH, W) 
        output = AHW + b
        #print(W2)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.a) + ' -> ' \
               + str(self.b) + ')'

class Weight_net(Module):
    def __init__(self, cuda=True):
        super(Weight_net, self).__init__()
        self.cuda = cuda
        self.fc1 = nn.Linear(2, 10)
        self.fc2 = nn.Linear(10, 5)
        self.fc3 = nn.Linear(5, 1)

    def forward(self, input_weight):
        out = F.relu(self.fc1(input_weight))
        out = F.dropout(out,0.25, training=self.training)
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return F.sigmoid(out)



def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = torch.tensor(mx.sum(1))
    r_inv = torch.pow(rowsum, -1).flatten()
    r_inv[torch.isinf(r_inv)] = 0.
    mx = mx * r_inv[:,None]
    #r_mat_inv = torch.diag(r_inv)
    #mx = r_mat_inv.dot(mx)
    return mx


def signal_shift_hypergraph_(hypergraph, H, W2):
    min_value, max_value = 1e-7, 1e1
    #torch.clamp_(H, min_value, max_value)
    new_signal = H.clone()
    for edge,nodes in hypergraph.items():
        nodes = nodes.to(dtype=torch.long) 
        connectivity_neighbor_nodes = H[nodes] * W2[nodes].reshape(-1,1)

        tmp_sum = torch.sum(connectivity_neighbor_nodes, dim=0)
        tmp_mat = tmp_sum.repeat(len(nodes)).reshape(len(nodes),H.shape[1])

        neighbor_signal = ((tmp_mat-H[nodes])/((len(nodes)-1)))
        new_signal[nodes] = new_signal[nodes] + neighbor_signal   
    return normalize(new_signal)

def signal_shift_hypergraph_inductive_p_alpha(hypergraph, H, W2, edge_count):
    min_value, max_value = 1e-7, 1e1
    torch.clamp_(H, min = min_value)
    new_signal = H.clone()
    p_value = 2
    alpha = 2
    for edge,nodes in hypergraph.items():
        nodes = nodes.to(dtype=torch.long)       
        connectivity_neighbor_nodes = torch.pow(H[nodes],p_value)
        connectivity_neighbor_nodes = connectivity_neighbor_nodes * W2[nodes].reshape(-1,1)

        for node in nodes:
            neighbors = [x.item() for x in nodes if x not in node]
            prob = W2[neighbors]
            prob = prob.detach().cpu().numpy()
            prob /= prob.sum() 
            if(alpha<len(neighbors)):
                prob_sample = np.random.choice(len(neighbors), alpha, replace=False, p=prob.T[0])
            else:
                prob_sample = np.arange(len(neighbors))
            tmp_sum = torch.sum(connectivity_neighbor_nodes[prob_sample], dim=0)
            new_signal[node] = new_signal[node] + tmp_sum/len(prob_sample)        
    edge_count = edge_count +1
    edge_count = edge_count.resize(len(edge_count),1)
    new_signal = torch.pow(new_signal/edge_count,(1.0/p_value))
    return normalize(new_signal)


import torch, numpy as np
import torch.nn as nn, torch.nn.functional as F

from torch.autograd import Variable
from model import utils 
import ipdb


class HyperMSG(nn.Module):
    def __init__(self, E, X, args):
        """
        d: initial node-feature dimension
        h: number of hidden units
        c: number of classes
        """
        super(HyperMSG, self).__init__()
        d, l, c = args.d, args.depth, args.c
        cuda = args.cuda and torch.cuda.is_available()
        
        h = [d]
        for i in range(l-1):
            power = l - i + 2
            h.append(2**power)
        h.append(c) 
        self.hmsg1 = utils.HyperMSGConvolution(d,16)
        self.hmsg2 = utils.HyperMSGConvolution(16,c)
        self.do, self.l = args.dropout, args.depth

    def forward(self, structure, H, input_weight):
        """
        an l-layer GCN
        """
        do, l= self.do, self.l
        H = F.relu(self.hmsg1(structure, H, input_weight))
        H = F.dropout(H, do, training=self.training)
        H = self.hmsg2(structure, H, input_weight)      
        return F.log_softmax(H, dim=1)


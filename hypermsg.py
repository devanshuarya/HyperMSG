from config import config
args = config.parse()
import random
from data_hyper import data
from collections import defaultdict
import copy
import os, torch, numpy as np
import time
from model import networks
import torch, os, numpy as np, scipy.sparse as sp
import torch.optim as optim, torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm
from model import utils
import itertools
import torch.multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
torch.manual_seed(args.seed)
np.random.seed(args.seed)
import logging
logging.propagate = False 
logging.getLogger().setLevel(logging.ERROR)


# gpu, seed
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"        
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ['PYTHONHASHSEED'] = str(args.seed)


# load data
test_result_each_split = []

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = np.diag(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


for num in range(10):
    res_split = []
    args.split = num+1
    print("SPLIT:   ", args.split)
    dataset, train, test = data.load(args)
    #print("length of train is", len(train))

    #wandb.init(config=args)
    HyperMSG = {}
    E =  dataset['hypergraph']
    X, Y = dataset['features'], dataset['labels']
   
    for k, v in E.items():
       E[k] = list(v)
       if k not in E[k]:
          E[k].append(k)
    
    #def get_data_insights(hypergraph,X):
    global_neighborhood = defaultdict(list)
    edge_count = np.zeros(X.shape[0])
    global_neighborhood_count = np.zeros(X.shape[0])
    unique_nodes = []
    for edge, nodes in E.items():
        for node in nodes:
            unique_nodes.append(node)
            neighbor_nodes = ([i for i in nodes if i != node])
            edge_count[node] = edge_count[node] + 1
            global_neighborhood[node].extend(neighbor_nodes)

    for k,v in global_neighborhood.items():
        global_neighborhood_count[k] = len(set(v))
    unique_nodes = list(set(unique_nodes))
    input_weight = np.concatenate((np.expand_dims(global_neighborhood_count, axis = 1), np.expand_dims(edge_count, axis = 1)), axis = 1)
    
    X = normalize(X)
    args.d, args.c = X.shape[1], Y.shape[1]
    hypermsg = networks.HyperMSG(E, X, args)
    optimiser = optim.Adam(list(hypermsg.parameters()), lr=args.rate, weight_decay=args.decay)
    #scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.95)

    scheduler = optim.lr_scheduler.StepLR(optimiser, 200, gamma=0.2, last_epoch=-1)

    def normalize(mx):
        """Row-normalize sparse matrix"""
        rowsum = np.array(mx.sum(1))
        r_inv = np.power(rowsum, -1).flatten()
        r_inv[np.isinf(r_inv)] = 0.
        r_mat_inv = np.diag(r_inv)
        mx = r_mat_inv.dot(mx)
        return mx


    X = torch.FloatTensor(np.array(X))

    Y = np.array(Y)
    Y = torch.LongTensor(np.where(Y)[1])
    idx_train = torch.LongTensor(train)
    idx_test = torch.LongTensor(test)
    input_weight = torch.FloatTensor(input_weight)
    
    # cuda
    args.Cuda = True and torch.cuda.is_available()
    if args.Cuda:
        hypermsg.cuda()
        X, Y = X.cuda(), Y.cuda()
        idx_test = idx_test.cuda()
        input_weight = input_weight.cuda()
    #    idx_val = idx_val.cuda()
        idx_train = idx_train.cuda()
        for key, value in E.items():
            E[key] = torch.Tensor(list(E[key])).cuda()


    def train(epoch):
        t = time.time()
        hypermsg.train()
        optimiser.zero_grad()
        output = hypermsg(E,X,input_weight)
        loss_train = F.nll_loss(output[idx_train], Y[idx_train])
        acc_train = accuracy(output[idx_train], Y[idx_train])
        loss_train.backward()
        optimiser.step()
        if(epoch%10==0):
            print('Epoch: {:04d}'.format(epoch+1),'loss_train: {:.4f}'.format(loss_train.item()),'acc_train: {:.4f}'.format(acc_train.item()),
           'time: {:.4f}s'.format(time.time() - t))

    def test():
        hypermsg.eval()
        output = hypermsg(E,X,input_weight)
        loss_test = F.nll_loss(output[idx_test], Y[idx_test])
        acc_test = accuracy(output[idx_test], Y[idx_test])
        print(acc_test.item(), end="\t", flush=True)

        return loss_test.item(),acc_test.item()

    def accuracy(Z, Y):
        """
        arguments:
        Z: predictions
        Y: ground truth labels

        returns: 
        accuracy
        """
        
        predictions = Z.max(1)[1].type_as(Y)
        correct = predictions.eq(Y).double()
        correct = correct.sum()

        accuracy = correct / len(Y)
        return accuracy


    # Train model
    t_total = time.time()
    for epoch in range(args.epochs):
        train(epoch)
        if(epoch%50==0):
            print(epoch)
            loss_test,acc_test = test()
            res_split.append(acc_test)
            
    fin_loss,fin_acc = test()
    print(res_split)
    test_result_each_split.append(fin_acc)
    print("\n")


print(sum(test_result_each_split)/10)







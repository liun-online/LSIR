import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dropout_feat):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_dim, out_dim)
        self.linear2 = nn.Linear(out_dim, out_dim)
        self.act = nn.PReLU()
        self.drop = dropout_feat
    
    def forward(self, x):
        x = self.act(self.linear1(x))
        x = F.dropout(x, self.drop, training=self.training)
        x = self.linear2(x)
        return x


class Propagation(nn.Module):
    def __init__(self, n_layers, dropout_adj):
        super(Propagation, self).__init__()
        self.n_layers = n_layers
        self.drop = dropout_adj
    
    def __dropout(self, graph, drop):
        size = graph.size()
        index = graph._indices().t()
        values = graph._values()
        random_index = torch.rand(len(values)) + drop
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / drop
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def forward(self, graph, emb, train=True):
        embs = [emb]
        if train:
            g_droped = self.__dropout(graph, self.drop)
        else:
            g_droped = graph
        for layer in range(self.n_layers):
            emb = torch.sparse.mm(g_droped, emb)
            embs.append(emb)
        embs = torch.stack(embs, dim=1)
        embs = torch.mean(embs, dim=1)
        return embs

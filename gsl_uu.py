import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import defaultdict
from utils import cal_cosine, cal_cosine_self, _convert_sp_mat_to_sp_tensor
import scipy.sparse as sp


class GSL4uu(nn.Module):
    def __init__(self, in_dim, m_head, dropout, min_keep, max_add, pseudo_num, pseudo_lam, 
    tau, edge_emb_flag=False, add_ori=False):
        super(GSL4uu, self).__init__()
        self.linear_node = nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in range(m_head)])
        self.edge_fusion = nn.Linear(2*in_dim, in_dim)
        self.linear_cluster = nn.ModuleList([nn.Linear(in_dim, in_dim) for _ in range(m_head)])
        self.act = nn.ReLU()
        self.m_head = m_head
        self.min_keep = min_keep
        self.max_add = max_add
        self.edge_emb_flag = edge_emb_flag
        self.dropout = nn.Dropout(dropout)

        self.att = nn.Parameter(torch.empty(size=(1, in_dim)), requires_grad=True)
        self.fc = nn.Linear(in_dim, in_dim, bias=True)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax()

        self.pseudo_num = pseudo_num
        self.pseudo_lam = pseudo_lam
        self.tau = tau

        self.add_ori = add_ori
    
    def new_stru(self, user_emb, prob_dele_edge, prob_add_edge, all_nodes, cluster_nodes, batch_subgraph):
        cal_edge_emb = user_emb[all_nodes]
        cluster_cal_edge_emb = user_emb[cluster_nodes]
        dele_prob = prob_dele_edge
        add_prob = prob_add_edge

        sim_dele = 0
        sim_add = 0
        for j in range(self.m_head):
            z = self.act(self.linear_node[j](cal_edge_emb))
            y = self.act(self.linear_cluster[j](cluster_cal_edge_emb))
            sim_dele += cal_cosine_self(z, batch_subgraph)
            sim_add += cal_cosine(z, y)
        sim_dele /= self.m_head
        num_node = batch_subgraph.shape[0]
        # sim_dele = torch.sparse_coo_tensor(batch_subgraph._indices(), sim_dele, (num_node, num_node))
        sim_add /= self.m_head

        ## dele ##
        temp = 0 
        num_nei = torch.sparse.sum(batch_subgraph, -1).values().cpu().data.numpy().astype(np.int32)
        dele_indices = batch_subgraph._indices()
        dele_final_sim = []
        dele_final_indices = []
        for one_num_nei in num_nei:
            one_dele_sim = sim_dele[temp : temp + one_num_nei]
            one_dele_indices  = dele_indices[:, temp : temp + one_num_nei]
            center = torch.unique(one_dele_indices[0]).cpu().data.numpy().astype(np.int32)
            temp += one_num_nei

            prob_i = min(self.min_keep, prob_dele_edge[center])
            keep_edge_num = int(one_num_nei*(1-prob_i))
            _, indices = one_dele_sim.topk(int(keep_edge_num))
            dele_final_sim.append(one_dele_sim[indices].reshape(1, -1))
            dele_final_indices.append(one_dele_indices[:, indices])
        dele_final_sim = torch.cat(dele_final_sim, -1)
        dele_final_indices = torch.cat(dele_final_indices, -1)

        ## add ##
        cluster_num = len(cluster_nodes)
        add_final_sim = []
        add_final_indices = []
        for i in range(sim_add.shape[0]):
            prob_i = min(self.max_add, prob_add_edge[i])
            add_edge_num = int(cluster_num * prob_i)
            if add_edge_num != 0:
                one_node_sim = sim_add[i]
                _, indices = one_node_sim.topk(add_edge_num)

                add_final_sim.append(one_node_sim[indices].reshape(1, -1))
                indices_ = indices.unsqueeze(0)
                idx = torch.ones_like(indices_) * i
                add_final_indices.append(torch.cat([idx, indices_], 0))
        add_final_sim = torch.cat(add_final_sim, -1)
        add_final_indices = torch.cat(add_final_indices, -1)
        return dele_final_indices, dele_final_sim, add_final_indices, add_final_sim

    def mimic_learning(self, batch_act_user, batch_inact_user, user_emb_uu, cluster_map, cluster_index):
        cluster_dict = {}
        for idx in range(len(cluster_index)):
            cluster_dict[cluster_index[idx]] = idx
        user_emb_uu_ = user_emb_uu.detach()
        cluster_emb = user_emb_uu_[cluster_index]
        batch_inact_emb = user_emb_uu_[batch_inact_user]
        batch_act_emb = user_emb_uu_[batch_act_user]

        act_cluster = cluster_map[:, 1]
        act_cluster = torch.LongTensor([cluster_dict[i] for i in act_cluster]).reshape(-1, 1).repeat(1, self.pseudo_num).reshape(-1)
        pseudo_act_emb = batch_act_emb.unsqueeze(1).repeat(1, self.pseudo_num, 1)  # act_user_num*pseudo_num*emb_size
        
        sele_index = np.random.choice(range(batch_inact_emb.shape[0]), size=pseudo_act_emb.shape[0]*self.pseudo_num, replace=True)
        sele_inact_emb = batch_inact_emb[sele_index].reshape_as(pseudo_act_emb)
        pseudo_act_emb = (self.pseudo_lam*pseudo_act_emb + (1-self.pseudo_lam)*sele_inact_emb).reshape(-1, pseudo_act_emb.shape[-1])
        
        #target_cluster_sim = torch.exp(self.p2c_sim(pseudo_act_emb, cluster_emb) / self.tau)
        target_cluster_sim = torch.softmax(cal_cosine(pseudo_act_emb, cluster_emb) / self.tau, -1)
        # target_cluster_sim /= (target_cluster_sim.sum(-1, keepdim=True) + 1e-8)
        mimic_loss = -torch.log(target_cluster_sim[range(target_cluster_sim.shape[0]), act_cluster] + 1e-8).mean()
        return mimic_loss

import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from layer import MLP, Propagation
from utils import bpr_loss, reg_loss
from gsl_uu import GSL4uu
from utils import _convert_sp_mat_to_sp_tensor
import scipy.sparse as sp

from encoder import Encoder

class Inac_rec(nn.Module):
    def __init__(self, args, num_users, num_items, cluster_map):
        super(Inac_rec, self).__init__()
        self.cluster_map = cluster_map  # [act_id, cluster_index]
        self.cluster_index = np.sort(list(set(self.cluster_map[:, 1])))
        self.cluster_num = len(self.cluster_index)
        self.loss_lam = args.loss_lam
        self.args = args
        self.GSL4uu = GSL4uu(args.hidden_dim, args.m_head, args.dropout, args.min_keep, args.max_add, \
            args.pseudo_num, args.pseudo_lam, args.tau, args.edge_emb_flag)

        self.uu_graph = None
        self.encoder = Encoder(args, num_users, num_items)
        self.num_users = num_users
        self.num_items = num_items
        self.weight = args.add_weight

    def train_encoder(self, batch_user_pos_neg, ui_graph, user_feat, item_feat):
        user_final_emb, item_emb, user_emb_ego, item_emb_ego = self.encoder(ui_graph, self.uu_graph, user_feat, item_feat)
        batch_user, batch_pos, batch_neg = batch_user_pos_neg[:, 0], batch_user_pos_neg[:, 1], batch_user_pos_neg[:, 2]

        rec_loss = bpr_loss(user_final_emb[batch_user], item_emb[batch_pos], item_emb[batch_neg])
        reg = reg_loss(user_emb_ego[batch_user], item_emb_ego[batch_pos], item_emb_ego[batch_neg])
        return rec_loss + reg*self.args.weight_decay
    
    def get_whole_stru(self, unique_user, final_dele_indices, final_dele_sim, final_add_indices, final_add_sim, user_num):
        dele_graph = torch.sparse_coo_tensor(final_dele_indices.t(), final_dele_sim, (user_num, user_num)).cuda()
        dele_graph = torch.sparse.softmax(dele_graph, 1)
        add_graph = torch.sparse_coo_tensor(final_add_indices.t(), final_add_sim, (user_num, user_num)).cuda()
        add_graph = torch.sparse.softmax(add_graph, 1)
        self_loop = torch.sparse_coo_tensor(torch.cat([unique_user.unsqueeze(0), unique_user.unsqueeze(0)]), torch.ones_like(unique_user), (self.num_users, self.num_users)).cuda()
        batch_graph = 1/2 * self_loop + 1/2 * (self.weight*add_graph + (1-self.weight)*dele_graph)
        return batch_graph

    def train_graph_generator(self, user_emb_ego, batch_user_pos_neg, batch_act_user, batch_inact_user, uu_dict, user_emb, item_emb, add_prob, dele_prob):
        batch_user, batch_pos, batch_neg = batch_user_pos_neg[:, 0], batch_user_pos_neg[:, 1], batch_user_pos_neg[:, 2]
        final_dele_indices, final_dele_sim, final_add_indices, final_add_sim = self.get_stru(user_emb_ego, batch_user, user_emb, uu_dict, add_prob, dele_prob)
        
        unique_user = torch.LongTensor(np.sort(list(set(batch_user))))
        batch_graph = self.get_whole_stru(unique_user, final_dele_indices, final_dele_sim, final_add_indices, final_add_sim, self.num_users)
        uu_emb = torch.sparse.mm(batch_graph, user_emb)
        user_final_emb = self.encoder.user_emb_map(torch.cat([user_emb_ego[batch_user], user_emb[batch_user], uu_emb[batch_user]], -1))        
        rec_loss = bpr_loss(user_final_emb, item_emb[batch_pos], item_emb[batch_neg])

        mimic_loss = self.GSL4uu.mimic_learning(batch_act_user, batch_inact_user, user_emb, self.cluster_map, self.cluster_index)
        return rec_loss + self.args.loss_lam * mimic_loss
    
    def get_stru(self, user_emb_ego, batch_user, user_emb_ori, uu_dict, add_prob, dele_prob, edge_emb=None):
        target = np.sort(list(set(batch_user)))

        link_target_nei = []
        target_nei = []
        for u in target:
            u_nei = uu_dict[u]
            target_nei += u_nei
            t = [u]*len(u_nei)
            link_target_nei.append(np.array([t, u_nei]).T)
            link_target_nei.append(np.array([u_nei, t]).T)
        link_target_nei = np.vstack(link_target_nei)
        target_nei = np.sort(list(set(target_nei)))

        link_nei_nei = []
        target_nei_nei = []
        for u in target_nei:
            u_nei = uu_dict[u]
            target_nei_nei += u_nei
            t = [u]*len(u_nei)
            link_nei_nei.append(np.array([t, u_nei]).T)
        link_nei_nei = np.vstack(link_nei_nei)
        target_nei_nei = np.sort(list(set(target_nei_nei)))

        cluster = self.cluster_index
        link_cluster_nei = []
        cluster_nei = []
        cluster_final_map = {}
        for i in range(len(cluster)):
            u_nei = uu_dict[cluster[i]]
            cluster_nei += u_nei
            t = [cluster[i]]*len(u_nei)
            link_cluster_nei.append(np.array([t, u_nei]).T)
            cluster_final_map[i] = cluster[i]
        link_cluster_nei = np.vstack(link_cluster_nei)
        cluster_nei = np.sort(list(set(cluster_nei)))

        all_nodes = np.sort(list(set(list(target) + list(target_nei) + list(target_nei_nei) + list(cluster) + list(cluster_nei))))
        all_nodes_num = len(all_nodes)
        all_nodes_map = {}
        for n in range(len(all_nodes)):
            all_nodes_map[all_nodes[n]] = n
        all_links_ = np.vstack([link_target_nei, link_nei_nei, link_cluster_nei])
        all_links = []
        for one_link in all_links_:
            all_links.append((all_nodes_map[one_link[0]], all_nodes_map[one_link[1]]))
        all_links = np.array(list(set(all_links)))
        all_links = all_links[np.argsort(all_links[:, 0])].T
        batch_subgraph = torch.sparse_coo_tensor(all_links, [1]*all_links.shape[1], (all_nodes_num, all_nodes_num)).cuda()

        tn_nodes = np.sort(list(set(list(target) + list(target_nei))))
        tn_nodes_ = np.array([all_nodes_map[n] for n in tn_nodes])
        tn_nodes_num = len(tn_nodes)
        tn_nodes_map = {}
        tn_final_map = {}
        for n in range(len(tn_nodes_)):
            tn_nodes_map[tn_nodes_[n]] = n
            tn_final_map[n] = tn_nodes[n]
        tn_links = []
        for one_link in link_target_nei:
            tn_links.append((tn_nodes_map[all_nodes_map[one_link[0]]], tn_nodes_map[all_nodes_map[one_link[1]]]))
        tn_links = np.array(list(set(tn_links)))
        tn_links = tn_links[np.argsort(tn_links[:, 0])].T
        tn_subgraph = torch.sparse_coo_tensor(tn_links, [1]*tn_links.shape[1], (tn_nodes_num, tn_nodes_num)).cuda()
        
        ## T=0 ##
        prob_dele_edge = dele_prob[all_nodes]
        prob_add_edge = add_prob[all_nodes]
        dele_final_indices, dele_final_sim, add_final_indices, add_final_sim = self.GSL4uu.new_stru(user_emb_ori, prob_dele_edge, prob_add_edge, all_nodes, cluster, batch_subgraph)        
        dele_sim_t0 = torch.sparse_coo_tensor(dele_final_indices, dele_final_sim[0], (all_nodes_num, all_nodes_num))
        dele_sim_t0 = torch.sparse.softmax(dele_sim_t0, 1)
        add_sim_t0 = torch.sparse_coo_tensor(add_final_indices, add_final_sim[0], (all_nodes_num, len(cluster)))
        add_sim_t0 = torch.sparse.softmax(add_sim_t0, 1)

        ori_feat = user_emb_ori[all_nodes]
        cluster_ori_feat = user_emb_ori[cluster]
        dele_emb = torch.sparse.mm(dele_sim_t0, ori_feat)
        add_emb = torch.sparse.mm(add_sim_t0, cluster_ori_feat)
        user_emb_t0 = 1/2 * ori_feat +  1/2 * (self.weight * add_emb + (1-self.weight) * dele_emb)
        user_emb_t0 = self.encoder.user_emb_map(torch.cat([user_emb_ego[all_nodes], ori_feat, user_emb_t0], -1)) 

        ## T=1 ##
        prob_dele_edge = dele_prob[tn_nodes]
        prob_add_edge = add_prob[tn_nodes]
        # user_emb_ori_ = user_emb_ori[all_nodes]
        cluster_ = np.array([all_nodes_map[c] for c in cluster])
        dele_final_indices, dele_final_sim, add_final_indices, add_final_sim = self.GSL4uu.new_stru(user_emb_t0, prob_dele_edge, prob_add_edge, tn_nodes_, cluster_, tn_subgraph)
        dele_final_indices = dele_final_indices.data.cpu().numpy()
        add_final_indices = add_final_indices.data.cpu().numpy()

        dele_sele = []
        final_dele_indices = []
        for i in range(dele_final_indices.shape[1]):
            if tn_final_map[dele_final_indices[0, i]] in target:
                dele_sele.append(i)
                final_dele_indices.append([tn_final_map[dele_final_indices[0, i]], tn_final_map[dele_final_indices[1, i]]])
        final_dele_indices = torch.LongTensor(final_dele_indices).cuda()
        final_dele_sim = dele_final_sim[0, dele_sele]
        
        add_sele = []
        final_add_indices = []
        for i in range(add_final_indices.shape[1]):
            if tn_final_map[add_final_indices[0, i]] in target:
                add_sele.append(i)
                final_add_indices.append([tn_final_map[add_final_indices[0, i]], cluster_final_map[add_final_indices[1, i]]])
        final_add_indices = torch.LongTensor(final_add_indices).cuda()
        final_add_sim = add_final_sim[0, add_sele]
        return final_dele_indices, final_dele_sim, final_add_indices, final_add_sim

    def get_emb(self, ui_graph, user_feat, item_feat, users, temp_flag=False):
        if temp_flag:
            user_final_emb, item_emb, user_emb_ego = self.encoder(ui_graph, self.uu_graph, user_feat, item_feat, train=False, temp_flag=temp_flag)
            return user_final_emb.detach()[users], item_emb.detach(), user_emb_ego.detach()
        else:
            user_final_emb, item_emb = self.encoder(ui_graph, self.uu_graph, user_feat, item_feat, train=False, temp_flag=temp_flag)
            return user_final_emb.detach()[users], item_emb.detach()
import numpy as np
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from layer import MLP, Propagation
from gsl_uu import GSL4uu


class Encoder(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(Encoder, self).__init__()
        if args.dataset in ['lastm']:
            self.user_emb_ego = torch.nn.Embedding(
                num_embeddings=num_users, embedding_dim=args.hidden_dim)
            self.item_emb_ego = torch.nn.Embedding(
                num_embeddings=num_items, embedding_dim=args.hidden_dim)
            nn.init.normal_(self.user_emb_ego.weight, std=0.1)
            nn.init.normal_(self.item_emb_ego.weight, std=0.1)
        else:
            self.user_map = MLP(args.user_feat_dim, args.hidden_dim, args.dropout)
            self.item_map = MLP(args.item_feat_dim, args.hidden_dim, args.dropout)
        self.prop_ui = Propagation(args.n_layers, args.dropout_adj)
        self.prop_uu = Propagation(1, args.dropout_adj)
        self.user_emb_map = nn.Linear(3*args.hidden_dim, args.hidden_dim)
        self.num_users = num_users
        self.num_items = num_items
    
    def forward(self, ui_graph, uu_graph, user_feat, item_feat, train=True, temp_flag=False):
        if user_feat is None:
            user_emb_ego = self.user_emb_ego.weight
            item_emb_ego = self.item_emb_ego.weight
        else:
            user_emb_ego = self.user_map(user_feat)
            item_emb_ego = self.item_map(item_feat)
        all_emb = torch.cat([user_emb_ego, item_emb_ego])
        all_emb = self.prop_ui(ui_graph, all_emb, train)
        user_emb, item_emb = torch.split(all_emb, [self.num_users, self.num_items])
        user_uu_emb = self.prop_uu(uu_graph, user_emb, train)
        user_final_emb = self.user_emb_map(torch.cat([user_emb_ego, user_emb, user_uu_emb], -1))
        if train:
            return user_final_emb, item_emb, user_emb_ego, item_emb_ego
        else:
            if temp_flag:
                return user_final_emb, item_emb, user_emb_ego
            else:
                return user_final_emb, item_emb
import numpy as np
import torch
import torch.nn.functional as F


def bpr_loss(batch_user_emb, batch_pos_emb, batch_neg_emb):
    ## lightgcn
    pos_scores = torch.mul(batch_user_emb, batch_pos_emb)
    pos_scores = torch.sum(pos_scores, dim=1)
    neg_scores = torch.mul(batch_user_emb, batch_neg_emb)
    neg_scores = torch.sum(neg_scores, dim=1)
    
    loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
    return loss

def reg_loss(batch_user_emb_ego, batch_pos_emb_ego, batch_neg_emb_ego):
    reg_loss = (1/2)*(batch_user_emb_ego.norm(2).pow(2) + 
                    batch_pos_emb_ego.norm(2).pow(2)  +
                    batch_neg_emb_ego.norm(2).pow(2))/float(batch_user_emb_ego.shape[0])
    return reg_loss

def generate_train_data(dataset):
    allPos = dataset.allPos
    active_user = dataset.index_act
    np.random.shuffle(active_user)
    S_act = []
    for user in active_user:
        pos_list = allPos[user]
        pos_len = len(pos_list)
        if pos_len == 0:
            continue
        posindex = np.random.randint(0, pos_len)
        positem = pos_list[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_item)
            if negitem in pos_list:
                continue
            else:
                break
        one_user = np.array([user, positem, negitem])
        S_act.append(one_user)
    S_act = np.array(S_act)
        
    inactive_user = dataset.index_inact
    np.random.shuffle(inactive_user)
    S_inact = []
    for user in inactive_user:
        pos_list = allPos[user]
        pos_len = len(pos_list)
        if pos_len == 0:
            continue
        posindex = np.random.randint(0, pos_len)
        positem = pos_list[posindex]
        while True:
            negitem = np.random.randint(0, dataset.m_item)
            if negitem in pos_list:
                continue
            else:
                break
        one_user = np.array([user, positem, negitem])
        S_inact.append(one_user)
    S_inact = np.array(S_inact)
    return S_act, S_inact

def get_add_dele_prob(args, users_D):
    add_prob = 1 / (1 + np.exp(users_D / args.r_add))
    dele_prob = (np.exp(users_D / args.r_dele) - np.exp(-users_D / args.r_dele)) / (np.exp(users_D / args.r_dele) + np.exp(-users_D / args.r_dele))
    return add_prob, dele_prob

def _convert_sp_mat_to_sp_tensor(X):
    coo = X.tocoo().astype(np.float32)
    row = torch.Tensor(coo.row).long()
    col = torch.Tensor(coo.col).long()
    index = torch.stack([row, col])
    data = torch.FloatTensor(coo.data)
    return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

def cal_cosine_self(z, subgraph):
    indices = subgraph._indices()

    idx_0 = indices[0]
    z_0 = z[idx_0]

    idx_1 = indices[1]
    z_1 = z[idx_1]
    sim = torch.cosine_similarity(z_0, z_1, -1)
    return sim

def cal_cosine(x, y):
    x = torch.unsqueeze(x, 1)
    x = x.repeat(1, y.shape[0], 1)
    sim_matrix = torch.cosine_similarity(x, y, -1)
    return sim_matrix
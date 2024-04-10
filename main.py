import numpy as np
from parse import set_params
from dataloader import Loader
import warnings
import utils
from model import Inac_rec
import torch
import torch.nn as nn
from evaluation import Evaluation
from datetime import datetime 


warnings.filterwarnings('ignore')
args = set_params()
if torch.cuda.is_available():
    args.device = torch.device("cuda")
else:
    args.device = torch.device("cpu")

class TrainFlow:
    def __init__(self, args):
        self.args = args
        self.own_str = args.dataset
        print(self.own_str)
        self.dataset = Loader(args)
        self.val_data = self.dataset.valDict
        self.test_data = self.dataset.testDict
        self.add_prob, self.dele_prob = utils.get_add_dele_prob(args, self.dataset.users_D)
        if args.edge_emb_flag:
            self.emb_initial_matrix = nn.Embedding(args.feat_dim, args.hidden_dim).cuda()
        self.model = Inac_rec(args, self.dataset.n_user, self.dataset.m_item, self.dataset.cluster_map).to(args.device)
        self.opt_encoder = torch.optim.Adam(self.model.encoder.parameters(), lr=args.lr_encoder)#, weight_decay=args.weight_decay)
        self.opt_gsl = torch.optim.Adam(self.model.GSL4uu.parameters(), lr=args.lr_gsl)#, weight_decay=args.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.999)

        self.eva_val = Evaluation(self.dataset.n_user, self.dataset.m_item, self.dataset.val_link, self.dataset.index_inact, self.args.eva_neg_num, \
            "./dataset/"+self.args.dataset+"/val_neg_links.npy")
        self.eva_test = Evaluation(self.dataset.n_user, self.dataset.m_item, self.dataset.test_link, self.dataset.index_inact, self.args.eva_neg_num, \
            "./dataset/"+self.args.dataset+"/test_neg_links.npy")

    def eva(self, data, eva, test_flag=False):
        self.model.eval()
        all_users = np.arange(self.dataset.n_user)
        all_user_emb, all_item_emb = self.model.get_emb(self.dataset.UI_Graph, self.dataset.user_feat, self.dataset.item_feat, all_users)
        all_user_emb = all_user_emb.data.cpu().numpy()
        all_item_emb = all_item_emb.data.cpu().numpy()
        results_10, results_inac_10 = eva.get_result(10, all_user_emb, all_item_emb)
        results_20, result_inacs_20 = eva.get_result(20, all_user_emb, all_item_emb)
        if test_flag:
            f = open(self.own_str+"_10.txt", "a")
            f.write(str(results_10[0])+"\t"+str(results_10[1])+"\t"+str(results_10[2])+"\t"+str(results_10[3])+"\t"+\
                str(results_inac_10[0])+"\t"+str(results_inac_10[1])+"\t"+str(results_inac_10[2])+"\t"+str(results_inac_10[3])+"\n")
            f.close()
            f = open(self.own_str+"_20.txt", "a")
            f.write(str(results_20[0])+"\t"+str(results_20[1])+"\t"+str(results_20[2])+"\t"+str(results_20[3])+"\t"+\
                str(result_inacs_20[0])+"\t"+str(result_inacs_20[1])+"\t"+str(result_inacs_20[2])+"\t"+str(result_inacs_20[3])+"\n")
            f.close()

    def train(self):
        self.model.uu_graph = self.dataset.UU_Graph
        for epoch in range(args.tot_epochs):
            ## Train encoder
            print("Train encoder")
            for inner_encoder in range(args.encoder_epochs):
                S_act, S_inact = utils.generate_train_data(self.dataset)
                act_batch_size = int(S_act.shape[0] / args.train_iters)
                inact_batch_size = int(S_inact.shape[0] / args.train_iters)
                act_temp = 0
                inact_temp = 0
                tot_loss = []
                print("New epoch!")
                terminal = True
                while terminal:
                    a=datetime.now()
                    self.model.train()
                    self.opt_encoder.zero_grad()
                    if args.train_iters == 1:
                        act_curr = S_act
                        inact_curr = S_inact
                        terminal = False
                    else:
                        if act_temp + act_batch_size < S_act.shape[0]:
                            act_curr = S_act[act_temp : act_temp + act_batch_size]
                            inact_curr = S_inact[inact_temp : inact_temp + inact_batch_size]
                            act_temp += act_batch_size
                            inact_temp += inact_batch_size
                        else:
                            act_curr = S_act[act_temp : ]
                            inact_curr = S_inact[inact_temp : ]
                            terminal = False
                    batch_user_pos_neg = np.vstack([act_curr, inact_curr])
                    batch_act_user = np.sort(list(set(act_curr[:, 0])))
                    batch_inact_user = np.sort(list(set(inact_curr[:, 0])))
                    batch_user = np.sort(list(set(batch_user_pos_neg[:, 0])))

                    loss = self.model.train_encoder(batch_user_pos_neg, self.dataset.UI_Graph, self.dataset.user_feat, self.dataset.item_feat)
                    loss.backward()
                    self.opt_encoder.step()
                    tot_loss.append(loss.cpu().data.numpy())
                b=datetime.now() 
                print("tot_epochs ", epoch, "\tencoder_epochs ", inner_encoder, "\tLoss ", np.array(tot_loss).mean(), "\tseconds ", (b-a).seconds)
            
            ## Get user item emb
            print("Get user item emb")
            all_users = np.arange(self.dataset.n_user)
            user_emb, item_emb, user_emb_ego = self.model.get_emb(self.dataset.UI_Graph, self.dataset.user_feat, self.dataset.item_feat, all_users, temp_flag=True)
            ## Train gsl
            print("Train gsl")
            for inner_gsl in range(args.gsl_epochs):
                S_act, S_inact = utils.generate_train_data(self.dataset)
                act_batch_size = int(S_act.shape[0] / args.train_iters)
                inact_batch_size = int(S_inact.shape[0] / args.train_iters)
                act_temp = 0
                inact_temp = 0
                tot_loss = []
                print("New epoch!")
                terminal = True
                while terminal:
                    a=datetime.now()
                    self.model.train()
                    self.opt_gsl.zero_grad()
                    if args.train_iters == 1:
                        act_curr = S_act
                        inact_curr = S_inact
                        terminal = False
                    else:
                        if act_temp + act_batch_size < S_act.shape[0]:
                            act_curr = S_act[act_temp : act_temp + act_batch_size]
                            inact_curr = S_inact[inact_temp : inact_temp + inact_batch_size]
                            act_temp += act_batch_size
                            inact_temp += inact_batch_size
                        else:
                            act_curr = S_act[act_temp : ]
                            inact_curr = S_inact[inact_temp : ]
                            terminal = False
                    batch_user_pos_neg = np.vstack([act_curr, inact_curr])
                    batch_act_user = np.sort(list(set(act_curr[:, 0])))
                    batch_inact_user = np.sort(list(set(inact_curr[:, 0])))
                    batch_user = np.sort(list(set(batch_user_pos_neg[:, 0])))
                    loss = self.model.train_graph_generator(user_emb_ego, batch_user_pos_neg, batch_act_user, batch_inact_user, self.dataset.uu_dict, user_emb, item_emb, add_prob=self.add_prob, dele_prob=self.dele_prob)
                    loss.backward()
                    self.opt_gsl.step()
                    tot_loss.append(loss.cpu().data.numpy())
                b=datetime.now() 
                print("tot_epochs ", epoch, "\tgsl_epochs ", inner_gsl, "\tLoss ", np.array(tot_loss).mean(), "\tseconds ", (b-a).seconds)
            ## Get whole structure
            with torch.no_grad():
                print("Get whole structure")
                all_users = torch.arange(self.dataset.n_user)
                final_dele_indices, final_dele_sim, final_add_indices, final_add_sim = self.model.get_stru(user_emb_ego, all_users, user_emb, self.dataset.uu_dict, self.add_prob, self.dele_prob)
                
                self.model.uu_graph =self.model.get_whole_stru(all_users, final_dele_indices, final_dele_sim, final_add_indices, final_add_sim, self.dataset.n_user)
                print("Get whole structure finish")

            
            print("Evaluation")
            self.eva(self.test_data, self.eva_test, test_flag=True)  
            # torch.save(self.model.state_dict(), self.own_str+'.pkl')
        
        ## test ##
        self.eva(self.test_data, self.eva_test, test_flag=True)            

if __name__ == '__main__':
    train = TrainFlow(args)
    train.train()

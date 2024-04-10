import argparse
import sys


argv = sys.argv
data = argv[1]

def yelp_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='yelp', help="")
    parser.add_argument('--edge_emb_flag', type=bool,default=False, help="")
    parser.add_argument('--hidden_dim', type=int,default=64, help="")
    parser.add_argument('--eva_neg_num', type=int,default=1000, help="")
    parser.add_argument('--n_layers', type=int,default=3)
    parser.add_argument('--train_iters', type=int,default=1)
    parser.add_argument('--weight_decay', type=float,default=1e-4, help="")
    parser.add_argument('--tot_epochs', type=int,default=100)
    parser.add_argument('--dropout_adj', type=float,default=0.4)
    parser.add_argument('--lr_encoder', type=float,default=0.005, help="")
    parser.add_argument('--lr_gsl', type=float,default=0.005, help="")
    parser.add_argument('--add_weight', type=float,default=0.5)
    parser.add_argument('--encoder_epochs', type=int,default=95)
    parser.add_argument('--gsl_epochs', type=int,default=5)
    parser.add_argument('--r_add', type=float,default=2)
    parser.add_argument('--r_dele', type=float,default=100.0)
    parser.add_argument('--min_keep', type=float, default=0.1)
    parser.add_argument('--max_add', type=float, default=0.1)
    parser.add_argument('--m_head', type=int,default=1)
    parser.add_argument('--dropout', type=float,default=0.)
    parser.add_argument('--loss_lam', type=float,default=0.1)
    parser.add_argument('--pseudo_num', type=int,default=10)
    parser.add_argument('--pseudo_lam', type=float,default=0.5)
    parser.add_argument('--tau', type=float,default=0.8)
    args, _ = parser.parse_known_args()
    return args

def flickr_params():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,default='flickr', help="")
    parser.add_argument('--edge_emb_flag', type=bool,default=False, help="")
    parser.add_argument('--hidden_dim', type=int,default=64, help="")
    parser.add_argument('--eva_neg_num', type=int,default=1000, help="")
    parser.add_argument('--n_layers', type=int,default=3)
    parser.add_argument('--train_iters', type=int,default=1)
    parser.add_argument('--weight_decay', type=float,default=1e-4, help="")
    parser.add_argument('--dropout_adj', type=float,default=0.5)
    parser.add_argument('--lr_encoder', type=float,default=0.001, help="") 
    parser.add_argument('--lr_gsl', type=float,default=0.001, help="") 
    parser.add_argument('--add_weight', type=float,default=0.5)
    parser.add_argument('--r_add', type=float,default=10)
    parser.add_argument('--r_dele', type=float,default=100)
    parser.add_argument('--encoder_epochs', type=int,default=90)
    parser.add_argument('--gsl_epochs', type=int,default=10)
    parser.add_argument('--min_keep', type=float, default=0.1)
    parser.add_argument('--max_add', type=float, default=0.1)
    parser.add_argument('--m_head', type=int,default=1)
    parser.add_argument('--dropout', type=float,default=0.) 
    parser.add_argument('--tot_epochs', type=int,default=100) 
    parser.add_argument('--loss_lam', type=float,default=0.01)
    parser.add_argument('--pseudo_num', type=int,default=5)
    parser.add_argument('--pseudo_lam', type=float,default=0.5)
    parser.add_argument('--tau', type=float,default=0.8)
    args, _ = parser.parse_known_args()
    return args

def set_params():
    if data == "yelp":
        args = yelp_params()
    elif data == "flickr":
        args = flickr_params()
    return args

import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import argparse
import config as cfg
import random
import numpy as np
import torch
from utils.general import read_pickle, write_pickle, str2bool
# from metaNA import MetaNA
from metaNA_adapt import MetaNA
from netEncode import NetEncode


def seed_torch(seed=2021):
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(data_dir, ratio=0.8):
    adj_s = read_pickle(data_dir + 'adj_s.pkl')
    adj_t = read_pickle(data_dir + 'adj_t.pkl')
    embeds = read_pickle(data_dir + 'emb_n2v1.pkl')
    # embeds = read_pickle(main_dir + 'embeds.pkl')
    links = read_pickle(data_dir + 'links_0.5.pkl')
    # links = read_pickle(main_dir + '/links_0.3.pkl')
    return (adj_s, adj_t, embeds, links)

def match(adj_s, adj_t,
          embeds, link_train_test,
          args):  # @emb: embeddings
    # @link_train_set: labeled linkage
    # 变成对称矩阵
    adj_s = ((adj_s + adj_s.T) > 0) * 1.
    adj_t = ((adj_t + adj_t.T) > 0) * 1.
    nets = [NetEncode(in_dim=cfg.dim_feature, out_dim=cfg.dim_feature),
            NetEncode(in_dim=cfg.dim_feature, out_dim=cfg.dim_feature)]
    # nets = [GNN(cfg.dim_feature, cfg.dim_out),
    #         GNN(cfg.dim_feature, cfg.dim_out)]
    uil = MetaNA(embeds, adj_s, adj_t, link_train_test, nets, args)
    # print(params_count(gnns[0]))
    uil.train(args.epochs)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='metaNA', type=str)
    parser.add_argument('--folder_dir', default='./dataset/', type=str)
    parser.add_argument('--dataset', default='FT', help='FT, DBLP, D_W_15K_V1', type=str)
    parser.add_argument('--device', default='cuda:1', type=str)
    parser.add_argument('--epochs', default=40, type=int, help='number of tasks')
    parser.add_argument('--ratio', default=cfg.ratio, type=float)
    parser.add_argument('--top_k', default=cfg.k, type=int)
    parser.add_argument('--adapt', default='True', type=str2bool)
    # parser.add_argument('--adapt', default=True, type=bool)
    parser.add_argument('--meta_lr', default=0.0005, type=float)   # FT 0.0005  D-W-15K 0.001
    parser.add_argument('--fast_lr', default=0.001, type=float)  # FT
    # parser.add_argument('--fast_lr', default=0.01, type=float)
    parser.add_argument('--n_way', default=5, type=int)
    parser.add_argument('--k_shot', default=5, type=int)  # FT 2  D-W-15K 5
    parser.add_argument('--meta_bsz', default=8, type=int)  # FT 8  D-W-15K 64
    # parser.add_argument('--options', default='structure', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    # os.chdir('../')
    seed_torch(2022)

    # 1. initial configuration
    args = get_args()
    print(args)
    cfg.init_args(args)

    # 2. load data
    data_dir = args.folder_dir + args.dataset + '/'
    adj_s, adj_t, embeds, link_train_test = load_data(data_dir, cfg.ratio)

    # 3. match
    match(adj_s, adj_t, embeds, link_train_test, args)

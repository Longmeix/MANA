import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import random
from deeplink import DeepLink_dualMap
import argparse
import torch
import networkx as nx
import config as cfg
from utils.general import read_pickle, write_pickle, str2bool
# from instructors.ULink import ULinkIns
import numpy as np
import time
from embedding_model import DeepWalk


def seed_torch(seed=2022):
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_data(data_dir, save_emb_path, ratio):
    adj_s = read_pickle(data_dir + 'adj_s.pkl').astype(np.float32)
    adj_t = read_pickle(data_dir + 'adj_t.pkl').astype(np.float32)
    embeds = read_pickle(save_emb_path)
    links = read_pickle(data_dir + f'links_{ratio}.pkl')

    return adj_s, adj_t, embeds, links


def train_embeddings(adj_path, embed_dim, save_emb_path):
    embeds = []

    # for evaluate time
    start = time.time()
    print("Start training embedding by deepwalk")
    for suffix in ('s', 't'):
        print(f'Creating graph {suffix}...')
        adj_mat = read_pickle(adj_path.format(suffix))
        adj_mat = ((adj_mat + adj_mat.T) > 0) * 1.
        G = nx.from_numpy_matrix(adj_mat, create_using=nx.DiGraph())

        deepwalk = DeepWalk(G, embed_dim=embed_dim, num_walks=1000, walk_len=5, window_size=5, \
                            num_cores=8, num_epochs=5)  # FT, DBLP
        # deepwalk = DeepWalk(G, embed_dim=embed_dim, num_walks=100, walk_len=5, window_size=5, \
        #                     num_cores=8, num_epochs=5)
        emb = deepwalk.get_embedding()
        embeds.append(emb)

    # embedding_epoch_time = time.time() - start

    print(f'Writing embeds to file {save_emb_path}')
    write_pickle(embeds, save_emb_path)
    return embeds


def train_align(adj_s, adj_t, emb, all_links, args):
    ins = DeepLink_dualMap(emb, adj_s, adj_t, all_links, args)
    ins.train_align()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='DeepLink', type=str)
    parser.add_argument('--folder_dir', default='../dataset/', type=str)
    parser.add_argument('--dataset', default='D_W_15K_V1', help='FT, DBLP, D_W_15K_V1', type=str)
    parser.add_argument('--device', default='cuda:3', type=str)
    parser.add_argument('--adapt', default='True', type=str2bool)
    parser.add_argument('--meta_test_cluster', default=1, type=int)
    parser.add_argument('--support', default='similarity', type=str, help='random/neighbor/similarity')
    parser.add_argument('--fast_lr', default=0.05, type=float)  # 0.05
    parser.add_argument('--n_way', default=5, type=int)
    parser.add_argument('--k_shot', default=5, type=int)
    parser.add_argument('--ratio', default=0.5, type=float)
    parser.add_argument('--top_k', default=10, type=int)
    # deeplink
    parser.add_argument('--unsupervised_lr', default=5e-4, type=float)
    parser.add_argument('--supervised_lr', default=5e-4, type=float)  # FT
    # parser.add_argument('--supervised_lr', default=0.005, type=float)
    parser.add_argument('--batch_size_mapping', default=32, type=int)  # FT 32  DW 5000
    parser.add_argument('--unsupervised_epochs', default=20, type=int)  # FT 20 DW 50
    parser.add_argument('--supervised_epochs', default=40, type=int)
    parser.add_argument('--embed_dim', default=cfg.dim_feature, type=int)
    parser.add_argument('--hidden_dim1', default=800, type=int)
    parser.add_argument('--hidden_dim2', default=1600, type=int)
    parser.add_argument('--alpha', default=0.8, type=float)

    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    # os.chdir('../')
    seed_torch(2023)

    args = get_args()
    cfg.init_args(args)

    # file path
    data_dir = args.folder_dir + args.dataset + '/'
    adj_path = data_dir + 'adj_{}.pkl'
    save_emb_path = data_dir + 'emb_deepwalk.pkl'
    # save_emb_path = data_dir + 'emb_n2v1.pkl'

    # train embeddings
    is_train_emb = False
    if not os.path.exists(save_emb_path) or is_train_emb:
        train_embeddings(adj_path, cfg.dim_feature, save_emb_path)
    # train mapping function
    train_align(*load_data(data_dir, save_emb_path, args.ratio), args)

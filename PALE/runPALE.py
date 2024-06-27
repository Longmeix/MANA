import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from time import time
from pale import PALE
import random
import numpy as np
import torch
import argparse
import os
from utils.general import read_pickle, str2bool
import config as cfg


def seed_torch(seed=2022):
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_data(data_dir, ratio):
    adj_s = read_pickle(data_dir + 'adj_s.pkl')
    adj_t = read_pickle(data_dir + 'adj_t.pkl')
    adj_s = ((adj_s + adj_s.T) > 0) * 1.
    adj_t = ((adj_t + adj_t.T) > 0) * 1.
    links = read_pickle(data_dir + f'links_{ratio}.pkl')
    # links = read_pickle(data_dir + f'links_degree.pkl')

    return adj_s, adj_t, links


def parse_args():
    parser = argparse.ArgumentParser(description="Network alignment")
    parser.add_argument('--model', default='PALE', type=str)
    parser.add_argument('--folder_dir', default='./dataset/', type=str)
    parser.add_argument('--dataset', default='FT', help='FT, DBLP', type=str)
    parser.add_argument('--train_emb', default=False, help='if train embeds', type=bool)
    parser.add_argument('--adapt', default='True', type=str2bool)
    parser.add_argument('--support', default='similarity', type=str, help='random/neighbor/similarity')
    parser.add_argument('--fast_lr', default=0.1, type=float)
    parser.add_argument('--n_way', default=5, type=int)
    parser.add_argument('--k_shot', default=5, type=int)
    parser.add_argument('--ratio', default=0.5, type=float)
    parser.add_argument('--save_embeds_file', default='embeds_pale_{}_meta.pkl', help='file type: .pkl', type=str)
    parser.add_argument('--save_best_embeds_path', default='best_embeds_pale.pkl', help='file type: .pkl', type=str)
    parser.add_argument('--seed', default=2023, type=int)
    parser.add_argument('--device', default='cuda:3', type=str)
    parser.add_argument('--top_k', default=cfg.k, type=int)

    parser.add_argument('--learning_rate1', default=0.01, type=float)
    parser.add_argument('--embedding_dim', default=cfg.dim_feature, type=int)
    parser.add_argument('--batch_size_embedding', default=512, type=int)
    parser.add_argument('--embedding_epochs', default=200, type=int)
    parser.add_argument('--neg_sample_size', default=5, type=int)
    parser.add_argument('--learning_rate2', default=5e-4, type=float)
    parser.add_argument('--batch_size_mapping', default=32, type=int)
    parser.add_argument('--mapping_epochs', default=100, type=int)
    parser.add_argument('--mapping_model', default='linear')
    parser.add_argument('--activate_function', default='sigmoid')

    args = parser.parse_args()
    print(args)
    return args


if __name__ == '__main__':
    # os.chdir('../')
    # seed_torch(2023)
    args = parse_args()
    cfg.init_args(args)

    data_dir = args.folder_dir + args.dataset + '/'
    # source_dataset = Dataset(args.source_dataset)
    # target_dataset = Dataset(args.target_dataset)
    # source_nodes = source_dataset.G.nodes()
    # target_nodes = target_dataset.G.nodes()
    # groundtruth_matrix = graph_utils.load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx)

    model = PALE(*load_data(data_dir, args.ratio), args)

    start_time = time()

    S = model.align(args.train_emb)
    # get_statistics(S, groundtruth_matrix)
    print("Full_time: ", time() - start_time)
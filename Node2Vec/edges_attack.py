import os
import random
import argparse
import numpy as np
import torch
from deeprobust.graph.global_attack.random_attack import Random
from deeprobust.graph.targeted_attack import Nettack
from utils.general import read_pickle, write_pickle
import scipy.sparse as sp
import copy


def seed_torch(seed=2021):
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# split head vs tail nodes
def split_nodes(adj, low=5, high_ratio=0.1):
    if not isinstance(adj, np.ndarray):
        adj = adj.toarray()
    num_links = np.sum(adj, axis=1)
    head_nodes_num = int(len(num_links) * high_ratio)
    high_degree = np.partition(num_links, kth=-head_nodes_num)[-head_nodes_num]  # select head nodes by the degree
    idx_high = np.where(num_links > high_degree)[0]
    idx_tail = np.where(num_links <= low)[0]

    return idx_high, idx_tail


class RandomAttack:
    def __init__(self, nnodes=None, attack_structure=True, device='cpu'):
        # super(RandomAttack, self).__init__(nnodes, attack_structure=attack_structure, device=device)
        self.nnodes = nnodes
        self.attack_structure = attack_structure
        self.device = device
        self.modified_adj = None

    def sample_forever(self, adj, target, exclude):
        """Randomly random sample edges from adjacency matrix, `exclude` is a set
        which contains the edges we do not want to sample and the ones already sampled
        """
        while True:
            # t = tuple(np.random.randint(0, adj.shape[0], 2))
            # t = tuple(random.sample(range(0, adj.shape[0]), 2))
            t = tuple(np.random.choice(adj.shape[0], 2, replace=False))
            if t not in exclude:
                yield t
                exclude.add(t)
                exclude.add((t[1], t[0]))

    def attack(self, adj, target_nodes, ptb_rate, type='add'):
        modified_adj = adj.tolil()
        # sample edges to add
        nonzero = set(zip(*adj.nonzero()))

        edges = self.random_sample_edges(adj, target_nodes, ptb_rate, exclude=nonzero)
        for n1, n2 in edges:
            modified_adj[n1, n2] = 1
            modified_adj[n2, n1] = 1
        self.check_adj(modified_adj)
        return modified_adj

    def check_adj(self, adj):
        """Check if the modified adjacency is symmetric and unweighted.
        """
        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.tocsr().max() == 1, "Max value should be 1!"
        assert adj.tocsr().min() == 0, "Min value should be 0!"

    def random_sample_edges(self, adj, target_nodes_list, ptb_rate, exclude):
        degrees = adj.sum(0).A1
        for target_node in target_nodes_list:
            cnt = 0
            n_perturbation = int(degrees[target_node] // 2 * ptb_rate)
            while cnt < n_perturbation:
                t = (np.random.choice(adj.shape[0], 1)[0], target_node)
                if t not in exclude:
                    yield t
                    exclude.add(t)
                    exclude.add((t[1], t[0]))
                    cnt += 1
        add_edges = exclude - set(zip(*adj.nonzero()))
        return add_edges


def random_add_edges(adj, ptb_rate, high_ratio=0.1):
    # perturb fans of high-degree nodes
    target_nodes, _ = split_nodes(adj, high_ratio=high_ratio)
    attacker = RandomAttack(attack_structure=True)
    perturbed_adj = attacker.attack(adj, target_nodes, ptb_rate, type='add')

    perturb_edges_set = set(zip(*perturbed_adj.nonzero())) - set(zip(*adj.nonzero()))
    print('Add edges: {}'.format(len(perturb_edges_set)))
    perturbed_adj = sp.csr_matrix(perturbed_adj)
    return perturbed_adj, perturb_edges_set


# def nettack_add_edges(adj, ptb_rate, high_ratio=0.1):
#     degrees = adj.sum(0).A1
#     target_nodes_list, _ = split_nodes(adj, high_ratio=high_ratio)
#     num = len(target_nodes_list)
#     print('=== Attacking %s nodes sequentially ===' % num)
#     modified_adj = adj
#     for target_node in target_nodes_list:
#         n_perturbations = int(degrees[target_node])
#         surrogate = None
#         features, labels = sp.random(2, 3, 0.5, 'csr'), None
#         model = Nettack(surrogate, nnodes=modified_adj.shape[0], attack_structure=True, attack_features=False,
#                         device=args.device)
#         model = model.to(args.device)
#         model.attack(features, modified_adj, labels, target_node, n_perturbations, verbose=False)
#         modified_adj = model.modified_adj
#     print('Add edges: {}'.format(modified_adj.nnz - adj.nnz))
#     perturbed_adj = sp.csr_matrix(modified_adj)
#     return perturbed_adj


def delete_edges4tail(row_adj, ptb_adj, idx_high, idx_low):
    row_adj = row_adj.toarray()
    adj = copy.deepcopy(row_adj)
    len_row_adj = len(adj.nonzero()[0])
    num_links = np.sum(adj, axis=1)
    # delete one edge on tail nodes despite degree(i==1)
    idx_2to5 = np.where((1 < num_links) & (num_links <= 5))[0]
    ptb_ratio_tail = args.ptb_rate
    size_del_edge4tail = int(ptb_ratio_tail * len(idx_2to5))
    idx_mask_dict = dict(zip(*adj.nonzero()))
    for r in np.random.choice(idx_2to5, size_del_edge4tail, replace=False):
        c = idx_mask_dict.get(r)
        adj[r, c] = 0
        adj[c, r] = 0

    adj[idx_high] = ptb_adj.toarray()[idx_high]  # perturb high degree nodes
    single_n = np.where(~adj.any(axis=1))[0]  # isolated nodes
    adj[single_n] = row_adj[single_n]
    adj = ((adj + adj.T) > 0) * 1.
    len_ptb_adj = len(adj.nonzero()[0])
    assert len(np.where(~adj.any(axis=1))[0]) == 0

    print('row adj: {}, perturbed adj: {}'.format(len_row_adj, len_ptb_adj))
    return sp.csr_matrix(adj)


# 4. check whether degree(i) = 0
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='FT', help='dataset name, {FT}')
    parser.add_argument('--ptb_rate', type=float, default=0.05, help='noise ptb_rate')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--seed', type=int, default=2022, help='Random seed.')

    args = parser.parse_args()
    seed_torch(2022)
    ptb_rate = args.ptb_rate
    folder = os.path.join('../../dataset', args.dataset)
    save_ptb_dir = os.path.join(folder, 'attack', str(ptb_rate))
    if not os.path.exists(save_ptb_dir):
        os.makedirs(save_ptb_dir)

    for net in ['s', 't']:
        adj_path = os.path.join(folder, f'adj_{net}.pkl')
        adj_ptb_path = os.path.join(save_ptb_dir, f'adj_ptb_{net}.pkl')
        # 1. original file
        adj = read_pickle(adj_path)
        adj = ((adj + adj.T) > 0) * 1.
        idx_high, idx_tail = split_nodes(adj)
        # 2. add edges
        perturbed_adj, perturb_edges_set = random_add_edges(adj, ptb_rate, high_ratio=0.1)
        # perturbed_adj = nettack_add_edges(adj, ptb_rate)

        # 3. random delete edges on tail nodes despite degree(i) = 1
        ptb_adj = delete_edges4tail(adj, perturbed_adj, idx_high, idx_tail)
        # write_pickle(perturbed_adj, adj_ptb_path)

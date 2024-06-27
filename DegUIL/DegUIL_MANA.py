import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from itertools import chain
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import argparse
from utils import data_process
from models.deggnn import DegGNN
from models.UILAggregator import UILAggregator, MLP
import deguil_config as cfg
import learn2learn as l2l
from learn2learn.algorithms import maml_update
from torch import autograd
from utils.general import str2bool


# Get parse argument
parser = argparse.ArgumentParser()
parser.add_argument('--folder_dir', default='./dataset/', type=str)
parser.add_argument("--dataset", type=str, default='FT', help='dataset: FT, DBLP')
parser.add_argument("--model", type=str, default='DegUIL', help='model name: DegUIL, Tail-GNN')
parser.add_argument('--ratio', default=0.5, type=float)
parser.add_argument("--gnn_type", type=int, default=3, help='1: gcn, 2: gat, 3: gcn+gat')
parser.add_argument("--hidden", type=int, default=64, help='hidden layer dimension')
# parser.add_argument("--batch_size", type=int, default=256, help='batch size')
parser.add_argument("--mu", type=float, default=0.001, help='missing/redundant info constraint')
parser.add_argument("--dropout", type=float, default=0.5, help='dropout')
parser.add_argument("--D", type=int, default=5, help='num of node neighbor')
parser.add_argument("--lr", type=float, default=5e-4, help='learning rate')
parser.add_argument("--seed", type=int, default=2022, help='Random seed')
parser.add_argument("--epochs", type=int, default=40, help='Epochs')  # 10
parser.add_argument("--device", type=str, default='cuda:3', help='gpu id or cpu')
# MANA
parser.add_argument('--adapt', default='True', type=str2bool)
# parser.add_argument('--adapt', default=True, type=bool)
# parser.add_argument('--adapt', default=False, type=bool)
parser.add_argument('--meta_test_cluster', default=1, type=int)  # >1 : using clustering meta-test tasks
parser.add_argument('--support', default='similarity', type=str, help='random/neighbor/similarity')
parser.add_argument('--fast_lr', default=0.1, type=float)
parser.add_argument('--n_way', default=5, type=int)
parser.add_argument('--k_shot', default=5, type=int)
parser.add_argument('--meta_bsz', default=4, type=int)
parser.add_argument('--sim_metric', default='cosine', type=str)

args = parser.parse_args()
cfg.init_args(args)
dataset = args.dataset
model = args.model
DEVICE = args.device
# print(args.support)
# DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f'dataset:{dataset}, model:{args.model}')

# os.chdir('../')  # 使用run.py运行DegUIL_MANA.py时注释此行
cfg.seed_torch(args.seed)
# print(os.getcwd())

def mse_loss(emb_h, emb_t, idx):
    loss = torch.mean(torch.norm(emb_h[idx] - emb_t[idx]))
    return loss


def fast_adapt(task, learner, data, adaptation_steps=1):
    support_links, query_links = task
    # fast_lr = 0.1
    if args.adapt:
        support_links = list(filter(None, support_links))
        # Adapt the model
        if len(support_links) > 0:
            for step in range(adaptation_steps):
                train_loss = uil_h.cal_loss(support_links, learner.module, *data)
                # learner.adapt(train_loss)
                grads_theta = autograd.grad(train_loss,
                                            learner.module[0].parameters(),
                                            retain_graph=False, create_graph=False, allow_unused=True)
                learner.module[0] = maml_update(learner.module[0], lr=args.fast_lr, grads=grads_theta)
                # learner.module[0] = maml_update(learner.module[0], lr=fast_lr, grads=grads_theta)
                # grads_theta = autograd.grad(train_loss,
                #                             learner.module[0].model[2].parameters(),
                #                             retain_graph=False, create_graph=False, allow_unused=True)
                # learner.module[0].model[2] = maml_update(learner.module[0].model[2], lr=args.fast_lr,
                #                                          grads=grads_theta)

    # Evaluate the adapted model
    valid_loss = uil_h.cal_loss(query_links, learner.module, *data)

    return valid_loss


def train_embed():
    with torch.no_grad():
        embed_norm_s, mis_n_s, rdd_n_s = embed_model(features[0], adj_s, deg='norm')
    if args.support == 'similarity':
        meta_batches = uil_h.meta_train.get_meta_batches(embed_norm_s.detach().cpu().numpy(), args.meta_bsz, sim_metric=args.sim_metric)
    else:
        meta_batches = uil_h.meta_train.get_meta_batches(args.meta_bsz)  # neighbor, random

    loss_epc = .0
    print_every = len(meta_batches) // 4

    for i, meta_batch in enumerate(meta_batches):
        learner = maml_map.clone()
        loss_batch = 0.

        embed_super_s, mis_s_s, _ = embed_model(features[0], adj_s, deg='super')
        embed_super_t, mis_s_t, _ = embed_model(features[1], adj_t, deg='super')
        embed_norm_s, mis_n_s, rdd_n_s = embed_model(features[0], adj_s, deg='norm')
        embed_norm_t, mis_n_t, rdd_n_t = embed_model(features[1], adj_t, deg='norm')
        embed_tail_s, _, rdd_t_s = embed_model(features[0], tail_adj_s, deg='tail')
        embed_tail_t, _, rdd_t_t = embed_model(features[1], tail_adj_t, deg='tail')

        data = (embed_super_s, mis_s_s, embed_super_t, mis_s_t,
                 embed_norm_s, mis_n_s, rdd_n_s, embed_norm_t, mis_n_t, rdd_n_t,
                 embed_tail_s, rdd_t_s, embed_tail_t, rdd_t_t)

        # ========= get meta batch data ===========
        for task in meta_batch:
            loss_batch += fast_adapt(task, learner, data)
        loss_batch /= len(meta_batch)
        loss_epc += loss_batch

        # update parameters of net
        optimizer.zero_grad()
        loss_batch.backward()
        optimizer.step()

        if (i+1) % print_every == 0:
            print('Iteration: {}/{} | loss: {}'.format(i, len(meta_batches), loss_batch))
        del data

    print('Epoch: {}, loss: {:.4f}'.format(epoch, loss_epc.item()))

    return loss_epc


features, (adj_s, adj_t), links, idx_s, idx_t = data_process.load_dataset(dataset, model, args.ratio, folder=args.folder_dir, k=args.D)
features = [emb.to(cfg.device) for emb in features]

# forge tail node
tail_adj_s = data_process.link_dropout(adj_s, idx_s[0])
extend_adj_s = data_process.extend_edges(adj_s, idx_s[0])
adj_s = torch.FloatTensor(adj_s.todense()).to(DEVICE)
extend_adj_s = torch.FloatTensor(extend_adj_s.todense()).to(DEVICE)
tail_adj_s = torch.FloatTensor(tail_adj_s.todense()).to(DEVICE)
tail_adj_t = data_process.link_dropout(adj_t, idx_t[0])
extend_adj_t = data_process.extend_edges(adj_t, idx_t[0])
adj_t = torch.FloatTensor(adj_t.todense()).to(DEVICE)
extend_adj_t = torch.FloatTensor(extend_adj_t.todense()).to(DEVICE)
tail_adj_t = torch.FloatTensor(tail_adj_t.todense()).to(DEVICE)

# Train model
t_total = time.time()
uil_h = UILAggregator(adj_s, adj_t, links, cfg.k, args)

# indexes of train nodes and test nodes
args.idx_norm_s = torch.LongTensor(idx_s[0])
args.idx_super_s = torch.LongTensor(idx_s[1])
args.idx_tail_s = torch.LongTensor(idx_s[2])
args.idx_norm_t = torch.LongTensor(idx_t[0])
args.idx_super_t = torch.LongTensor(idx_t[1])
args.idx_tail_t = torch.LongTensor(idx_t[2])

print("Data Processing done!")

out_dim = cfg.dim_feature

# Model and optimizer
embed_model = DegGNN(nfeat=features[0].shape[1], out_dim=out_dim, params=args, device=DEVICE).to(DEVICE)

dim = out_dim * 2

mapping = nn.ModuleList([
            MLP(dim, cfg.dim_feature),
            MLP(dim, cfg.dim_feature)
        ]).to(DEVICE)

maml_map = l2l.algorithms.MAML(mapping, lr=args.fast_lr, first_order=True)

optimizer = optim.Adam(
            chain(embed_model.parameters(),
                  maml_map.parameters(),
                  ),
            lr=args.lr,
            # weight_decay=args.lamda
            )


print_info_epochs = 10
loss_epc = .0
time0 = time.time()
for epoch in range(args.epochs):
    t = time.time()

    embed_model.train()
    mapping.train()

    loss_batch = train_embed()
    loss_epc += loss_batch

    # Evaluation:
    embed_model.eval()
    mapping.eval()
    embed_super_s, _, _ = embed_model(features[0], adj_s, deg='super')
    embed_super_t, _, _ = embed_model(features[1], adj_t, deg='super')
    embed_tail_s, _, _ = embed_model(features[0], adj_s, deg='tail')
    embed_tail_t, _, _ = embed_model(features[1], adj_t, deg='tail')
    embed_tail_s[args.idx_super_s] = embed_super_s[args.idx_super_s]
    embed_tail_t[args.idx_super_t] = embed_super_t[args.idx_super_t]
    uil_h.eval_hit(epoch, embed_tail_s, embed_tail_t, maml_map)
time1 = time.time()

uil_h.print_performance_k(time1-time0)

print("Training Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

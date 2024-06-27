import time
from itertools import chain
import numpy as np
import torch
from torch import nn
from scipy.sparse import csr_matrix
import torch.optim as opt
from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange, tqdm
import config as cfg
import utils.extensions as npe
from models.loss import NSLoss
from models.base import DataWrapper
from models.base import UIL
from prep_meta_tasks.dataProcess import DataProcess
from utils.general import write_pickle, read_pickle
import learn2learn as l2l
from torch.distributions import Beta
# from prep_meta_tasks.task_generate import MetaTaskRandom, MetaTaskSimilarity, MetaTaskNeighbor
import torch.nn.functional as F

net_key = ['s', 't']


class MANA(UIL):
    def __init__(self, embeds, adj_s, adj_t, links, args):
        super(MANA, self).__init__(links, k=args.top_k)
        # s,t网络的边属性字典 {'s':{pairs, weight, adj_mat}, 't':{}}
        # s,t网络的边属性字典 {'s':{pairs, weight, adj_mat}, 't':{}}
        shape = adj_s.shape[0], adj_t.shape[0]  # (9734,9514)
        self.device = args.device
        self.log.info(args)

        if links is not None:
            link_train, link_test = links
            link_mat = npe.pair2sparse(link_train, shape)  # s-->t有link的矩阵位置为1 (9734,9514)
            self.link_attr_train = self.addEdgeAttr(link_mat)  # real links of train set
        # s,t两个网络中 每一列是一条表边(vi,vj) edge['s'].size (2, E)
        # self.edge_idx = dict(s=self.adj2edge(adj_s), t=self.adj2edge(adj_t))
        self.links_train = link_train
        self.links_test = link_test

        # transfer embeds type to tensor
        if isinstance(embeds[0], torch.Tensor):
            self.embeds = dict(zip(net_key, (emb.to(self.device) for emb in embeds)))
            # self.embeds = dict(zip(net_key, (F.normalize(emb, dim=1, p=2).to(self.device) for emb in embeds)))
        else:
            self.embeds = dict(zip(net_key, (torch.tensor(emb).float().to(self.device) for emb in embeds)))
            # self.embeds = dict(zip(net_key, (F.normalize(torch.tensor(emb).float(), dim=1, p=2).to(self.device) for emb in embeds)))

        self.pairs_l, self.wgt_l, self.adj_l = self.link_attr_train.values()  # link match
        self.adj_s, self.adj_t = adj_s, adj_t

        dim = args.embed_dim
        self.adapt = args.adapt
        self.meta_bsz = args.meta_bsz
        self.fast_lr = args.fast_lr
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.support = args.support  # ['random', 'neighbor', 'similarity']
        self.meta_test_cluster = args.meta_test_cluster
        self.save_best_embeds_path = cfg.save_best_embeds_path
        self.save_adapt_best_embeds_path = cfg.save_adapt_best_embeds_path
        self.save_model_path = cfg.save_model_path

        self.mapping = MLP(dim, dim).to(self.device)

        # self.line = MLP(1, 1).to(self.device)

        self.opt_map = opt.Adam(
            chain(self.mapping.parameters(),
                  ),
            lr=args.mapping_lr,
            # weight_decay=1e-4
        )

        self.loss_label = NSLoss(
            sim=nn.CosineSimilarity(),
            # act=nn.ReLU(),
            loss=nn.MSELoss(reduction='sum')
        )
        self.mse = nn.MSELoss()

    @staticmethod
    def addEdgeAttr(adj_mat, exponent=3/4, percent=cfg.percent):
        """
       Given a similarity matrix, create weights for negative sampling and indices of similar users.
       Args:
           mat: similarity matrix
           exponent: a coefficient to downplay the similarities to create negative sampling weights, default: 3/4 (as suggested by word2vec)
           percent: percent of users to filter, range in [0, 100]
       Return:
           pairs: user pairs with high similairties
           weights: negative sampling weights
           mat: similarity matrix
       """
        if not isinstance(adj_mat, np.ndarray):
            adj_mat = adj_mat.toarray()
        weights = np.abs(adj_mat > 0).sum(axis=0) ** exponent
        clamp = npe.clamp_mat(adj_mat, percent)
        pairs = [i.tolist() for i in clamp.nonzero()]
        pairs = list(zip(*pairs))

        attr_keys = ['pairs', 'weights', 'adj_mat']
        attr_value = pairs, weights, csr_matrix(adj_mat)
        return dict(zip(attr_keys, attr_value))

    def adj2edge(self, adj_mat):
        # get edge(vi, vj) from adjacent matrix
        # size (2, E)
        return torch.tensor(list(zip(*adj_mat.nonzero()))).long().t().to(self.device)

    def get_embeds(self, is_map=False):
        # get node embedding of two networks
        # @is_map: map source embeddings or not
        embed_s, embed_t = self.embeds['s'], self.embeds['t']
        if is_map:
            embed_s = self.mapping(embed_s)
        return embed_s, embed_t

    def get_sims(self):
        f_s, f_t = self.get_embeds(is_map=True)
        sims = self.sim_pairwise(f_s, f_t)
        return sims

    def match_loss(self, embed_s, embed_t, mapping, pair, weight, adj_mat):
        idx_s, idx_t = [list(i) for i in list(zip(*pair))]
        # idx_s, idx_t = pair
        loss_batch = self.loss_label(
            embed_s, embed_t,
            idx_s, idx_t,
            mapping,
            # mapping[1],
            lambda x: x,
            cfg.neg_num,
            weight,
            adj_mat,
        )
        return loss_batch

    def fast_adapt(self, task, learner, adaptation_steps=1):
        emb_s, emb_t = self.get_embeds(is_map=False)
        support_links, query_links = task
        # support_links = query_links  # tr-tr

        support_links = list(filter(None, support_links))
        # Adapt the model
        if len(support_links) > 0:
            for step in range(adaptation_steps):
                train_loss = self.match_loss(emb_s, emb_t, learner.module, support_links,
                            self.wgt_l, self.adj_l)
                learner.adapt(train_loss)

        # Evaluate the adapted model
        valid_loss = self.match_loss(emb_s, emb_t, learner.module, query_links,
                        self.wgt_l, self.adj_l)

        return valid_loss

    def hit_test_adapt(self, test_batches, base_mdoel, top_k=30, adaptation_steps=2):
        test_losses = []
        cnt = 0
        test_tasks = test_batches[0]
        emb_s, emb_t = self.get_embeds(is_map=False)
        emb_s_adapt_map = emb_s.clone()

        for i in trange(len(test_tasks)):
            task = test_tasks[i]
            learner = base_mdoel.clone()
            support_link, query_link = task

            support_link = list(filter(None, support_link))
            if len(support_link) > 0:
                for step in range(adaptation_steps):
                    train_loss = self.match_loss(emb_s, emb_t, learner.module, support_link,
                                                 self.wgt_l, self.adj_l)
                    learner.adapt(train_loss)

            # Evaluate the adapted model
            with torch.no_grad():
                emb_s_map = learner.module(emb_s)
                test_loss = self.match_loss(emb_s, emb_t, learner.module, query_link,
                            self.wgt_l, self.adj_l)

            test_losses.append(test_loss.item())
            test_num = len(query_link)
            cnt += test_num

            test_sid, test_tid = [list(i) for i in list(zip(*query_link))]
            emb_s_adapt_map[test_sid] = emb_s_map[test_sid]

        mrr, hit_p, success_p = self.eval_metrics_na(emb_s_adapt_map, emb_t, self.k)
        test_loss = np.mean(test_losses)
        self.log.info('-- Test loss: {:.4f}, MRR: {:.4f}, Hit: {:.4f}, Success:{:.4f}  --'.format(
            test_loss, mrr, hit_p, success_p
        ))

        return test_loss, mrr, hit_p, success_p, (emb_s_adapt_map, emb_t)

    def train_mapping(self, n_epochs):
        hit_best, hit_best_adapt = 0., 0.
        mrr_best, mrr_best_adapt = 0., 0.
        success_best, success_best_adapt = 0., 0.
        meta_bsz = self.meta_bsz   # number of tasks per batch
        n_way = self.n_way
        k_shot = self.k_shot
        time_local_mapping, time_global_mapping = [], []

        if not self.adapt or self.meta_test_cluster == 1:
            from prep_meta_tasks.task_generate import MetaTaskRandom, MetaTaskNeighbor, MetaTaskSimilarity
        else:
            from prep_meta_tasks.meta_tasks_cluster import MetaTaskRandom, MetaTaskNeighbor, MetaTaskSimilarity

        if self.support == 'random':
            meta_train = MetaTaskRandom(self.links_train, self.links_train, n_way=n_way, k_shot=k_shot, q_query=1)
            meta_test = MetaTaskRandom(self.links_test, self.links_train, n_way=1, k_shot=k_shot, q_query=1)
            test_batches = meta_test.get_test_batches()
        elif self.support == 'neighbor':
            meta_train = MetaTaskNeighbor(self.adj_s, self.links_train, self.links_train, n_way=n_way, k_shot=k_shot, q_query=1)
            meta_test = MetaTaskNeighbor(self.adj_s, self.links_test, self.links_train, n_way=1, k_shot=k_shot, q_query=1)
            test_batches = meta_test.get_test_batches()
        elif self.support == 'similarity':
            embed_s_np = self.embeds['s'].detach().cpu().numpy()
            meta_train = MetaTaskSimilarity(self.links_train, self.links_train, n_way=n_way, k_shot=k_shot, q_query=1)
            meta_test = MetaTaskSimilarity(self.links_test, self.links_train, n_way=self.meta_test_cluster, k_shot=k_shot, q_query=1)
            test_batches = meta_test.get_test_batches(embed_s_np)
        else:
            print('Support Strategy Error')

        maml_map = l2l.algorithms.MAML(self.mapping, lr=self.fast_lr, first_order=True)

        self.log.info('Starting training...')
        time4training_start = time.time()
        for epoch in range(1, n_epochs + 1):
            self.mapping.train()
            loss_epc = .0
            if self.support == 'similarity':
                meta_batches = meta_train.get_meta_batches(embed_s_np, meta_bsz)
            else:
                meta_batches = meta_train.get_meta_batches(meta_bsz)  # neighbor, random

            for b_idx in range(len(meta_batches)):
                meta_batch = meta_batches[b_idx]
                learner = maml_map.clone()
                loss_batch = 0.
                # ========= get meta batch data ===========
                for task in meta_batch:
                    loss_batch += self.fast_adapt(task, learner)
                loss_batch /= meta_bsz
                loss_epc += loss_batch
                # update parameters of net
                self.optimize(self.opt_map, loss_batch)

            self.log.info('Epoch: {}, loss: {:.4f}'.format(epoch, loss_epc.item() / len(meta_batches)))

            # ======= evaluate ==========
            self.mapping.eval()
            emb_s_map, emb_t_map = self.get_embeds(is_map=True)
            time_com = time.time()
            mrr, hit_p, success_p = self.eval_metrics_na(emb_s_map, emb_t_map, self.k)
            time_global_mapping.append(round(time.time() - time_com, 2))
            if mrr > mrr_best:
                hit_best = hit_p
                mrr_best = mrr
                success_best = success_p
                if not self.adapt:
                    write_pickle([emb_s_map, emb_t_map], self.save_best_embeds_path)
            self.log.info('== Common best == MRR: {:.4f}, Hit: {:.4f}, Success:{:.4f}'.format(mrr_best, hit_best, success_best))

            if self.adapt and epoch > n_epochs-30 and mrr > mrr_best-0.005:  # FT
            # if self.adapt and epoch > n_epochs-5 and mrr > mrr_best-0.005:
            # if self.adapt and mrr > mrr_best-0.005:
                # adaptive testing
                time0 = time.time()
                test_loss, mrr, hit_p, success_p, best_adapt_map_emb = self.hit_test_adapt(test_batches, maml_map,
                                                                          self.k, adaptation_steps=2)   # adaptation=1 for DW15K
                local_time = round(time.time() - time0, 2)
                time_local_mapping.append(local_time)
                # print('Time of local mapping: {}'.format(local_time))
                if mrr > mrr_best_adapt:
                    mrr_best_adapt = mrr
                    hit_best_adapt = hit_p
                    success_best_adapt = success_p
                    # torch.save(maml_map.state_dict(), self.save_model_path)
                    write_pickle(best_adapt_map_emb, self.save_adapt_best_embeds_path)
                self.log.info('== Adapt best == MRR: {:.4f}, Hit: {:.4f}. Success: {:.4f} ==='.format(mrr_best_adapt, hit_best_adapt, success_best_adapt))

        time4training_all = time.time() - time4training_start
        self.log.info('Time for training: {:.2f}'.format(time4training_all))
        self.log.info('--- Final Results ---')
        if self.adapt:
            # maml_map.load_state_dict(torch.load(self.save_model_path))
            # maml_map.eval()
            emb_s, emb_t = read_pickle(self.save_adapt_best_embeds_path)
            self.report_hit(emb_s, emb_t)
            self.log.info('Time of local mapping: {:.2f}'.format(np.mean(time_local_mapping)))
            self.rename_log('/mrr{:.2f}_{}_'.format(mrr_best_adapt * 100, self.adapt).join(
                cfg.log.split('/')
            ))
        else:
            emb_s, emb_t = read_pickle(self.save_best_embeds_path)
            self.report_hit(emb_s, emb_t)
            self.log.info('Time of global mapping: {:.2f}'.format(np.mean(time_global_mapping)))
            self.rename_log('/mrr{:.2f}_{}_'.format(mrr_best * 100, self.adapt).join(
                cfg.log.split('/')
            ))


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out,
                 dim_hid=None, act=None):
        super(MLP, self).__init__()
        if act is None:
            act = nn.Tanh()
        if dim_hid is None:
            dim_hid = dim_in * 2   # 原文x_feature=64, dim_hid=128, 两倍关系
        # 2-layers
        self.model = nn.Sequential(
            # nn.Linear(dim_in, dim_hid),
            # # nn.Dropout(0.5),
            # act,
            # nn.Linear(dim_hid, dim_out)
            nn.Linear(dim_in, dim_out)
        )

    def forward(self, x):
        return self.model(x)
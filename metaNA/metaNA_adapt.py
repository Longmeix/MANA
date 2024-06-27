import math

from torch import Tensor
from tqdm import trange
import config as cfg
import torch
import torch.optim as optim
from itertools import chain
import torch.nn as nn
import torch.nn.functional as F
from utils.general import UILDataset
from utils.extensions import pair2sparse, cosine
import learn2learn as l2l
from learn2learn.data.task_dataset import TaskDataset
import numpy as np
from models.base import UIL
from utils.general import write_pickle, read_pickle
from torch import autograd
from learn2learn.algorithms import maml_update
from prep_meta_tasks.task_generate import MetaTaskRandom, MetaTaskNeighbor, MetaTaskSimilarity


net_name = ['s', 't']


class MetaNA(UIL):
    def __init__(self, embeds, adj_s, adj_t, links, nets, args):
        super(MetaNA, self).__init__(links, k=args.top_k)
        # transfer embeds type to tensor
        if isinstance(embeds[0], torch.Tensor):
            self.embeds = dict(zip(net_name, (emb.to(cfg.device) for emb in embeds)))
        else:
            self.embeds = dict(zip(net_name,
                                   [torch.tensor(emb).float().to(cfg.device)
                                    for emb in embeds]))
        # s,t两个网络中 每一列是一条表边(vi,vj) edge['s'].size (2, E)
        self.edge_index = dict(s=self.adj2edge(adj_s),
                               t=self.adj2edge(adj_t))
        # links: [train_links, test_links]
        self.links = links
        self.train_anchor_links, self.test_anchor_links = links
        # link matrix for train
        shape = adj_s.shape[0], adj_t.shape[0]  # (9734,9514)
        self.link_train_adjmat = pair2sparse(self.links[0], shape)  # (9734,9514)

        self.adj_t = adj_t
        self.nets = nn.ModuleList(nets).to(cfg.device)
        self.k = args.top_k
        self.adapt = args.adapt
        self.fast_lr = args.fast_lr
        self.k_shot = args.k_shot
        self.n_way = args.n_way
        self.meta_bsz = args.meta_bsz
        self.meta_lr = args.meta_lr
        # self.alpha = nn.Parameter(torch.tensor(0.5).to(cfg.device))
        self.support = 'similarity'
        self.save_best_embeds_path = cfg.save_best_embeds_path
        self.save_model_path = cfg.save_model_path

        self.opt_embed = optim.Adam(
            chain(self.nets.parameters()),
            lr=self.meta_lr,
            weight_decay=cfg.weight_decay
        )


    @staticmethod
    def adj2edge(adj_mat):
        # get edge(vi, vj) from adjacent matrix
        # size (2, E)
        return torch.tensor(list(zip(*adj_mat.nonzero()))).long().t().to(cfg.device)

    def get_embeds(self, is_map=True):
        embed_s, embed_t = [self.nets[0](self.embeds['s'], self.edge_index['s']),
                            self.nets[1](self.embeds['t'], self.edge_index['t'])
                            ]
        return embed_s, embed_t
        # return self.max_min_standard(embed_s), self.max_min_standard(embed_t)

    @staticmethod
    def max_min_standard(x):
        # Min-Max scaling
        min_x, _ = torch.min(x, dim=-1, keepdim=True)
        max_x, _ = torch.max(x, dim=-1, keepdim=True)
        x_scale = (x - min_x) / (max_x - min_x)
        return x_scale

    @staticmethod
    def distance_pairwise(xs, ys, scale=1):
        # xs /= scale
        # ys /= scale
        x_sq = (xs ** 2).sum(dim=1, keepdim=True)
        y_sq = (ys ** 2).sum(dim=1, keepdim=True)
        xy = xs @ ys.t()
        return (x_sq - 2 * xy + y_sq.t()).sqrt()
        # return max_min_standard(dis)

    def meta_loss(self, embed_s, embed_t, links):
        s_idx, t_idx = [list(i) for i in list(zip(*links))]
        x_s, x_t = embed_s[s_idx], embed_t[t_idx]
        # compute the distance between each support node and query set
        # dis = torch.vstack([F.pairwise_distance(x, x_t) for x in x_s])
        dis = torch.cat([self.distance_pairwise(x.unsqueeze(0), x_t)
                         for x in x_s], dim=0)
        dis = F.normalize(dis, dim=-1, p=2)
        # dis = self.max_min_standard(dis)
        # link_mat = link_adjmat[s_idx.tolist(), :]  # label matrix
        # link_mat = link_mat[:, t_idx.tolist()]
        # row, col = torch.tensor(list(zip(*link_mat.nonzero()))).long().t()
        # y_pred = self.sim_pairwise(x_s, x_t)  # 相似度还是欧式距离？
        # y = link_adjmat[s_idx, t_idx]  # label
        # loss_sameclass = dis[row, col].sum()
        diag = torch.diag(dis)  # 同一类的距离在对角线上
        loss_pos = diag.mean()
        non_diagonal = dis - torch.diag_embed(diag)
        n = len(non_diagonal)
        non_diagonal = non_diagonal.flatten()[:-1].view(n - 1, n + 1)[:, 1:].reshape(n, -1)
        # ep = 0.001
        ep = 0
        loss_neg = torch.log(torch.exp(-non_diagonal).mean(dim=0) * 5 + ep).mean()
        # dis[row, col] = 0  # erase the distance of link node(in same class)
        # exp每个元素，再对每一行(s_node与查询集的每个t_node)求和
        # loss_other = torch.log(torch.exp(-dis).sum(dim=1)).sum()
        # loss = loss_sameclass + loss_other/cfg.n_way # 加个比例？
        # ratio = 2 / len(links)  # FT: n_way
        ratio = 10
        loss = loss_pos + ratio * loss_neg
        # print('ratio:{:.2f}'.format(loss_neg/loss_pos))
        # print('Dis max: {}'.format(torch.max(dis).item()))
        # print('loss_pos = {:.2f}, loss_neg = {:.2f}, ratio = {:.2f}'.format(loss_pos.item(), loss_neg.item(), (loss_neg/loss_pos).item()))
        return loss

    @staticmethod
    def sample(neg_num: int, probs: Tensor, batch_size: int, node_num: int=16) ->Tensor:
        """
        Get indices of negative samples w.r.t. given probability distribution.
        Args:
            neg: number of negative samples
            probs: a probability vector for a multinomial distribution
            batch_size: batch size
            scale: maximum index, valid only when probs is None, leading to a uniform distribution over [0, scale - 1]
        Return:
            a LongTensor with shape [neg, batch_size]
        """
        assert neg_num > 0
        if probs is None:
            idx = torch.Tensor(batch_size * neg_num).uniform_(0, node_num).long()
        else:
            if not isinstance(probs, torch.Tensor):
                probs = torch.tensor(probs)
            idx = torch.multinomial(probs, batch_size * neg_num, True)  # true 有放回取样
        return idx.view(neg_num, batch_size)

    def loss_sim(self, link_adjmat, embed_s, embed_t, s_idx, t_idx, probs):
        s_idx, t_idx = s_idx.tolist(), t_idx.tolist()

        x_s, x_t = embed_s[s_idx], embed_t[t_idx]
        y_pos = torch.FloatTensor(link_adjmat[s_idx, t_idx]).squeeze()
        # negative sample
        idx_neg = self.sample(1, probs, cfg.n_way, len(embed_t))
        x_neg = torch.stack([embed_t[idx] for idx in idx_neg])
        x_t = torch.cat([x_t.unsqueeze(0), x_neg], dim=0)  # x_pos + x_neg for x_t
        y_neg = torch.stack([torch.FloatTensor(link_adjmat[s_idx, idx.tolist()])
                             for idx in idx_neg
        ]).view(-1)
        y = torch.cat([y_pos.view(-1), y_neg]).to(cfg.device)

        y_hat = F.relu(
                     torch.stack([F.cosine_similarity(x_s, x) for x in x_t
                             ]).view(-1)
                )
        loss = F.mse_loss(y_hat, y)
        return loss

    @staticmethod
    def get_pair_task(pairs, batch_size):
        '''
        @:pairs: sample pairs[(vs, vt), ...], size=cfg.batch_size
        @:return: batch [(vs1, vs2, ...), (vt1, ...)]
        '''
        idx = np.random.choice(range(len(pairs)), batch_size, replace=False)
        pairs = [pairs[i] for i in idx]
        link_task = [list(i) for i in zip(*pairs)]
        return link_task

    def get_embeds_learner(self, learner):
        embed_s, embed_t = [learner.module[0](self.embeds['s'], self.edge_index['s']),
                            learner.module[1](self.embeds['t'], self.edge_index['t'])
                            ]
        return embed_s, embed_t

    def fast_adapt(self, task, learner, adaptation_steps=1):
        support_links, query_links = task
        if self.adapt:
            support_links = list(filter(None, support_links))
            # Adapt the model
            if len(support_links) > 0:
                for step in range(adaptation_steps):
                    embed_s, embed_t = self.get_embeds_learner(learner)
                    train_loss = self.meta_loss(embed_s, embed_t, support_links)
                    # learner.adapt(train_loss)
                    grads_theta = autograd.grad(train_loss,
                                                learner.module[0].sgc.lin.parameters(),
                                                retain_graph=True, create_graph=True, allow_unused=False)
                    learner.module[0].sgc.lin = maml_update(learner.module[0].sgc.lin, lr=self.fast_lr,
                                                            grads=grads_theta)

        # Evaluate the adapted model
        embed_s, embed_t = self.get_embeds_learner(learner)
        valid_loss = self.meta_loss(embed_s, embed_t, query_links)

        return valid_loss

    def train(self, epochs):
        hit_best, hit_best_adapt = 0., 0.
        mrr_best, mrr_best_adapt = 0., 0.
        success_best, success_best_adapt = .0, .0
        num_tasks = epochs
        meta_bsz = self.meta_bsz  # number of tasks per batch
        # meta_lr = 0.001
        n_way = self.n_way
        best_test_batches = None

        if self.support == 'random':
            meta_train = MetaTaskRandom(self.train_anchor_links, self.train_anchor_links, n_way=5, k_shot=2, q_query=1)
            meta_test = MetaTaskRandom(self.test_anchor_links, self.train_anchor_links, n_way=1, k_shot=2, q_query=1)
            test_batches = meta_test.get_test_batches()
        elif self.support == 'neighbor':
            meta_train = MetaTaskNeighbor(self.adj_s, self.train_anchor_links, self.train_anchor_links, n_way=5,
                                          k_shot=2, q_query=1)
            meta_test = MetaTaskNeighbor(self.adj_s, self.test_anchor_links, self.train_anchor_links, n_way=1, k_shot=2,
                                         q_query=1)
            test_batches = meta_test.get_test_batches()
        elif self.support == 'similarity':
            meta_train = MetaTaskSimilarity(self.train_anchor_links, self.train_anchor_links, n_way=n_way, k_shot=self.k_shot,
                                            q_query=1)
            meta_test = MetaTaskSimilarity(self.test_anchor_links, self.train_anchor_links, n_way=1, k_shot=self.k_shot,
                                           q_query=1)
        else:
            print('Support Strategy Error')

        embed_s_np = self.embeds['s'].detach().cpu().numpy()
        test_batches = meta_test.get_test_batches(embed_s_np)

        maml_model = l2l.algorithms.MAML(self.nets, lr=self.fast_lr, first_order=True)
        optimizer = torch.optim.Adam(maml_model.parameters(), self.meta_lr)
        cnt = 0
        self.alpha = 50
        for epoch in range(num_tasks):
            loss_epc = .0
            loss_batch = .0
            # generate meta batches
            if self.support == 'similarity':
                meta_batches = meta_train.get_meta_batches(embed_s_np, meta_bsz)
            else:
                meta_batches = meta_train.get_meta_batches(meta_bsz)  # neighbor, random

            # link_task = self.get_pair_task(self.links[0], cfg.n_way)
            for meta_batch in meta_batches:
                learner = maml_model.clone()
                loss_batch = 0.
                for task in meta_batch:
                    loss_batch += self.fast_adapt(task, learner)
                loss_batch /= meta_bsz
                loss_epc += loss_batch
                # update parameters of net
                self.optimize(optimizer, loss_batch)

            self.log.info('Epoch: {}, loss: {:.4f}'.format(epoch, loss_epc.item() / len(meta_batches)))

            # eval
            self.nets.eval()
            emb_s_map, emb_t_map = self.get_embeds(is_map=True)
            mrr, hit_p, success_p = self.eval_metrics_na(emb_s_map, emb_t_map, self.k)
            # hit_p = self.hit_p_task(list(zip(*link_task)))  # evaluate一个task的任务
            if mrr > mrr_best:
                mrr_best = mrr
                hit_best = hit_p
                success_best = success_p
                if not self.adapt:
                    write_pickle([emb_s_map, emb_t_map], self.save_best_embeds_path)
            self.log.info('== Common best == MRR: {:.4f}, Hit: {:.4f}, Success:{:.4f}'.format(mrr_best, hit_best, success_best))

            if self.adapt and epoch > num_tasks-10 and mrr > mrr_best-0.05:  #
            # if self.adapt and (epoch+1)%5==0 and mrr > mrr_best-0.05:  #
                # adaptive testing
                # test_batches = meta_test.get_test_batches(emb_s_map.detach().cpu().numpy())
                test_loss, mrr, hit_p, success_p, _ = self.hit_test_adapt(test_batches, maml_model, adaptation_steps=2)
                if mrr > mrr_best_adapt:
                    hit_best_adapt = hit_p
                    mrr_best_adapt = mrr
                    success_best_adapt = success_p
                    best_test_batches = test_batches
                    torch.save(maml_model.state_dict(), self.save_model_path)
                self.log.info('== Adapt best == MRR: {:.4f}, Hit: {:.4f}. Success: {:.4f} ==='.format(mrr_best_adapt, hit_best_adapt, success_best_adapt))

        self.log.info('--- Final Results ---')
        if self.adapt:
            maml_model.load_state_dict(torch.load(self.save_model_path))
            maml_model.eval()
            res = self.hit_test_adapt(best_test_batches, maml_model)
            adapted_best_emb = res[-1]
            self.report_hit_multi(*adapted_best_emb)
            self.rename_log('/mrr{:.2f}_{}_'.format(mrr_best_adapt * 100, self.adapt).join(
                cfg.log.split('/')
            ))
        else:
            emb_s, emb_t = read_pickle(self.save_best_embeds_path)
            self.report_hit_multi(emb_s, emb_t)
            self.rename_log('/mrr{:.2f}_{}_'.format(mrr_best * 100, self.adapt).join(
                cfg.log.split('/')
            ))

    def hit_test_adapt(self, test_batches, base_mdoel, adaptation_steps=2):
        # emb_s, emb_t = self.get_embeds()
        # mrr_list, hit_list, success_list = [], [], []
        # hit_list_class = []
        test_losses = []
        cnt = 0
        test_tasks = test_batches[0]
        # test_tasks = [test_task.extend(b) for b in test_batches]
        emb_s, emb_t = self.get_embeds()

        for i in trange(len(test_tasks)):
            task = test_tasks[i]
            learner = base_mdoel.clone()
            support_link, query_link = task

            support_link = list(filter(None, support_link))
            if len(support_link) > 0:
                for step in range(adaptation_steps):
                    embed_s_adapt, embed_t_adapt = self.get_embeds_learner(learner)
                    train_loss = self.meta_loss(embed_s_adapt, embed_t_adapt, support_link)
                    # learner.adapt(train_loss)
                    grads_theta = autograd.grad(train_loss,
                                                learner.module[0].sgc.lin.parameters(),
                                                retain_graph=True, create_graph=True, allow_unused=False)
                    learner.module[0].sgc.lin = maml_update(learner.module[0].sgc.lin, lr=self.fast_lr,
                                                            grads=grads_theta)

            # Evaluate the adapted model
            with torch.no_grad():
                embed_s_adapt, embed_t_adapt = self.get_embeds_learner(learner)
                test_loss = self.meta_loss(embed_s_adapt, embed_t_adapt, query_link)
            test_losses.append(test_loss.item())
            test_num = len(query_link)
            cnt += test_num

            test_sid, test_tid = [list(i) for i in list(zip(*query_link))]
            # emb_s[test_sid], emb_t[test_tid] = embed_s_adapt[test_sid], embed_t_adapt[test_tid]
            emb_s[test_sid] = embed_s_adapt[test_sid]


        #
            # mrr, hit_p, success_p = self.calculate_hit(embed_s_adapt, embed_t_adapt,
        #                                          train_link=self.train_anchor_links, test_link=query_link,
        #                                          top_k=top_k)
        #     mrr_list.append(mrr.item() * test_num)
        #     hit_list.append(hit_p.item() * test_num)
        #     success_list.append(success_p.item() * test_num)
        #     hit_list_class.append(hit_p.item())
        #     # plt.bar(list(range(len(hit_list_class))), hit_list_class)
        # test_loss, mrr, hit_p, success_p = np.mean(test_losses), np.sum(mrr_list) / cnt, np.sum(hit_list) / cnt, np.sum(success_list) / cnt
        # self.log.info('-- Test loss: {:.4f}, MRR: {:.4f}, Hit: {:.4f}, Success:{:.4f}  --'.format(
        #     test_loss, mrr, hit_p, success_p
        # ))


        mrr, hit_p, success_p = self.eval_metrics_na(emb_s, emb_t, self.k)
        test_loss = np.mean(test_losses)

        return test_loss, mrr, hit_p, success_p, (emb_s, emb_t)
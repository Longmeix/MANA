import numpy as np
import torch
from torch import nn
from scipy.sparse import csr_matrix
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler
from tqdm import trange
import deguil_config as cfg
import utils.extensions as npe
from DegUIL.models.loss_neg import NSLoss
from DegUIL.models.base import DataWrapper
from DegUIL.models.base import UIL
from utils.general import write_pickle, read_pickle
from learn2learn.algorithms import maml_update
from torch import autograd
# from prep_meta_tasks.task_generate import MetaTaskRandom, MetaTaskNeighbor, MetaTaskSimilarity
# from DegUIL import cal_loss

net_key = ['s', 't']


class UILAggregator(UIL):
    def __init__(self, adj_s, adj_t, links, k, args):
        super(UILAggregator, self).__init__(links, k=k)
        self.log.info(args)
        adj_s = csr_matrix(adj_s.cpu().numpy())
        adj_t = csr_matrix(adj_t.cpu().numpy())
        self.edgeAttr = self.getEdgeAttr(adj_s, adj_t)
        shape = adj_s.shape[0], adj_t.shape[0]

        if links is not None:
            self.link_train, self.link_test = links
            link_mat = npe.pair2sparse(self.link_train, shape)
            self.link_attr_train = self.addEdgeAttr(link_mat)  # real links of train set
        # s,t两个网络中 每一列是一条表边(vi,vj) edge['s'].size (2, E)
        self.edge_idx = dict(s=self.adj2edge(adj_s), t=self.adj2edge(adj_t))

        self.pairs_s, self.wgt_s, self.adj_s = self.edgeAttr['s'].values()  # source net
        self.pairs_t, self.wgt_t, self.adj_t = self.edgeAttr['t'].values()  # target net
        self.pairs_l, self.wgt_l, self.adj_l = self.link_attr_train.values()  # link match

        self.hit_best, self.mrr_best, self.success_best = .0, .0, .0
        self.hit_best_adapt, self.mrr_best_adapt, self.success_best_adapt = .0, .0, .0
        self.args = args
        self.meta_test_cluster = args.meta_test_cluster

        self.save_best_embeds_path = cfg.save_best_embeds_path
        self.save_adapt_best_embeds_path = cfg.save_adapt_best_embeds_path
        self.save_model_path = cfg.save_model_path

        # loss
        self.loss_intra = NSLoss(
            act=nn.Sigmoid(),
            sim=nn.CosineSimilarity(),
            loss=nn.BCEWithLogitsLoss()
        )

        self.loss_cosine = NSLoss(
            sim=nn.CosineSimilarity(),
            loss=nn.MSELoss(reduction='sum')
        )
        self.mse = nn.MSELoss(reduction='mean')
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.support = args.support
        self.best_test_batches = []
        self.log.info(f'N-way={self.n_way}, K-shot={self.k_shot}')

        self.generate_meta_batch()

    def generate_meta_batch(self):
        if not self.args.adapt or self.meta_test_cluster == 1:
            from prep_meta_tasks.task_generate import MetaTaskRandom, MetaTaskNeighbor, MetaTaskSimilarity
        else:
            from prep_meta_tasks.meta_tasks_cluster import MetaTaskRandom, MetaTaskNeighbor, MetaTaskSimilarity

        if self.support == 'random':
            meta_train = MetaTaskRandom(self.link_train, self.link_train, n_way=self.n_way, k_shot=self.k_shot,
                                        q_query=1)
            meta_test = MetaTaskRandom(self.link_test, self.link_train, n_way=1, k_shot=self.k_shot,
                                       q_query=1)
        elif self.support == 'neighbor':
            meta_train = MetaTaskNeighbor(self.adj_s, self.link_train, self.link_train, n_way=self.n_way,
                                          k_shot=self.k_shot, q_query=1)
            meta_test = MetaTaskNeighbor(self.adj_s, self.link_test, self.link_train, n_way=1,
                                         k_shot=self.k_shot,
                                         q_query=1)
        elif self.support == 'similarity':
            meta_train = MetaTaskSimilarity(self.link_train, self.link_train, n_way=self.n_way,
                                            k_shot=self.k_shot,
                                            q_query=1)
            meta_test = MetaTaskSimilarity(self.link_test, self.link_train, n_way=self.meta_test_cluster, k_shot=self.k_shot,
                                           q_query=1)
        else:
            print('Support Strategy Error')
        self.meta_train, self.meta_test = meta_train, meta_test
        # return meta_train, meta_test

    @staticmethod
    def addEdgeAttr(adj_mat, exponent=3 / 4, percent=cfg.percent):
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

    @staticmethod
    def adj2edge(adj_mat):
        # get edge(vi, vj) from adjacent matrix
        # size (2, E)
        return torch.tensor(list(zip(*adj_mat.nonzero()))).long().t().to(cfg.device)

    def getEdgeAttr(self, adj_s, adj_t):
        # get [pair, weight, adj_mat] of network s and t
        edgeAttr = {'s': {}, 't': {}}
        edgeAttr['s'] = self.addEdgeAttr(adj_s)
        edgeAttr['t'] = self.addEdgeAttr(adj_t)
        return edgeAttr

    def get_sims(self):
        f_s, f_t = self.get_embeds(is_eval=True)
        sims = self.sim_pairwise(f_s, f_t)
        return sims

    @staticmethod
    def get_pair_batch(pairs, batch_size):
        '''
        @:pairs: sample pairs[(vs, vt), ...], size=cfg.batch_size
        @:return: batch [(vs1, vs2, ...), (vt1, ...)]
        '''
        idx = RandomSampler(pairs, replacement=True,
                            num_samples=batch_size)
        pair_list = [pairs[i] for i in list(idx)]
        data = DataWrapper(pair_list)
        batches = DataLoader(
            data, batch_size=batch_size,
            shuffle=True)
        _, batch = next(enumerate(batches))  # only one batch in batches
        return batch

    def global_loss(self, embed, pair, weight, adj_mat):
        idx_s, idx_t = pair
        loss_batch = self.loss_cosine(
            embed, embed,
            idx_s, idx_t,
            lambda x: x,
            lambda x: x,
            cfg.neg_num,
            weight,
            adj_mat,
            adj_mat,
            'deg'
        )

        return loss_batch

    def local_loss(self, embed, pair):
        idx_s, idx_t = pair
        loss_batch = self.mse(embed[idx_s], embed[idx_t])
        return loss_batch

    def embed_map_loss(self, embed_s, embed_t, l_pair, maps):
        idx_s, idx_t = l_pair
        source_emb, target_emb = embed_s[idx_s], embed_t[idx_t]
        source_emb_after_map = F.normalize(maps[0](source_emb))
        target_emb_after_map = F.normalize(maps[1](target_emb))
        loss_st = self.mse(source_emb_after_map, target_emb)
        loss_ts = self.mse(target_emb_after_map, source_emb)
        return loss_st + loss_ts

    def match_loss(self, embed_s, embed_t, common, pair, weight, adj_mat, adj_t):
        idx_s, idx_t = [torch.LongTensor(i) for i in list(zip(*pair))]

        loss_batch = self.loss_cosine(
            embed_s, embed_t,
            idx_s, idx_t,
            common[0],
            common[1],
            cfg.neg_num,
            weight,
            adj_mat,
            adj_t,
            'deg'
        )
        return loss_batch

    def train(self, embed_s, embed_t, anchor_link, common):
        # get batch data
        s_pair = self.get_pair_batch(self.pairs_s, cfg.batch_size)
        t_pair = self.get_pair_batch(self.pairs_t, cfg.batch_size)
        # l_pair = self.get_pair_batch(self.pairs_l, cfg.batch_size)

        # ========= global loss ==========
        loss_g_s = self.global_loss(embed_s, s_pair, self.wgt_s, self.adj_s)
        loss_g_t = self.global_loss(embed_t, t_pair, self.wgt_t, self.adj_t)
        loss_global = loss_g_s + loss_g_t

        loss_match = self.match_loss(embed_s, embed_t, common, anchor_link, self.wgt_l, self.adj_l, self.adj_t)

        # sum all loss
        loss_batch = 0.2 * loss_global + loss_match

        return loss_batch

    @staticmethod
    def normalize_output(out_feat, idx):
        sum_m = 0
        for m in out_feat:
            sum_m += torch.mean(torch.norm(m[idx], dim=1))
        sum_m /= len(out_feat)
        return sum_m

    def cal_loss(self, anchor_link, mapping, *data):
        (embed_super_s, mis_s_s, embed_super_t, mis_s_t,
         embed_norm_s, mis_n_s, rdd_n_s, embed_norm_t, mis_n_t, rdd_n_t,
         embed_tail_s, rdd_t_s, embed_tail_t, rdd_t_t) = data
        # loss
        L_uil_super = self.train(embed_super_s, embed_super_t, anchor_link, mapping)
        L_uil_h = self.train(embed_norm_s, embed_norm_t, anchor_link, mapping)
        L_uil_t = self.train(embed_tail_s, embed_tail_t, anchor_link, mapping)
        L_uil = L_uil_super + L_uil_h + L_uil_t

        m_sup_s = self.normalize_output(mis_s_s, self.args.idx_super_s)
        m_sup_t = self.normalize_output(mis_s_t, self.args.idx_super_t)
        m_sup = m_sup_s + m_sup_t

        m_norm_s = self.normalize_output(mis_n_s + rdd_n_s, self.args.idx_norm_s)  # Loss for potential information constraint
        m_norm_t = self.normalize_output(mis_n_t + rdd_n_t, self.args.idx_norm_t)
        m_norm = m_norm_s + m_norm_t

        m_tail_s = self.normalize_output(rdd_t_s, self.args.idx_tail_s)
        m_tail_t = self.normalize_output(rdd_t_t, self.args.idx_tail_t)
        m_tail = m_tail_s + m_tail_t
        m = (m_sup + m_norm + m_tail) / 3

        L_all = L_uil + self.args.mu * m

        return L_all

    # ======= evaluate ==========
    def eval_hit(self, epoch, embed_s, embed_t, maml_mapping):
        maml_mapping.eval()
        emb_s_map, emb_t_map = maml_mapping.module[0](embed_s), maml_mapping.module[1](embed_t)
        self.log.info(f'Epoch: {epoch}')

        mrr, hit_p, success_p = self.eval_metrics_na(emb_s_map, emb_t_map, self.k)
        if mrr > self.mrr_best:
            self.mrr_best = mrr
            self.hit_best = hit_p
            self.success_best = success_p
            # if epoch > 0:  # save best model, saving time
            write_pickle([emb_s_map.detach().cpu(), emb_t_map.detach().cpu()], self.save_best_embeds_path)
                # self.save_best_embeds = [emb_s_map.detach().cpu(), emb_t_map.detach().cpu()]
        # self.log.info('Epoch: {}, MRR_best: {:.4f}, Hit_best: {:.4f}'.format(epoch, self.mrr_best, self.hit_best))
        self.log.info(
            '== Common best == MRR: {:.4f}, Hit: {:.4f}, Success: {:.4f}'.format(self.mrr_best, self.hit_best, self.success_best))

        # if self.args.adapt and epoch > 0 and mrr > self.mrr_best-0.005:
        # begin = 0
        # if cfg.dataset == 'DBLP':
        #     begin = 0
        if self.args.adapt and epoch > self.args.epochs - 10 and mrr > self.mrr_best-0.005:
        # if self.args.adapt:
            test_batches = self.meta_test.get_test_batches(embed_s.detach().cpu().numpy(), sim_metric=self.args.sim_metric)
            # adaptive testing
            mrr, hit_p, success_p, best_adapt_map_emb = self.hit_test_adapt(test_batches, embed_s, embed_t,
                                                                      maml_mapping, adaptation_steps=1)
            if mrr > self.mrr_best_adapt:
                self.mrr_best_adapt = mrr
                self.hit_best_adapt = hit_p
                self.success_best_adapt = success_p
                self.best_test_batches = test_batches
                self.best_adapt_map_emb = best_adapt_map_emb
                # self.best_adapt_map_emb = [emb.detach().cpu() for emb in best_adapt_map_emb]
                best_adapt_map_emb = [emb.detach().cpu().numpy() for emb in self.best_adapt_map_emb]
                write_pickle(best_adapt_map_emb, self.save_adapt_best_embeds_path)
                # torch.save(maml_mapping.state_dict(), self.save_model_path)

            self.log.info('== Adapt best == MRR: {:.4f}, Hit: {:.4f}, Success: {:.4f} ==='
                          .format(self.mrr_best_adapt, self.hit_best_adapt, self.success_best_adapt))

    def print_performance_k(self, time_train):
        self.log.info('--- Final Results ---')
        self.log.info('Time of training: {:.2f}s'.format(time_train))
        if self.args.adapt:
            # maml_mapping.load_state_dict(torch.load(self.save_model_path))
            # maml_mapping.eval()
            best_adapt_map_emb = read_pickle(self.save_adapt_best_embeds_path)
            embed_s, embed_t = [torch.from_numpy(emb) for emb in best_adapt_map_emb]
            self.report_hit(embed_s, embed_t)
            # self.rename_log('/mrr{:.2f}_{}_'.format(self.mrr_best_adapt * 100, self.args.adapt).join(
            #     cfg.log.split('/')
            # ))
            self.rename_log('logs/mrr{:.2f}_{}_'.format(self.mrr_best_adapt * 100, self.args.adapt).join(
                cfg.log.split('logs/')
            ))
        else:
            # emb_s, emb_t = self.save_best_embeds
            emb_s, emb_t = read_pickle(self.save_best_embeds_path)
            self.report_hit(emb_s, emb_t)
            # self.rename_log('/mrr{:.2f}_{}_'.format(self.mrr_best * 100, self.args.adapt).join(
            #     cfg.log.split('/')
            self.rename_log('logs/mrr{:.2f}_{}_'.format(self.mrr_best * 100, self.args.adapt).join(
                cfg.log.split('logs/')
            ))

    def hit_test_adapt(self, test_batches, emb_s, emb_t, base_mdoel, adaptation_steps=1):
        cnt = 0
        test_tasks = test_batches[0]

        emb_s, emb_t = emb_s.detach(), emb_t.detach()
        emb_s_map, emb_t_map = base_mdoel.module[0](emb_s), base_mdoel.module[1](emb_t)

        for i in trange(len(test_tasks)):
            task = test_tasks[i]
            learner = base_mdoel.clone()
            support_link, query_link = task

            support_link = list(filter(None, support_link))
            if len(support_link) > 0:
                for step in range(adaptation_steps):
                    train_loss = self.train(emb_s, emb_t, support_link, learner.module)
                    # learner.adapt(train_loss)
                    grads_theta = autograd.grad(train_loss,
                                                learner.module[0].parameters(),
                                                retain_graph=False, create_graph=False, allow_unused=True)
                    learner.module[0] = maml_update(learner.module[0], lr=self.args.fast_lr, grads=grads_theta)

            # Evaluate the adapted model
            with torch.no_grad():
                source_after_mapping = learner.module[0](emb_s)
                # target_after_mapping = learner.module[1](emb_t)
            # test_losses.append(test_loss.item())
            test_num = len(query_link)
            cnt += test_num

            test_sid, _ = [list(i) for i in list(zip(*query_link))]
            emb_s_map[test_sid] = source_after_mapping[test_sid]
            # emb_t_map[test_tid] = target_after_mapping[test_tid]

        mrr, hit_p, success_p = self.eval_metrics_na(emb_s_map, emb_t_map, self.k)
        # test_loss = np.mean(test_losses)

        # test_loss, mrr, hit_p, success_p = np.mean(test_losses), np.sum(mrr_list) / cnt, np.sum(hit_list) / cnt, np.sum(success_list) / cnt
        self.log.info('-- MRR: {:.4f}, Hit: {:.4f}, Success: {:.4f} --'.format(mrr, hit_p, success_p))

        return mrr, hit_p, success_p, (emb_s_map, emb_t_map)


class MLP(nn.Module):
    def __init__(self, dim_in, dim_out,
                 dim_hid=None, act=None):
        super(MLP, self).__init__()
        if act is None:
            act = nn.Tanh()
        if dim_hid is None:
            dim_hid = dim_in * 2
        # 2-layers
        self.model = nn.Sequential(
            nn.Linear(dim_in, dim_hid, bias=True),
            act,
            nn.Linear(dim_hid, dim_out, bias=True)
        )

    def forward(self, x):
        return self.model(x)

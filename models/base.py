import logging

import os
from torch.utils.data import Dataset, DataLoader
import torch
from torch import Tensor
from utils.extensions import cosine
import config as cfg
from utils.log import create_logger
import numpy as np


class Instructor:
    def __init__(self):
        self.log = create_logger(
            __name__, silent=False,
            to_disk=True, log_file=cfg.log)

    def rename_log(self, filename):
        logging.shutdown()
        os.rename(cfg.log, filename)

    @staticmethod
    def optimize(opt, loss):
        opt.zero_grad()
        loss.backward()
        opt.step()

    @staticmethod
    def load_data(input, batch_size):
        data = DataWrapper(input)
        batches = DataLoader(
            data,
            batch_size=batch_size,
            shuffle=True)
        return batches

    @staticmethod
    def early_stop(current, results,
                   size=3, epsilon=5e-5):
        results[:-1] = results[1:]
        results[-1] = current
        assert len(results) == 2 * size
        pre = results[:size].mean()
        post = results[size:].mean()
        return abs(pre - post) > epsilon


class DataWrapper(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class UIL(Instructor):
    """
    A base class for training a UIL model
    """

    def __init__(self, links, k):
        """
        Args:
            idx: ground truth user pairs for training and testing
            k: number of candidates
        """
        super(UIL, self).__init__()
        self.links = links
        self.k = k  # k candidates

    def get_embeds(self, is_map):
        raise NotImplementedError

    def eval_hit_p(self, sim_mat, k, mask=None, default=-1.):
        """
        Evaluation precision@k and hit-precision@k in the training set and testing set.
        Args:
            mask: a matrix masking known matched user pairs
            default: default similarity for matched user pairs in the training set
        Return:
            hit-precision@k
        """
        with torch.no_grad():
            # similarity_mat = self.sim_pairwise(embed_s, embed_t)
            similarity_mat = sim_mat.clone()
            if mask is not None:
                similarity_mat *= mask
            train_link, test_link = self.links
            mrr, hit_p, suc_p = self.get_metrics(similarity_mat,
                                               k, train_link)
            self.log.info('Train MRR {:.4f} | Hit@{} {:.4f} | Success@{} {:.4f}'.format(
                mrr, k, hit_p, k, suc_p
            ))

            row, col = [list(i) for i in zip(*train_link)]
            # delete nodes and connected link in train set
            # mask similarities of matched user pairs in the train set
            similarity_mat[row] = default
            similarity_mat[:, col] = default
            mrr, hit_p, suc_p = self.get_metrics(similarity_mat,
                                               k, test_link)
            self.log.info('Test MRR {:.4f} | Hit@{} {:.4f} | Success@{} {:.4f}'.format(
                mrr, k, hit_p, k, suc_p
            ))
            return mrr, hit_p, suc_p

    def sim_pairwise(self, xs, ys):
        return cosine(xs, ys)

    def get_metrics(self, sims_mat, k, link):
        """
        Calculate the average precision@k and hit_precision@k from two sides, i.e., source-to-target and target-to-source.
        Args:
          sims_mat: similarity matrix
          k: number of candidates
          link: index pairs of matched users, i.e., the ground truth
        Return:
          coverage: precision@k
          hit: hit_precison@k
        """
        # row: source node  col: target node
        row, col = [list(i) for i in zip(*link)]
        target = sims_mat[row, col].reshape((-1, 1))  # s,t两用户节点真实link的相似度
        s_sim = sims_mat[row]
        t_sim = sims_mat.t()[col]
        # match users from source to target
        mrr_s, h_s, s_s = self.score(s_sim, target, k)
        # match users from target to source
        mrr_t, h_t, s_t = self.score(t_sim, target, k)
        # averaging the scores from both sides
        return (mrr_s + mrr_t) / 2, (h_s + h_t) / 2, (s_s + s_t) / 2
        # return mrr_s, h_s

    @staticmethod
    def score(sims: Tensor, target: Tensor, k: int) -> tuple:
        """
        Calculate the average precision@k and hit_precision@k from while matching users from the source network to the target network.
        Args:
            sims: similarity matrix
            k: number of candidates
            test: index pairs of matched users, i.e., the ground truth
        Return:
            coverage: precision@k
            hit: hit_precison@k
        """
        # h(x) = (k - (hit(x) - 1) / k  review 4.2节的公式
        # number of users with similarities larger than the matched users
        device = sims.device
        rank = (sims >= target).sum(1)
        mrr = (1. / rank).mean()
        success_k = (rank <= k).sum() / rank.shape[0]
        # rank = min(rank, k + 1)
        rank = rank.min(torch.tensor(k + 1).to(device))
        temp = (k + 1 - rank).float()
        # hit_precision@k
        hit_p = (temp / k).mean()
        # precision@k
        # coverage = (temp > 0).float().mean()
        # print('Precision_k: {:.4f}, Coverage: {:.4f}'.format(precision_k, coverage))
        return mrr, hit_p, success_k

    @staticmethod
    def csls_sim(sim_mat, csls_k):
        """
        https://github.com/cambridgeltl/eva/blob/948ffbf3bf70cd3db1d3b8fcefc1660421d87a1f/src/utils.py#L287
        Compute pairwise csls similarity based on the input similarity matrix.
        Parameters
        ----------
        sim_mat : matrix-like
            A pairwise similarity matrix.
        csls_k : int
            The number of nearest neighbors.
        Returns
        -------
        csls_sim_mat : A csls similarity matrix of n1*n2.
        """

        nearest_values1 = torch.mean(torch.topk(sim_mat, csls_k)[0], 1)
        nearest_values2 = torch.mean(torch.topk(sim_mat.t(), csls_k)[0], 1)
        sim_mat = 2 * sim_mat.t() - nearest_values1
        sim_mat = sim_mat.t() - nearest_values2
        return sim_mat

    @staticmethod
    def recip_mul(string_mat, k=5, t=None):
        if isinstance(string_mat, torch.Tensor):
            string_mat = string_mat.detach().cpu().numpy()
        sorted_mat = -np.partition(-string_mat, k + 1, axis=0)
        max_values = np.mean(sorted_mat[0:k, :], axis=0)
        a = string_mat - max_values + 1
        sorted_mat = -np.partition(-string_mat, k + 1, axis=1)
        max_values = np.mean(sorted_mat[:, 0:k], axis=1)
        b = (string_mat.T - max_values) + 1
        del string_mat
        del max_values
        from scipy.stats import rankdata
        a_rank = rankdata(-a, axis=1)
        del a
        b_rank = rankdata(-b, axis=1)
        del b
        recip_sim = (a_rank + b_rank.T) / 2.0
        del a_rank
        del b_rank
        # recip_sim = (a+b.T)/2.0
        return recip_sim

    @staticmethod
    def matrix_sinkhorn(pred_or_m, iter=100):
        def view3(x):
            if x.dim() == 3:
                return x
            return x.view(1, x.size(0), -1)

        def view2(x):
            if x.dim() == 2:
                return x
            return x.view(-1, x.size(-1))

        def sinkhorn(a: torch.Tensor, b: torch.Tensor, M: torch.Tensor, eps: float,
                     max_iters: int = 100, stop_thresh: float = 1e-3):
            """
            Compute the Sinkhorn divergence between two sum of dirac delta distributions, U, and V.
            This implementation is numerically stable with float32.
            :param a: A m-sized minibatch of weights for each dirac in the first distribution, U. i.e. shape = [m, n]
            :param b: A m-sized minibatch of weights for each dirac in the second distribution, V. i.e. shape = [m, n]
            :param M: A minibatch of n-by-n tensors storing the distance between each pair of diracs in U and V.
                     i.e. shape = [m, n, n] and each i.e. M[k, i, j] = ||u[k,_i] - v[k, j]||
            :param eps: The reciprocal of the sinkhorn regularization parameter
            :param max_iters: The maximum number of Sinkhorn iterations
            :param stop_thresh: Stop if the change in iterates is below this value
            :return:
            """
            # a and b are tensors of size [m, n]
            # M is a tensor of size [m, n, n]

            nb = M.shape[0]
            m = M.shape[1]
            n = M.shape[2]

            if a.dtype != b.dtype or a.dtype != M.dtype:
                raise ValueError(
                    "Tensors a, b, and M must have the same dtype got: dtype(a) = %s, dtype(b) = %s, dtype(M) = %s"
                    % (str(a.dtype), str(b.dtype), str(M.dtype)))
            if a.device != b.device or a.device != M.device:
                raise ValueError("Tensors a, b, and M must be on the same device got: "
                                 "device(a) = %s, device(b) = %s, device(M) = %s"
                                 % (a.device, b.device, M.device))
            if len(M.shape) != 3:
                raise ValueError("Got unexpected shape for M (%s), should be [nb, m, n] where nb is batch size, and "
                                 "m and n are the number of samples in the two input measures." % str(M.shape))
            if torch.Size(a.shape) != torch.Size([nb, m]):
                raise ValueError(
                    "Got unexpected shape for tensor a (%s). Expected [nb, m] where M has shape [nb, m, n]." %
                    str(a.shape))

            if torch.Size(b.shape) != torch.Size([nb, n]):
                raise ValueError(
                    "Got unexpected shape for tensor b (%s). Expected [nb, n] where M has shape [nb, m, n]." %
                    str(b.shape))

            # Initialize the iteration with the change of variable
            u = torch.zeros(a.shape, dtype=a.dtype, device=a.device)
            v = eps * torch.log(b)

            M_t = torch.transpose(M, 1, 2)

            def stabilized_log_sum_exp(x):
                max_x = torch.max(x, dim=2)[0]
                x = x - max_x.unsqueeze(2)
                ret = torch.log(torch.sum(torch.exp(x), dim=2)) + max_x
                return ret

            for current_iter in range(max_iters):
                u_prev = u
                v_prev = v

                summand_u = (-M + v.unsqueeze(1)) / eps
                u = eps * (torch.log(a) - stabilized_log_sum_exp(summand_u))

                summand_v = (-M_t + u.unsqueeze(1)) / eps
                v = eps * (torch.log(b) - stabilized_log_sum_exp(summand_v))

                err_u = torch.max(torch.sum(torch.abs(u_prev - u), dim=1))
                err_v = torch.max(torch.sum(torch.abs(v_prev - v), dim=1))

                if err_u < stop_thresh and err_v < stop_thresh:
                    break

            log_P = (-M + u.unsqueeze(2) + v.unsqueeze(1)) / eps

            P = torch.exp(log_P)

            return P

        # from fml.functional import sinkhorn
        device = pred_or_m.device
        M = view3(pred_or_m).to(torch.float32)
        m, n = tuple(pred_or_m.size())
        a = torch.ones([1, m], requires_grad=False, device=device)
        b = torch.ones([1, n], requires_grad=False, device=device)
        P = sinkhorn(a, b, M, 1e-3, max_iters=iter, stop_thresh=1e-3)  # max_iters=300
        return view2(P)

    def report_hit_multi(self, emb_s, emb_t):
        from utils.extensions import cosine
        with torch.no_grad():
            cos_sim_mat = cosine(emb_s, emb_t)
            csls_sim_mat = self.csls_sim(cos_sim_mat, csls_k=5)
            rinf_sim_mat = 1 - self.recip_mul(cos_sim_mat, k=5)
            sinkhorn_sim_mat = self.matrix_sinkhorn(1 - cos_sim_mat, iter=100)

            self.log.info('\n ----- Cosine Sim -------')
            # for k in [1] + [10 * i for i in range(1, 6)]:
            for k in [1, 10, 30]:
                self.eval_hit_p(cos_sim_mat, k)

            self.log.info('\n ----- CSLS + Cosine Sim -------')
            # for k in [1] + [10 * i for i in range(1, 6)]:
            for k in [1, 10, 30]:
                self.eval_hit_p(csls_sim_mat, k)

            self.log.info('\n ----- RInf + Cosine Sim -------')
            # for k in [1] + [10 * i for i in range(1, 6)]:
            for k in [1, 10, 30]:
                self.eval_hit_p(torch.tensor(rinf_sim_mat), k)

            self.log.info('\n ----- Sinkhorn + Cosine Sim -------')
            # for k in [1] + [10 * i for i in range(1, 6)]:
            for k in [1, 10, 30]:
                self.eval_hit_p(sinkhorn_sim_mat, k)

    def report_hit(self, emb_s, emb_t):
        with torch.no_grad():
            # for k in [1] + [10 * i for i in range(1, 6)]:
            for k in [1, 10, 30]:
                self.eval_metrics_na(emb_s, emb_t, k)

    def eval_metrics_na(self, emb_s, emb_t, k):
        cos_sim_mat = cosine(emb_s, emb_t)
        mrr, hit_p, success_p = self.eval_hit_p(cos_sim_mat, k)
        return mrr, hit_p, success_p

    def calculate_hit(self, emb_s_map, emb_t_map, train_link, test_link, top_k):
        with torch.no_grad():
            similarity_mat = self.sim_pairwise(emb_s_map, emb_t_map)
            row, col = [list(i) for i in zip(*train_link)]  # train nodes
            mask_default = -1.  # cosine similarity minimum
            similarity_mat[row] = mask_default
            similarity_mat[:, col] = mask_default

            coverage, hit_p, success_p = self.get_metrics(similarity_mat,
                                               top_k, test_link)
            return coverage, hit_p, success_p

    def getPairs(self, link, adj_s, adj_t):
        '''得到link对在两个网络中的邻接节点'''
        s_idx, t_idx = link
        s_adj, t_adj = adj_s.copy(), adj_t.copy()
        s_n, t_n = set(range(adj_s.shape[0])), \
                   set(range(adj_t.shape[0]))
        # 选出不属于link对的节点
        # s_i, t_i = list(s_n - set(s_idx.numpy())), \
        #            list(t_n - set(t_idx.numpy()))
        s_i, t_i = list(s_n - set(s_idx)), \
                   list(t_n - set(t_idx))
        s_adj[s_i, :] = 0
        s_adj[:, s_i] = 0
        t_adj[t_i, :] = 0
        t_adj[:, t_i] = 0

        s_pair = list(zip(*[i.astype(np.int64) for i in s_adj.nonzero()]))
        t_pair = list(zip(*[i.astype(np.int64) for i in t_adj.nonzero()]))
        # s_pair = [torch.from_numpy(i) for i in s_adj.nonzero()]
        # t_pair = [torch.from_numpy(i) for i in t_adj.nonzero()]
        return s_pair, t_pair


class UILDeg(Instructor):
    """
    A base class for training a UIL model
    """
    def __init__(self, links, k, adj_s):
        """
        Args:
            idx: ground truth user pairs for training and testing
            k: number of candidates
        """
        super(UILDeg, self).__init__()
        # self.links = links
        self.train_links, self.test_links = links
        self.k = k  # k candidates

        # divide anchor nodes into classes according to degree
        # source, target = [list(i) for i in self.test_links]
        # test_source_deg = adj_s[source].sum(1).astype(int)  # sum each row
        link_deg_dict = {}
        self.d_test_link_deg = {}
        for anchor in self.test_links:
            sn = anchor[0]
            deg = adj_s[sn].sum(1).astype(int).item()  # source node degree
            link_deg_dict.setdefault(deg, []).append(anchor)
        self.d_test_link_deg = dict(sorted(link_deg_dict.items(), key=lambda data: data[0], reverse=True))  # high to low
        # self.d_test_link_deg = dict(sorted(link_deg_dict.items(), key=lambda data: data[0]))
        # self.d_test_link_deg = dict(link_deg_dict.items())

    def get_embeds(self, is_map):
        raise NotImplementedError

    def sim_pairwise(self, xs, ys):
        return cosine(xs, ys)

    def get_metrics(self, sims_mat, k, link):
        """
        Calculate the average precision@k and hit_precision@k from two sides, i.e., source-to-target and target-to-source.
        Args:
          sims_mat: similarity matrix
          k: number of candidates
          link: index pairs of matched users, i.e., the ground truth
        Return:
          coverage: precision@k
          hit: hit_precison@k
        """
        # row: source node  col: target node
        row, col = [list(i) for i in zip(*link)]
        target = sims_mat[row, col].reshape((-1, 1))  # s,t两用户节点真实link的相似度
        s_sim = sims_mat[row]
        t_sim = sims_mat.t()[col]
        # match users from source to target
        c_s, h_s = self.score(s_sim, target, k)
        # match users from target to source
        c_t, h_t = self.score(t_sim, target, k)
        # averaging the scores from both sides
        # return (c_s + c_t) / 2, (h_s + h_t) / 2
        return c_s, h_s

    @staticmethod
    def score(sims: Tensor, target: Tensor, k: int) -> tuple:
        """
        Calculate the average precision@k and hit_precision@k from while matching users from the source network to the target network.
        Args:
            sims: similarity matrix
            k: number of candidates
            test: index pairs of matched users, i.e., the ground truth
        Return:
            coverage: precision@k
            hit: hit_precison@k
        """
        # h(x) = (k - (hit(x) - 1) / k  review 4.2节的公式
        # number of users with similarities larger than the matched users
        rank = (sims >= target).sum(1)
        # rank = min(rank, k + 1)
        rank = rank.min(torch.tensor(k + 1).to(cfg.device))
        temp = (k + 1 - rank).float()
        # hit_precision@k
        hit_p = (temp / k)
        # precision@k
        coverage = (temp > 0).float()
        return coverage, hit_p

    def report_hit(self, sims_orig, mask=None, default=0.):
        with torch.no_grad():
            train, test = self.links
            for k in [10 * i for i in range(1, 6)]:
                sims = sims_orig.clone()
                coverage, hit_p = self.get_metrics(sims, k, train)
                self.log.info('Train Coverage@{} {:.4f} | Hit@{} {:.4f}'.format(
                    k, coverage, k, hit_p
                ))
                row, col = [list(i) for i in zip(*train)]
                sims[row] = default
                sims[:, col] = default
                coverage, hit_p = self.get_metrics(sims, k, test)
                self.log.info('Test Coverage@{} {:.4f} | Hit@{} {:.4f}'.format(
                    k, coverage, k, hit_p
                ))

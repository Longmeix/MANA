import numpy as np

# k = 10


# def eval_uil_hit(f_s, f_t, train, test, k, mask=None, default=-1.):
#     sims = sim_cosine(f_s, f_t)
#     if mask is not None:
#         sims = sims * mask
#     # train, test = self.idx
#     mrr, hit_p = bi_hit_x(sims, k, train)
#     print('Train MRR {:.4f} | Hit@{} {:.4f}'.format(
#         mrr, k, hit_p
#     ))
#     n1 = f_s.shape[0]
#     row, col = [list(i) for i in zip(*train)]
#     col = [i-n1 for i in col]
#     sims[row] = default
#     sims[:, col] = default
#     mrr, hit_p = bi_hit_x(sims, k, test)
#     print('Test MRR {:.4f} | Hit@{} {:.4f}'.format(
#         mrr, k, hit_p
#     ))
#
#     return mrr, hit_p
#
#
# def sim_cosine(xs, ys, epsilon=1e-8):
#     mat = xs @ ys.T
#     # Plus a small epsilon to avoid dividing by zero.
#     x_norm = np.linalg.norm(xs, axis=1) + epsilon
#     y_norm = np.linalg.norm(ys, axis=1) + epsilon
#     x_diag = np.diag(1 / x_norm)
#     y_diag = np.diag(1 / y_norm)
#     return x_diag @ mat @ y_diag
#
#
# def bi_hit_x(sims, k, test):
#     row, col = [list(i) for i in zip(*test)]
#     col = [i-sims.shape[0] for i in col]
#     target = sims[row, col].reshape((-1, 1))
#     left = sims[row]
#     right = sims.T[col]
#     c_l, h_l = score(left, target, k)
#     c_r, h_r = score(right, target, k)
#     return (c_l + c_r) / 2, (h_l + h_r) / 2
#     # return c_l, h_l
#
#
# def score(mat, target, k):
#     rank = (mat >= target).sum(1)
#     mrr = (1. / rank).mean()
#     # rank = rank.min(torch.tensor(k + 1).to(cfg.device))
#     rank = np.minimum(rank, k+1)
#     tmp = (k + 1 - rank).astype(float)
#     hit_score = (tmp / k).mean()
#     # coverage = np.mean((tmp > 0).astype(float))
#     return mrr, hit_score

import numpy as np
import torch


# k = 10
def eval_na_hit(embed_s, embed_t, train_link, test_link, k, mask=None, default=-1.):
    """
    Evaluation precision@k and hit-precision@k in the training set and testing set.
    """
    def rename_idx(link, num1):
        row, col = [list(i) for i in zip(*link)]
        col = [i - num1 for i in col]
        return row, col

    with torch.no_grad():
        similarity_mat = sim_cosine(embed_s, embed_t)
        if mask is not None:
            similarity_mat *= mask

        n1 = embed_s.shape[0]
        row, col = rename_idx(train_link, n1)

        mrr, hit_p, suc_p = get_metrics(similarity_mat, k, (row, col))
        print('Train MRR {:.4f} | Hit@{} {:.4f} | Success@{} {:.4f}'.format(
                mrr, k, hit_p, k, suc_p
            ))

        default = similarity_mat.min() - 1

        # delete nodes and connected link in train set
        # mask similarities of matched user pairs in the train set
        similarity_mat[row] = default
        similarity_mat[:, col] = default
        mrr, hit_p, suc_p = get_metrics(similarity_mat, k, rename_idx(test_link, n1))
        print('Test MRR {:.4f} | Hit@{} {:.4f} | Success@{} {:.4f}'.format(
                mrr, k, hit_p, k, suc_p
            ))
        return mrr, hit_p, suc_p


def get_metrics(sims_mat, k, link):
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
    # row, col = link
    # col = [i - sims_mat.shape[0] for i in col]
    target = sims_mat[row, col].reshape((-1, 1))  # s,t两用户节点真实link的相似度
    s_sim = sims_mat[row]
    t_sim = sims_mat.T[col]
    # match users from source to target
    mrr_s, h_s, s_s = score(s_sim, target, k)
    # match users from target to source
    mrr_t, h_t, s_t = score(t_sim, target, k)
    # averaging the scores from both sides
    return (mrr_s + mrr_t) / 2, (h_s + h_t) / 2, (s_s + s_t) / 2


def score(sims, target, k: int) -> tuple:
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
    # device = sims.device
    rank = (sims >= target).sum(1)
    mrr = (1. / rank).mean()
    success_k = (rank <= k).sum() / rank.shape[0]
    rank = np.minimum(rank, k + 1)
    # rank = rank.min(torch.tensor(k + 1))
    temp = (k + 1 - rank).astype(np.float64)
    # hit_precision@k
    hit_p = (temp / k).mean()
    # precision@k
    return mrr, hit_p, success_k


def sim_cosine(xs, ys, epsilon=1e-8):
    mat = xs @ ys.T
    # Plus a small epsilon to avoid dividing by zero.
    x_norm = np.linalg.norm(xs, axis=1) + epsilon
    y_norm = np.linalg.norm(ys, axis=1) + epsilon
    x_diag = np.diag(1 / x_norm)
    y_diag = np.diag(1 / y_norm)
    return x_diag @ mat @ y_diag



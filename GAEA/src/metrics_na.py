import numpy as np
import torch


# k = 10
def eval_hit_p(embed_s, embed_t, train_link, test_link, k, mask=None, default=-1.):
    """
    Evaluation precision@k and hit-precision@k in the training set and testing set.
    """
    with torch.no_grad():
        similarity_mat = sim_cosine(embed_s, embed_t)
        if mask is not None:
            similarity_mat *= mask
        mrr, hit_p, suc_p = get_metrics(similarity_mat, k, train_link)
        print('Train MRR {:.4f} | Hit@{} {:.4f} | Success@{} {:.4f}'.format(
                mrr, k, hit_p, k, suc_p
            ))

        default = similarity_mat.min() - 1
        row, col = [list(i) for i in zip(*train_link)]
        # delete nodes and connected link in train set
        # mask similarities of matched user pairs in the train set
        similarity_mat[row] = default
        similarity_mat[:, col] = default
        mrr, hit_p, suc_p = get_metrics(similarity_mat, k, test_link)
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
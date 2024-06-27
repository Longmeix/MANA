import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import time
import torch
import numpy as np
from torch.utils.data import DataLoader
import argparse
from prep_meta_tasks.task_generate import MetaTaskRandom, MetaTaskNeighbor, MetaTaskSimilarity
from src.loss import *
from src.dataset import KGData, Dataset
from src.utils import *
from src.baselines import *
from src.model import GAEA
from src.metrics_na import eval_hit_p
import learn2learn as l2l
from torch import autograd
from learn2learn.algorithms import maml_update
from tqdm import trange
from utils.general import write_pickle, read_pickle


np.set_printoptions(suppress=True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"\n- current device is {device}\n")

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--epoch", type=int, default=10, help="epoch to run")
parser.add_argument("--model", type=str, default='gaea', help="the model's name", choices=['gaea', 'gat', 'gcn'])
# parser.add_argument("--task", type=str, default='en_fr_15k', help="the alignment task name", choices=['en_fr_15k', 'en_de_15k', 'd_w_15k', 'd_y_15k', 'en_fr_100k', 'en_de_100k', 'd_w_100k', 'd_y_100k'])
parser.add_argument("--task", type=str, default='DBLP', help="the alignment task name",
                    choices=['FT', 'DBLP', 'd_w_15k'])
parser.add_argument("--fold", type=int, default=1, help="the fold cross-validation")
parser.add_argument("--train_ratio", type=int, default=5, help="training set ratio")
parser.add_argument("--val", action="store_true", default=False, help="need validation?")
parser.add_argument("--il", action="store_true", default=False, help="Iterative learning?")
parser.add_argument("--il_start", type=int, default=200, help="If Il, when to start?")
parser.add_argument("--il_iter", type=int, default=100, help="If IL, what's the update step?")
# parser.add_argument('--lr', type=float, default=0.001, help="learning rate")
parser.add_argument('--lr', type=float, default=0.0005, help="learning rate")
parser.add_argument('--k', type=float, default=10, help="hit@k")
parser.add_argument("--eval_metric", type=str, default="euclidean", help="the distance metric of entity pairs",
                    choices=["cityblock", "euclidean", "csls", "inner"])
parser.add_argument('--neg_samples_size', type=int, default=5, help="number of negative samples")
parser.add_argument('--neg_iter', type=int, default=10, help="re-calculate epoch of negative samples")
parser.add_argument("--neg_metric", type=str, default="inner",
                    choices=["cityblock", "euclidean", "csls", "inner"])  # same as BootEA
parser.add_argument('--weight_decay', type=float, default=1e-5, help="weight decay coefficient")
parser.add_argument("--csls", action="store_true", default=False, help="use CSLS for inference?")
parser.add_argument("--csls_k", type=int, default=10, help="top k for csls")
parser.add_argument('--patience', type=int, default=5, help='patience default 5')
parser.add_argument('--optim', type=str, default='adam', help='the optimizer')
parser.add_argument('--truncated_epsilon', type=float, default=0.9, help='the epsilon of truncated negative sampling')
parser.add_argument('--loss_fn', type=str, default="margin_based", choices=["limit_based", "margin_based"],
                    help="the selected loss function")
parser.add_argument('--loss_norm', type=str, default='l2', help='the distance metric of loss function')
parser.add_argument('--pos_margin', type=float, default=0.01, help="positive margin in limit-based loss function")
parser.add_argument('--neg_margin', type=float, default=1.0, help='negative margin for loss computation')
parser.add_argument('--neg_param', type=float, default=0.2, help="the neg_margin_balance")
parser.add_argument('--save', action="store_true", default=False)
parser.add_argument('--result', action="store_true", default=True)
parser.add_argument('--init_type', default="xavier", choices=["xavier", "normal"], type=str)
parser.add_argument('--direct', action="store_true", default=True, help="use the diretions of relations?")
parser.add_argument('--res', action="store_true", default=False, help="use residual link?")
parser.add_argument('--ent_dim', type=int, default=256, help="hidden dimension of entity embeddings")
parser.add_argument('--rel_dim', type=int, default=128, help="hidden dimension of relation embeddings")
parser.add_argument('--dropout', type=float, default=0.2, help="dropout rate")
parser.add_argument('--layer', type=int, default=2, help="layer number of GNN-based encoder")
parser.add_argument('--n_head', type=int, default=1, help="number of multi-head attention")
parser.add_argument('--pr', type=float, default=0.1, help='the edge drop rate')
# parser.add_argument('--aug_balance', type=float, default=100, help='the hyper-parameter of consistency loss')
parser.add_argument('--aug_balance', type=float, default=10, help='the hyper-parameter of consistency loss')
parser.add_argument('--aug_iter', type=int, default=1)
# for meta learning
parser.add_argument('--adapt', default=True, type=bool)
# parser.add_argument('--adapt', default=False, type=bool)
parser.add_argument('--support', default='similarity', type=str, help='random/neighbor/similarity')
parser.add_argument("--train_batch_size", type=int, default=1, help="train batch_size (-1 means all in)")
parser.add_argument('--fast_lr', default=0.1, type=float)
parser.add_argument('--n_way', default=5, type=int)
parser.add_argument('--k_shot', default=5, type=int)
parser.add_argument('--save_embeds_file', default='embeds_gaea_{}_meta.pkl', help='file type: .pkl', type=str)
parser.add_argument('--save_best_embeds_path', default='gaea_best_embs.pkl', help='file type: .pkl', type=str)

# os.chdir('../')
args = parser.parse_args()
set_random_seed(args.seed)
if args.save:
    save_args(args=args)

save_folder = './dataset/models/GAEA/' + args.model
if not os.path.exists(save_folder):
    os.makedirs(save_folder)
save_best_embeds_path = save_folder + '/{}_{}_best_embs.pkl'.format(args.task, args.train_ratio/10)
save_adapt_best_embeds_path = save_folder + '/{}_{}_best_embs_adapt.pkl'.format(args.task, args.train_ratio/10)
save_model_path = save_folder + '/model.pt'

# STEP: load data
kgdata = KGData(model=args.model, task=args.task, device=device, neg_samples_size=args.neg_samples_size, fold=args.fold,
                train_ratio=args.train_ratio / 10, val=args.val, direct=args.direct)
kgdata.data_summary()
assert args.train_batch_size < kgdata.train_pair_size
meta_bsz = args.train_batch_size
n_way, k_shot = args.n_way, args.k_shot

# STEP: model initialization
print('[model initializing...]\n')
'''model selection'''
if args.model == "gcn":
    model = GCN(num_sr=kgdata.kg1_ent_num, num_tg=kgdata.kg2_ent_num, adj_sr=kgdata.tensor_adj1,
                adj_tg=kgdata.tensor_adj2, embedding_dim=args.ent_dim, dropout=args.dropout, layer=args.layer)
elif args.model == "gat":
    model = GAT(num_sr=kgdata.kg1_ent_num, num_tg=kgdata.kg2_ent_num, adj_sr=kgdata.tensor_adj1,
                adj_tg=kgdata.tensor_adj2, embedding_dim=args.ent_dim, dropout=args.dropout, layer=args.layer)
elif args.model == "gaea":
    model = GAEA(num_sr=kgdata.kg1_ent_num, num_tg=kgdata.kg2_ent_num, adj_sr=kgdata.tensor_adj1,
                 adj_tg=kgdata.tensor_adj2, rel_num=kgdata.rel_num, rel_adj_sr=kgdata.tensor_rel_adj1,
                 rel_adj_tg=kgdata.tensor_rel_adj2, args=args)
model = model.to(device=device)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

meta_lr = args.lr
maml_model = l2l.algorithms.MAML(model, lr=args.fast_lr, first_order=True)
optimizer = torch.optim.Adam(model.parameters(), meta_lr)

'''loss function selection'''
if args.loss_fn == "limit_based":
    loss_fn = limited_based_loss
elif args.loss_fn == "margin_based":
    loss_fn = margin_based_loss

print("---------------model summary---------------\n")
print(f'#Meta batch size: {meta_bsz}\n')
print(f'#n_way: {n_way}, #k_shot: {k_shot}')
print(f'#total params: {pytorch_total_params}\n')
print("#number of parameter: %.2fM \n" % (pytorch_total_params / 1e6))
print(f"model architecture:\n {model}")
print("-------------------------------------------\n")


def evaluate_na(train_pair, test_pair, k, phase="test", model_path=None):
    if phase == "test" and args.val and args.save:
        with open(model_path, "rb") as f:
            model.load_state_dict(torch.load(f))

    model.eval()
    sr_embedding, tg_embedding = model(phase="eval")
    sr_embedding = sr_embedding.detach().cpu().numpy()  # Returns a new Tensor, detached from the current graph. The result will never require gradient. Before transform to numpy need transfer to cpu firstly.
    tg_embedding = tg_embedding.detach().cpu().numpy()

    mrr, hit_p, suc_p = eval_hit_p(sr_embedding, tg_embedding, train_pair, test_pair, k=k)

    return mrr, hit_p, suc_p


def report_hit(emb_s, emb_t):
    if isinstance(emb_s, torch.Tensor):
        emb_s = emb_s.detach().cpu().numpy()
        emb_t = emb_t.detach().cpu().numpy()
    with torch.no_grad():
        # for k in [1] + [10 * i for i in range(1, 6)]:
        for k in [1, 10, 30]:
            eval_hit_p(emb_s, emb_t, kgdata.mapped_train_pair, kgdata.mapped_test_pair, k)


def cal_loss(data, learner):
    # learner = learner.module
    a1_align, a2_align = [list(i) for i in list(zip(*data))]
    a1_align, a2_align = torch.IntTensor(a1_align), torch.IntTensor(a2_align)

    sr_embedding, tg_embedding = learner(phase="norm")
    '''generate negative samples'''
    neg1_left, neg1_right, neg2_left, neg2_right = kgdata.generate_neg_sample_batch(a1_align.numpy(),
                                                                                    a2_align.numpy(),
                                                                                    sr_embedding.detach().cpu().numpy(),
                                                                                    tg_embedding.detach().cpu().numpy(),
                                                                                    neg_samples_size=args.neg_samples_size)

    if args.pr != 0:
        aug_sr_embedding, aug_tg_embedding = learner(aug_adj1, aug_rel_adj1, aug_adj2, aug_rel_adj2,
                                                   phase="augment")
        '''alignment loss'''
        loss = loss_fn(aug_sr_embedding, aug_tg_embedding, a1_align, a2_align, neg1_left, neg1_right,
                       neg2_left, neg2_right, neg_samples_size=args.neg_samples_size,
                       loss_norm=args.loss_norm, pos_margin=args.pos_margin, neg_margin=args.neg_margin,
                       neg_param=args.neg_param)
    else:
        sr_embedding, tg_embedding = learner(phase="norm")
        loss = loss_fn(sr_embedding, tg_embedding, a1_align, a2_align, neg1_left, neg1_right, neg2_left,
                       neg2_right, neg_samples_size=args.neg_samples_size, loss_norm=args.loss_norm,
                       pos_margin=args.pos_margin, neg_margin=args.neg_margin, neg_param=args.neg_param)
    '''contrastive loss'''
    if args.pr != 0 and args.aug_balance != 0:
        aug_loss1 = learner.contrastive_loss(sr_embedding, aug_sr_embedding, kgdata.kg1_ent_num)
        aug_loss2 = learner.contrastive_loss(tg_embedding, aug_tg_embedding, kgdata.kg2_ent_num)
        loss = loss + args.aug_balance * (aug_loss1 + aug_loss2)

    return loss, sr_embedding, tg_embedding


def fast_adapt(task, learner, adaptation_steps=1):
    support_links, query_links = task
    support_links = list(filter(None, support_links))
    fast_lr = 0.1

    # Adapt the model
    if args.adapt:
        if len(support_links) > 0:
            for step in range(adaptation_steps):
                train_loss, _, _ = cal_loss(support_links, learner.module)
                # learner.adapt(train_loss)

                grads_theta = autograd.grad(train_loss,
                                            learner.module.multihead_attention.parameters(),
                                            # learner.module.proj_head.parameters(),
                                            # learner.module.entity_embedding.parameters(),
                                            retain_graph=False, create_graph=False, allow_unused=True)
                learner.module.multihead_attention = maml_update(learner.module.multihead_attention, lr=fast_lr, grads=grads_theta)
                # learner.module.entity_embedding = maml_update(learner.module.entity_embedding, lr=fast_lr, grads=grads_theta)

    # Evaluate the adapted model
    valid_loss, _, _ = cal_loss(query_links, learner.module)

    return valid_loss


def hit_test_adapt(test_batches, base_model, fast_lr, adaptation_steps=1):
    test_losses = []
    cnt = 0
    test_tasks = test_batches[0]
    emb_s, emb_t = base_model.module(phase='eval')
    fast_lr = 0.1

    for i in trange(len(test_tasks)):
        task = test_tasks[i]
        learner = base_model.clone()
        support_links, query_link = task
        support_links = list(filter(None, support_links))

        if len(support_links) > 0:
            for step in range(adaptation_steps):
                train_loss, _, _ = cal_loss(support_links, learner.module)
                # learner.adapt(train_loss)
                grads_theta = autograd.grad(train_loss,
                                            learner.module.multihead_attention.parameters(),
                                            # learner.module.entity_embedding.parameters(),
                                            retain_graph=False, create_graph=False, allow_unused=True)
                learner.module.multihead_attention = maml_update(learner.module.multihead_attention, lr=fast_lr,
                                                                 grads=grads_theta)
                # learner.module.entity_embedding = maml_update(learner.module.entity_embedding, lr=fast_lr, grads=grads_theta)

        # Evaluate the adapted model
        with torch.no_grad():
            # test_loss, emb_s_adapt, emb_t_adapt = cal_loss(query_link, learner.module)
            emb_s_adapt, emb_t_adapt = learner.module(phase="norm")

        # test_losses.append(test_loss.item())
        test_num = len(query_link)
        cnt += test_num

        # refine global embeddings with local mapping
        test_sid, test_tid = [list(i) for i in list(zip(*query_link))]
        emb_s[test_sid] = emb_s_adapt[test_sid]
        # emb_t[test_tid] = emb_t_adapt[test_tid]

    mrr, hit_p, success_p = eval_hit_p(emb_s.detach().cpu().numpy(), emb_t.detach().cpu().numpy(),
                                       kgdata.mapped_train_pair, kgdata.mapped_test_pair, args.k)
    # test_loss = np.mean(test_losses)

    # print('-- Test loss: {:.4f}, MRR: {:.4f}, Hit: {:.4f}, Success:{:.4f}  --'.format(
    #     test_loss, mrr, hit_p, success_p
    # ))
    print('-- MRR: {:.4f}, Hit: {:.4f}, Success:{:.4f}  --'.format(
        mrr, hit_p, success_p
    ))

    return mrr, hit_p, success_p, (emb_s, emb_t)


# STEP: begin training
try:
    print("[start training...]\n")
    best_val = 0.0
    bad_count = 0
    t_start = time.time()
    t_total_start = time.time()
    model_path = f"save/{args.model}_{round(t_total_start, 3)}"
    sr_embedding, tg_embedding = None, None

    '''generate augmented KG graphs'''
    if args.pr != 0:
        pr1, pr2 = random.uniform(0, args.pr), random.uniform(0, args.pr)
        aug_adj1, aug_rel_adj1 = kgdata.generate_aug_graph(kgdata.triples1, kgdata.kg1_ent_num, kgdata.rel_num,
                                                           kgdata.kg1_ent_ids, kgdata.ent2node1, kgdata.d_v1, pr=pr1)
        aug_adj2, aug_rel_adj2 = kgdata.generate_aug_graph(kgdata.triples2, kgdata.kg2_ent_num, kgdata.rel_num,
                                                           kgdata.kg2_ent_ids, kgdata.ent2node2, kgdata.d_v2, pr=pr2)
    # strategy of tasks
    if args.support == 'random':
        meta_train = MetaTaskRandom(kgdata.mapped_train_pair, kgdata.mapped_train_pair, n_way=n_way, k_shot=k_shot,
                                    q_query=1)
        meta_test = MetaTaskRandom(kgdata.mapped_test_pair, kgdata.mapped_train_pair, n_way=1, k_shot=k_shot, q_query=1)
        test_batches = meta_test.get_test_batches()
    elif args.support == 'neighbor':
        meta_train = MetaTaskNeighbor(kgdata.adj1, kgdata.mapped_train_pair, kgdata.mapped_train_pair, n_way=n_way,
                                      k_shot=k_shot, q_query=1)
        meta_test = MetaTaskNeighbor(kgdata.adj2, kgdata.mapped_test_pair, kgdata.mapped_train_pair, n_way=1,
                                     k_shot=k_shot, q_query=1)
        test_batches = meta_test.get_test_batches()
    elif args.support == 'similarity':
        meta_train = MetaTaskSimilarity(kgdata.mapped_train_pair, kgdata.mapped_train_pair, n_way=n_way, k_shot=k_shot,
                                        q_query=1)
        meta_test = MetaTaskSimilarity(kgdata.mapped_test_pair, kgdata.mapped_train_pair, n_way=1, k_shot=k_shot,
                                       q_query=1)
    else:
        print('Support Strategy Error')

    hit_best, hit_best_adapt = 0., 0.
    mrr_best, mrr_best_adapt = 0., 0.
    success_best, success_best_adapt = .0, .0
    for e in range(args.epoch):
        maml_model.train()
        loss_epc = .0
        # generate meta batches
        with torch.no_grad():
            if args.support == 'similarity':
                # maml_model.eval()
                sr_embedding, _ = model(phase="norm")
                meta_batches = meta_train.get_meta_batches(sr_embedding.detach().cpu().numpy(), meta_bsz)
            else:
                meta_batches = meta_train.get_meta_batches(meta_bsz)  # neighbor, random

        '''model training'''
        for meta_batch in meta_batches:
            learner = maml_model.clone()

            loss_batch = 0.
            for task in meta_batch:
                loss_batch += fast_adapt(task, learner)
            loss_batch /= meta_bsz
            loss_epc += loss_batch

            '''multi-loss learning'''
            optimizer.zero_grad()
            loss_batch.backward()  # print([x.grad for x in optimizer.param_groups[0]['params']])
            optimizer.step()

        '''update augmented knowledge graph'''
        if (e + 1) % args.aug_iter == 0 and e > 0 and args.pr != 0 and e + 1 != args.epoch:
            pr1, pr2 = random.uniform(0, args.pr), random.uniform(0, args.pr)
            aug_adj1, aug_rel_adj1 = kgdata.generate_aug_graph(kgdata.triples1, kgdata.kg1_ent_num, kgdata.rel_num,
                                                               kgdata.kg1_ent_ids, kgdata.ent2node1, kgdata.d_v1,
                                                               pr=pr1)
            aug_adj2, aug_rel_adj2 = kgdata.generate_aug_graph(kgdata.triples2, kgdata.kg2_ent_num, kgdata.rel_num,
                                                               kgdata.kg2_ent_ids, kgdata.ent2node2, kgdata.d_v2,
                                                               pr=pr2)

        # if (e + 1) % 10 == 0:
        print(f"epoch: {e + 1}, loss: {round(loss_epc.item(), 3)}, time: {round((time.time() - t_start), 2)}\n")

        # ======= evaluate ==========
        maml_model.eval()
        sr_embedding, tg_embedding = model(phase="norm")
        sr_embedding, tg_embedding = sr_embedding.detach().cpu().numpy(), tg_embedding.detach().cpu().numpy()
        mrr, hit_p, success_p = eval_hit_p(sr_embedding, tg_embedding,
                                            kgdata.mapped_train_pair, kgdata.mapped_test_pair, k=10)
        # mrr, hit_p, success_p = evaluate_na(kgdata.mapped_train_pair, kgdata.mapped_test_pair, k=10)
        if mrr > mrr_best:
            hit_best = hit_p
            mrr_best = mrr
            success_best = success_p
            if not args.adapt:
                write_pickle([sr_embedding, tg_embedding], save_best_embeds_path)
        print('Common MRR_best: {:.4f}, Hit_best: {:.4f}, Success:{:.4f}'.format(mrr_best, hit_best, success_best))

        # if self.adapt and epoch > n_epochs-30 and mrr > mrr_best-0.005:
        if args.adapt and e+1 > 6 and mrr > mrr_best - 0.005:  # and e+1 > 2
            # adaptive testing
            test_batches = meta_test.get_test_batches(sr_embedding)
            mrr, hit_p, success_p, best_adapt_map_emb = hit_test_adapt(test_batches, maml_model, args.fast_lr,
                                                                      adaptation_steps=2)
            if mrr > mrr_best_adapt:
                mrr_best_adapt = mrr
                hit_best_adapt = hit_p
                success_best_adapt = success_p
                # best_test_batches = test_batches
                # torch.save(maml_model.state_dict(), save_model_path)
                # best_adapt_map_emb = [emb.detach().cpu().numpy() for emb in best_adapt_map_emb]
                write_pickle(best_adapt_map_emb, save_adapt_best_embeds_path)
            print('== Adapt best == MRR: {:.4f}, Hit: {:.4f}. Success: {:.4f} ==='.format(mrr_best_adapt, hit_best_adapt, success_best_adapt))

        t_start = time.time()
        # del sr_embedding, tg_embedding

    print("[evaluating...]\n")
    print('--- Final Results ---')
    if args.adapt:
        emb_s, emb_t = read_pickle(save_adapt_best_embeds_path)
        report_hit(emb_s, emb_t)
    else:
        emb_s, emb_t = read_pickle(save_best_embeds_path)
        report_hit(emb_s, emb_t)

except KeyboardInterrupt:
    print('-' * 40)
    print(f'Exiting from training early, epoch {e + 1}')

# # STEP: begin testing
# print("[evaluating...]\n")
# # del neg1_left, neg1_right, neg2_left, neg2_right
# if args.pr != 0:
#     del aug_adj1, aug_rel_adj1, aug_adj2, aug_rel_adj2
# try:
#     # hit_1_score, hit_k_score, mrr = \
#     evaluate_na(train_pair=kgdata.mapped_train_pair, test_pair=kgdata.mapped_test_pair,
#                 k=args.k, phase="test", model_path=model_path)
#     # print("----------------final score----------------\n")
#     # print(f'+ total time consume: {datetime.timedelta(seconds=int(time.time()-t_total_start))}\n')
#     # print(f"+ Hit@1: {round(hit_1_score, 3)}\n")
#     # print(f"+ Hit@{args.k}: {round(hit_k_score, 3)}\n")
#     # print(f"+ MRR: {round(mrr, 3)}\n")
#     # # print(f"+ mean rank: {round(mean, 3)}\n")
#     # print("-------------------------------------------\n")
#     # # record experimental results
#     # if args.result:
#     #     print("---------------save result-----------------\n")
#     #     with open(f'result/{args.model}_{args.task}.csv', 'a', encoding='utf-8') as file:
#     #         file.write('\n')
#     #         file.write(f"{args.model}, {round(hit_1_score, 3)}, {round(hit_k_score, 3)}, {round(mrr, 3)}, {args.ent_dim}, {args.rel_dim}, {args.lr}, {args.pr}, {args.aug_balance}, {args.eval_metric}, {args.neg_margin}")
#     #     print("-------------------------------------------\n")
# except KeyboardInterrupt:
#     sys.exit()



import argparse
import random
import os


def get_arguments():
    parser = argparse.ArgumentParser()
    # My arguments
    parser.add_argument('--scale', type=str, default='small', help='dataset scale, '
                                                                   'small -> IDS15K'
                                                                   'medium -> IDS100K'
                                                                   'large -> DBP1M')
    parser.add_argument('--ds', type=str, default='na', help='dataset from')
    # parser.add_argument('--lang', type=str, default='fr', help='dataset language (fr, de)')
    parser.add_argument('--dataset', type=str, default='FT', help='FT, DBLP')
    parser.add_argument('--k', type=int, default=-1, help='mini-batch number')
    parser.add_argument('--it_round', type=int, default=1)
    parser.add_argument('--train_ratio', type=float, default=0.5)
    parser.add_argument("--epoch", type=int, default=-1, help="number of epochs to train")
    parser.add_argument('--model', type=str, default='dual-amn', help='model used for training, '
                                                                        'including [gcn-align, rrea, dual-amn,'
                                                                        ' gcn-large, rrea-large, dual-large].'
                                                                        '\'-large\' indicates the '
                                                                        'sampling version of the model')
    parser.add_argument("--save_folder", type=str, default='../../output/results/')
    parser.add_argument("--result_folder", type=str, default='logs/')
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--enhance", type=str, default='sinkhorn', help='mini-batch normalization')
    parser.add_argument("--samplers", type=str, default='CST', help='C -> CMCS, S->ISCS(src->trg), T-> ISCS(trg->src)')
    parser.add_argument('--local_only', action='store_true', default=False)
    parser.add_argument('--no_csls', action='store_true', default=False)
    parser.add_argument("--skip_if_complete", action='store_true', default=False)
    parser.add_argument("--max_sinkhorn_sz", type=int, default=33000,
                        help="max matrix size to run Sinkhorn iteration"
                             ", if the matrix size is higher than this value"
                             ", it will calculate kNN search without normalizing to avoid OOM"
                             ", default is set for 33000^2 (for RTX3090)."
                             " could be set to higher value in case there is GPU with larger memory")
    parser.add_argument("--gcn_max_iter", type=int, default=-1, help="max iteration of GCN for partition")
    parser.add_argument("--cuda", default='cuda:1', help="cuda:{n} or gpu")
    parser.add_argument("--faiss_gpu", action="store_true", default=True, help="whether to use FAISS GPU")
    parser.add_argument("--norm", action="store_true", default=True, help="whether to normalize embeddings")
    return parser.parse_args()


def seed_torch(seed=2023):
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

global_arguments = get_arguments()

# from framework import *
from dataset import *
import time
import utils


# seed_torch(2023)

scale = global_arguments.scale
ds = global_arguments.ds
dataset = global_arguments.dataset
train_ratio = global_arguments.train_ratio

device = global_arguments.cuda
norm = global_arguments.norm
n_semi_iter = global_arguments.it_round
partition_k = global_arguments.k
model_name = global_arguments.model
faiss_use_gpu = global_arguments.faiss_gpu
max_sinkhorn_sz = global_arguments.max_sinkhorn_sz
gcn_max_iter = global_arguments.gcn_max_iter
save_folder = global_arguments.save_folder
enhance = global_arguments.enhance
sampler_methods = global_arguments.samplers
fuse_global = not global_arguments.local_only
apply_csls = not global_arguments.no_csls
result_folder = global_arguments.result_folder
skip_if_complete = global_arguments.skip_if_complete
model_dims = {'gcn-align': 200,
              'rrea': 600,
              'rrea-large': 600,
              'dual-amn': 768,
              'dual-large': 768,
              'gcn-large': 200}

PHASE_TRAINING = 1
PHASE_PARTITION = 2

if partition_k == -1:
    partition_k = dict(small=5, medium=10, large=30)[scale]

if gcn_max_iter == -1:
    gcn_max_iter = dict(small=800, medium=1500, large=3000)[scale]
if global_arguments.epoch == -1:
    train_epoch = \
        {'gcn-align': [2000] * n_semi_iter, 'rrea': [1200] * n_semi_iter, 'dual-amn': [60] + [5] * (n_semi_iter - 1),
         'gcn-large': [50], 'dual-large': [20], 'rrea-large': [50]}[
            model_name]
    if model_name in ['dual-large'] and scale == 'large':
        train_epoch = [10]
else:
    train_epoch = global_arguments.epoch


def ablation_args(val, default_val, name=''):
    if val != default_val:
        return f'_{name}{val}'
    return ''


def get_suffix(phase):
    now = 'embeddings.pkl' if PHASE_TRAINING == phase else 'sim.pkl'
    if phase == PHASE_PARTITION:
        now += f"_{scale}_{ds}_{dataset}_{model_name}_gcn{gcn_max_iter}_k{partition_k}_it{n_semi_iter}_norm{norm}"

        now += ablation_args(enhance, 'sinkhorn')
        # now += ablation_args(sampler_methods, 'CST')
    elif phase == PHASE_TRAINING:
        now += f"_{scale}_{ds}_{dataset}_{model_name}_it{n_semi_iter}"
    else:
        raise NotImplementedError
    now += ablation_args(train_ratio, 30)
    return now


def save_curr_objs(objs, phase):
    saveobj(objs, save_folder + get_suffix(phase))


def load_curr_objs(phase):
    try:
        return readobj(save_folder + get_suffix(phase))
    except:
        return readobj(save_folder + get_suffix(phase))


if model_name == 'rrea':
    norm = True


def step1_training():
    if skip_if_complete:
        ok = False
        try:
            load_curr_objs(PHASE_TRAINING)
        except:
            ok = True

        if ok:
            add_log('skip_training', True)
            return

    ea = load_dataset(scale, ds, dataset, train_ratio=train_ratio)
    from align_batch import get_whole_batch
    # TODO fix training
    whole_batch = get_whole_batch(ea, backbone=model_name)
    model = whole_batch.model
    tic = time.time()
    for ep in range(n_semi_iter):
        model.train1step(train_epoch[ep])
        if ep < n_semi_iter - 1:
            # embeddings = model.get_curr_embeddings('cpu')
            model.mraea_iteration()
    # model.update_trainset(get_seed(ea, embeddings))
    embeddings = model.get_curr_embeddings()
    toc = time.time()
    # add_log('training time', toc - tic)
    # save_curr_objs((ea, embeddings), PHASE_TRAINING)
    # saveobj(embeddings, save_folder + 'embeds_all_dualamn.pkl')

    del ea, embeddings


def understandable_sampler_args():
    cst = dict(C='KMeansXGB', S='MetisGCN_s2t', T='MetisGCN_t2s')
    return list(map(lambda x: cst[x], sampler_methods))


def run():
    log_file = f'{result_folder}{scale}_{dataset}_{model_name}'
    # from utils.log import create_logger
    # from time import strftime, localtime
    # logname = strftime("logging/JMAC_{}_tr{}_%m-%d_%H:%M:%S.txt".format(
    #     dataset, train_ratio
    # ), localtime())
    # logger = create_logger(
    #     __name__, silent=False,
    #     to_disk=True, log_file=logname)

    torch.cuda.set_device(0)
    # eval_large()
    step = global_arguments.step
    if step == 1:
        print('------------Training------------------')
        step1_training()
    # with open(log_file, 'a') as f:
    #     f.write('---------------------\n')
    #     f.write(f'step : {step}\n')
    #     for k, v in global_dict.items():
    #         f.write(f'{k} : {v}\n')


if __name__ == '__main__':
    run()

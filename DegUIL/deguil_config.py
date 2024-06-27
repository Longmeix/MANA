import os
import torch
from time import strftime, gmtime, localtime
import random
import numpy as np

ratio = 0.5
k = 10  # hit@k
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = 'DegUIL'
dataset = 'FT'
options = 'structure'
epochs = 100

# ---- netEncode ----
dim_feature = 256
num_layer = 2
lr = 5e-4
weight_decay = 1e-5
batch_size = 2 ** 7

# ---- UILAggregator ----
neg_num = 5
supervised = True
msa_out_dim = 64
alpha = 10
beta = 1

# ----- other config -----
percent = 99

MLP_hid = 128
fast_lr = 0.1
n_way = 5
k_shot = 5
support = 'similarity'
save_best_embeds_path, save_model_path = None, None
save_adapt_best_embeds_path = None

log = strftime("logs/{}_{}_{:.1f}_%m-%d_%H:%M:%S.txt".format(model, fast_lr, ratio), localtime())


def init_args(args):
    global device, model, epochs, log, dataset, ratio, fast_lr, support, n_way, k_shot,\
            save_best_embeds_path, save_model_path, save_adapt_best_embeds_path
    global alpha, beta

    # ratio = args.ratio
    device = args.device
    model = args.model
    dataset = args.dataset
    ratio = args.ratio
    fast_lr = args.fast_lr
    support = args.support
    n_way = args.n_way
    k_shot = args.k_shot
    cur_path = os.path.abspath(__file__)
    # print('===>', os.path.dirname(cur_path))
    log_path = os.path.dirname(cur_path) + '/logs'
    if not os.path.exists(log_path):
        os.mkdir(log_path)
    # best_embs_path = './data/bestEmbs'
    # if not os.path.exists(best_embs_path):
    #     os.makedirs(best_embs_path)

    # save path
    save_folder = args.folder_dir + 'models/' + args.model
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_best_embeds_path = save_folder + '/{}_{}_best_embs.pkl'.format(dataset, ratio)
    save_adapt_best_embeds_path = save_folder + '/{}_{}_best_embs_adapt.pkl'.format(dataset, ratio)
    save_model_path = save_folder + '/model.pt'

    # log = strftime("logs/{}_{}_{:.1f}_%m-%d_%H:%M:%S.txt".format(model, fast_lr, ratio), localtime())
    # log = strftime("logs/{}_{}_{}_{:.1f}_%m-%d_%H:%M:%S.txt".format(model, support, fast_lr, ratio), localtime())
    log = strftime(log_path + "/{}_{}n_{}k_{}_{}_{:.1f}_%m-%d_%H:%M:%S.txt".format(model, n_way, k_shot, support, fast_lr, ratio), localtime())
    print('===>' + log)


def seed_torch(seed=2022):
    random.seed(seed)  # Python random module.
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)  # Numpy module.
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
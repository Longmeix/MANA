from time import strftime, gmtime, localtime
import torch
import os

model = 'DegUIL'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


ratio = 0.5
k = 10

# ----- net encode -----
dim_feature = 256
dim_out = 64

# data process, task generate
tail_node_threshold = 5

batch_size = 2 ** 7
neg_num = 5
supervised = True


# save similarity when hit best
sims_path = 'data/{}/{}_sims.pkl'.format(model, model)

log = strftime("logs/{}_{:.1f}_%m-%d_%H:%M:%S.txt".format(
        model, ratio
    ), localtime())

save_best_embeds_path, save_model_path = None, None
save_adapt_best_embeds_path = None

save_folder = './'


def init_args(args):
    global device, ratio, model, epochs, k, log, fast_lr, save_best_embeds_path, save_model_path, save_adapt_best_embeds_path

    ratio = args.ratio
    device = args.device
    model = args.model
    # epochs = args.epochs
    k = args.top_k
    fast_lr = args.fast_lr
    dataset = args.dataset

    log = strftime("logs/{}_f{}_r{:.1f}_%m-%d_%H:%M:%S.txt".format(
        model, fast_lr, ratio
    ), localtime())

    # log = strftime("logs/{}_{}_{}_{:.1f}_%m-%d_%H:%M:%S.txt".format(
    #     model, args.support, fast_lr, ratio
    # ), gmtime())

    # log = strftime("logs/{}_{}_{}_{:.1f}_%m-%d_%H:%M:%S.txt".format(
    #     model, args.k_shot, fast_lr, ratio
    # ), gmtime())

    # log = strftime("logs/{}_{}_r{:.1f}_%m-%d_%H:%M:%S.txt".format(
    #     model, fast_lr, ratio
    # ), gmtime())

    # log = strftime("logs/{}_{}way_{}shot_{}_{:.1f}_%m-%d_%H:%M:%S.txt".format(  # deeplink
    #     model, args.n_way, args.k_shot, fast_lr, ratio
    # ), gmtime())

    # if hasattr(args, 'meta_test_cluster'):
    #     log = strftime("logs/{}_{}_r{:.1f}_c{}_%m-%d_%H:%M:%S.txt".format(
    #             model, fast_lr, ratio, args.meta_test_cluster
    #         ), localtime())

    if hasattr(args, 'ptb_p'):
        log = strftime("logs/{}_ptb{}_{}_{:.1f}_%m-%d_%H:%M:%S.txt".format(
            model, args.ptb_p, fast_lr, ratio
        ), localtime())

    # save path
    save_folder = args.folder_dir + 'models/' + args.model
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    save_best_embeds_path = save_folder + '/{}_{}_best_embs.pkl'.format(dataset, ratio)
    save_adapt_best_embeds_path = save_folder + '/{}_{}_best_embs_adapt.pkl'.format(dataset, ratio)
    save_model_path = save_folder + '/model.pt'


if __name__ == '__main__':
    pass
from learn2learn.algorithms import maml_update
from torch import autograd
from mapping_model import MappingModel
import numpy as np
import torch
import time
from models.base import UIL
from prep_meta_tasks.task_generate import MetaTaskRandom, MetaTaskNeighbor, MetaTaskSimilarity
# from prep_meta_tasks.meta_tasks_cluster import MetaTaskRandom, MetaTaskNeighbor, MetaTaskSimilarity
import learn2learn as l2l
from tqdm import trange
from utils.general import write_pickle, read_pickle
import config as cfg
import torch.nn.functional as F
from itertools import chain


class DeepLink_dualMap(UIL):
    def __init__(self, embeds, adj_s, adj_t, links, args):
        """
        Parameters
        ----------
        source_dataset: Dataset
            Dataset object of source dataset
        target_dataset: Dataset
            Dataset object of target dataset
        args: argparse.ArgumentParser.parse_args()
            arguments as parameters for model.
        """
        super(DeepLink_dualMap, self).__init__(links, args.top_k)
        self.log.info(args)
        self.adj_s = adj_s
        self.adj_t = adj_t
        self.alpha = args.alpha
        self.map_batchsize = args.batch_size_mapping
        # self.cuda = args.cuda
        self.embed_dim = args.embed_dim
        # self.embedding_epochs = args.embedding_epochs
        self.supervised_epochs = args.supervised_epochs
        self.unsupervised_epochs = args.unsupervised_epochs
        self.supervised_lr = args.supervised_lr
        self.unsupervised_lr = args.unsupervised_lr
        # self.num_cores = args.num_cores

        # gt = load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        # self.full_gt = {}
        # self.full_gt.update(gt)
        # test_gt = load_gt(args.groundtruth, source_dataset.id2idx, target_dataset.id2idx, 'dict')
        # self.full_gt.update(test_gt)
        # self.full_gt = {self.source_dataset.id2idx[k]: self.target_dataset.id2idx[v] for k, v in self.full_gt.items()}
        # self.train_dict = {self.source_dataset.id2idx[k]: self.target_dataset.id2idx[v] for k,v in gt.items()}

        # self.source_embedding = None
        # self.target_embedding = None
        # self.source_after_mapping = None
        # self.source_train_nodes = np.array(list(self.train_dict.keys()))
        # self.source_anchor_nodes = np.array(list(self.train_dict.keys()))
        # anchors
        self.train_anchor_links, self.test_anchor_links = links
        self.train_anchors_s, self.train_anchors_t = [list(x) for x in list(zip(*self.train_anchor_links))]
        self.test_anchors_s, self.test_anchors_t = [list(x) for x in list(zip(*self.test_anchor_links))]
        # self.anchors_s = self.train_anchors_s + self.test_anchors_s
        # self.anchors_t = self.train_anchors_t + self.test_anchors_t
        # self.anchors_df = pd.DataFrame(self.train_anchor_links + self.test_anchor_links)
        self.anchors = self.train_anchor_links + self.test_anchor_links
        # # save path
        # save_folder = args.folder_dir + args.model
        # if not os.path.exists(save_folder):
        #     os.mkdir(save_folder)
        # self.save_best_embeds_path = save_folder + '/best_embs.pkl'
        # self.save_model_path = save_folder + '/model.pt'
        self.save_best_embeds_path = cfg.save_best_embeds_path
        self.save_model_path = cfg.save_model_path

        self.adapt = args.adapt
        self.meta_test_cluster = args.meta_test_cluster
        self.support = args.support
        self.fast_lr = args.fast_lr
        # self.meta_bsz = args.meta_bsz
        self.k_shot = args.k_shot
        self.n_way = args.n_way
        self.save_adapt_best_embeds_path = cfg.save_adapt_best_embeds_path

        self.hidden_dim1 = args.hidden_dim1
        self.hidden_dim2 = args.hidden_dim2

        self.top_k = args.top_k
        self.device = args.device

        # self.embed_s, self.embed_t = [torch.from_numpy(emb).to(self.device) for emb in embeds]
        self.embed_s, self.embed_t = [F.normalize(torch.from_numpy(emb), dim=1, p=2).to(self.device) for emb in embeds]

        self.mapping_model = MappingModel(
            embedding_dim=self.embed_dim,
            hidden_dim1=self.hidden_dim1,
            hidden_dim2=self.hidden_dim2,
            source_embedding=self.embed_s,
            target_embedding=self.embed_t
        ).to(self.device)

    # def get_ali
    def train_align(self):
        mapping_model = self.mapping_model
        # unsupervised
        # m_optimizer_us = torch.optim.SGD(filter(lambda p: p.requires_grad, mapping_model.parameters()),
        #                                  lr = self.unsupervised_lr, momentum=0.9)
        m_optimizer_us = torch.optim.Adam(mapping_model.parameters(), lr=self.unsupervised_lr)
        self._unsupervised_mapping_train(mapping_model, m_optimizer_us)
        #
        # m_optimizer_s = torch.optim.SGD(filter(lambda p: p.requires_grad, mapping_model.parameters()),
        #                                 lr = self.supervised_lr, momentum=0.9)
        # m_optimizer_s = torch.optim.Adam(mapping_model.parameters(), lr=self.supervised_lr)
        self._supervised_mapping_train(mapping_model)

    def _unsupervised_mapping_train(self, model, optimizer):
        batch_size = self.map_batchsize
        n_epochs = self.unsupervised_epochs
        for epoch in range(1, n_epochs+1):
            batches = self.load_data(self.anchors, batch_size=batch_size)
            loss_epoch = 0.
            start = time.time()
            for batch in batches:
                source_batch, target_batch = batch
                loss = model.unsupervised_loss(source_batch, target_batch)
                loss_epoch += loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            un_mapping_epoch_time = time.time() - start
            print(f'Epoch: {epoch}, loss: {loss_epoch}, time: {un_mapping_epoch_time}')

    def fast_adapt(self, task, learner, fast_lr, adaptation_steps=1):
        support_links, query_links = task
        # support_links = query_links  # tr-tr

        # # freeze learner former layer
        # not_freezen_list = ['theta.4', 'inversed_theta.4']
        # for p_name, param in learner.module.named_parameters():
        #     if p_name not in not_freezen_list:
        #         param.requires_grad = False
        # ori_freeze = list(learner.module.theta[0].parameters())[0][0][0]
        # ori = list(learner.module.theta[4].parameters())[0][0][0]

        if self.adapt:
            # Adapt the model
            support_links = list(filter(None, support_links))
            if len(support_links) > 0:
                for step in range(adaptation_steps):
                    train_loss = learner.module.supervised_loss(support_links, alpha=self.alpha)
                    # TODO: only update the last layer
                    # learner.adapt(train_loss)
                    grads_theta = autograd.grad(train_loss,
                                                learner.module.theta[-1].parameters(),
                                                retain_graph=False, create_graph=False, allow_unused=True)
                    learner.module.theta[-1] = maml_update(learner.module.theta[-1], lr=fast_lr, grads=grads_theta)
                    # grads_inversed = autograd.grad(train_loss, learner.module.inversed_theta[4].parameters(),
                    #                                retain_graph=True, create_graph=True, allow_unused=False)
                    # learner.module.inversed_theta[4] = maml_update(learner.module.inversed_theta[4], lr=fast_lr, grads=grads_inversed)

        # new_freeze = list(learner.module.theta[0].parameters())[0][0][0]
        # new = list(learner.module.theta[4].parameters())[0][0][0]
        # print('freeze update:', ori_freeze.item() == new_freeze.item())
        # print('update params:', ori.item() == new.item())

        # emb_s, emb_t = self.get_embeds(\
        # Evaluate the adapted model
        valid_loss = learner.module.supervised_loss(query_links, alpha=self.alpha)

        return valid_loss

    def _supervised_mapping_train(self, map_model):
        # meta-learning
        # meta_bsz = self.meta_bsz
        meta_bsz = 8
        # meta_bsz = 64
        n_way = self.n_way
        k_shot = self.k_shot
        fast_lr = self.fast_lr
        hit_best, hit_best_adapt = 0., 0.
        mrr_best, mrr_best_adapt = 0., 0.
        success_best, success_best_adapt = .0, .0
        n_epochs = self.supervised_epochs

        if not self.adapt or self.meta_test_cluster == 1:
            from prep_meta_tasks.task_generate import MetaTaskRandom, MetaTaskNeighbor, MetaTaskSimilarity
        else:
            from prep_meta_tasks.meta_tasks_cluster import MetaTaskRandom, MetaTaskNeighbor, MetaTaskSimilarity

        if self.support == 'random':
            meta_train = MetaTaskRandom(self.train_anchor_links, self.train_anchor_links, n_way=n_way, k_shot=k_shot, q_query=1)
            meta_test = MetaTaskRandom(self.test_anchor_links, self.train_anchor_links, n_way=1, k_shot=k_shot, q_query=1)
            test_batches = meta_test.get_test_batches()
        elif self.support == 'neighbor':
            meta_train = MetaTaskNeighbor(self.adj_s, self.train_anchor_links, self.train_anchor_links, n_way=n_way, k_shot=k_shot, q_query=1)
            meta_test = MetaTaskNeighbor(self.adj_s, self.test_anchor_links, self.train_anchor_links, n_way=1, k_shot=k_shot, q_query=1)
            test_batches = meta_test.get_test_batches()
        elif self.support == 'similarity':
            embed_s_np = self.embed_s.detach().cpu().numpy()
            meta_train = MetaTaskSimilarity(self.train_anchor_links, self.train_anchor_links, n_way=n_way, k_shot=k_shot, q_query=1)
            meta_test = MetaTaskSimilarity(self.test_anchor_links, self.train_anchor_links, n_way=self.meta_test_cluster, k_shot=k_shot, q_query=1)
            test_batches = meta_test.get_test_batches(embed_s_np)
        else:
            print('Support Strategy Error')

        maml_map = l2l.algorithms.MAML(map_model, lr=fast_lr, first_order=True)
        optimizer = torch.optim.Adam(map_model.parameters(), self.supervised_lr)

        self.log.info('Starting supervised mapping training...')
        time_local_mapping, time_global_mapping = [], []
        time4training_start = time.time()

        for epoch in range(1, n_epochs + 1):
            maml_map.train()
            loss_epc = .0
            # get data batch, shuffle
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
                    loss_batch += self.fast_adapt(task, learner, fast_lr)
                loss_batch /= meta_bsz
                loss_epc += loss_batch
                # update parameters of net
                self.optimize(optimizer, loss_batch)

            self.log.info('Epoch: {}, loss: {:.4f}'.format(epoch, loss_epc.item() / len(meta_batches)))
            # print('Epoch: {}, loss: {:.4f}'.format(epoch, loss_epc))

            # ======= evaluate ==========
            maml_map.eval()
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
            self.log.info('Common MRR_best: {:.4f}, Hit_best: {:.4f}, Success:{:.4f}'.format(mrr_best, hit_best, success_best))

            if self.adapt and epoch > n_epochs-5 and mrr > mrr_best-0.01:
            # if self.adapt and mrr > mrr_best-0.01:
                # adaptive testing
                time0 = time.time()
                test_loss, mrr, hit_p, success_p, best_adapt_map_emb = self.hit_test_adapt(test_batches, maml_map, fast_lr, adaptation_steps=2)
                local_time = round(time.time() - time0, 2)
                time_local_mapping.append(local_time)

                if mrr > mrr_best_adapt:
                    mrr_best_adapt = mrr
                    hit_best_adapt = hit_p
                    success_best_adapt = success_p

                    write_pickle(best_adapt_map_emb, self.save_adapt_best_embeds_path)
                self.log.info('== Adapt best == MRR: {:.4f}, Hit: {:.4f}, Success: {:.4f} ==='
                              .format(mrr_best_adapt, hit_best_adapt, success_best_adapt))

        time4training_all = time.time() - time4training_start
        self.log.info('Time for training: {:.2f}'.format(time4training_all))
        self.log.info('--- Final Results ---')
        if self.adapt:
            # maml_map.load_state_dict(torch.load(self.save_model_path))
            # maml_map.eval()
            # res = self.hit_test_adapt(test_batches, maml_map, fast_lr)
            emb_s, emb_t = read_pickle(self.save_adapt_best_embeds_path)
            self.report_hit_multi(emb_s, emb_t)
            self.log.info('Time of local mapping: {:.2f}'.format(np.mean(time_local_mapping)))
            self.rename_log('/mrr{:.2f}_{}_'.format(mrr_best_adapt * 100, self.adapt).join(
                cfg.log.split('/')
            ))
        else:
            emb_s, emb_t = read_pickle(self.save_best_embeds_path)
            self.report_hit_multi(emb_s, emb_t)
            self.log.info('Time of global mapping: {:.2f}'.format(np.mean(time_global_mapping)))
            self.rename_log('/mrr{:.2f}_{}_'.format(mrr_best * 100, self.adapt).join(
                cfg.log.split('/')
            ))

    def get_embeds(self, is_map=False):
        # get node embedding of two networks
        if is_map:
            embed_s_after_mapping = self.mapping_model(self.embed_s, mode='map')
        else:
            embed_s_after_mapping = self.embed_s
        return embed_s_after_mapping, self.embed_t

    def hit_test_adapt(self, test_batches, base_model, fast_lr, adaptation_steps=2):
        # cover_list, hit_list = [], []
        # hit_list_class = []
        test_losses = []
        cnt = 0
        test_tasks = test_batches[0]
        emb_s_map, emb_t_map = base_model.module(self.embed_s, mode='map'), self.embed_t

        # # freeze learner former layer
        # not_freezen_list = ['theta.4', 'inversed_theta.4']
        # for p_name, param in base_model.named_parameters():
        #     if p_name not in not_freezen_list:
        #         param.requires_grad = False

        for i in trange(len(test_tasks)):
            task = test_tasks[i]
            learner = base_model.clone()
            support_links, query_link = task

            support_links = list(filter(None, support_links))
            if len(support_links) > 0:
                for step in range(adaptation_steps):
                    train_loss = learner.module.supervised_loss(support_links, alpha=self.alpha)
                    # learner.adapt(train_loss)
                    grads_theta = autograd.grad(train_loss,
                                                learner.module.theta[4].parameters(),
                                                retain_graph=False, create_graph=False, allow_unused=True)
                    learner.module.theta[4] = maml_update(learner.module.theta[4], lr=fast_lr, grads=grads_theta)
                    # grads_inversed = autograd.grad(train_loss, learner.module.inversed_theta[4].parameters(),
                    #                                retain_graph=False, create_graph=True, allow_unused=False)
                    # learner.module.inversed_theta[4] = maml_update(learner.module.inversed_theta[4], lr=fast_lr, grads=grads_inversed)

            # Evaluate the adapted model
            with torch.no_grad():
                test_loss = learner.module.supervised_loss(query_link, alpha=self.alpha)
                emb_s_adapt_map = learner.module(self.embed_s, mode='map')
            test_losses.append(test_loss.item())
            test_num = len(query_link)
            cnt += test_num

            # refine global embeddings with local mapping
            test_sid, test_tid = [list(i) for i in list(zip(*query_link))]
            emb_s_map[test_sid] = emb_s_adapt_map[test_sid]

        mrr, hit_p, success_p = self.eval_metrics_na(emb_s_map, emb_t_map, self.k)
        test_loss = np.mean(test_losses)

        # test_loss, mrr, hit_p, success_p = np.mean(test_losses), np.sum(mrr_list) / cnt, np.sum(hit_list) / cnt, np.sum(success_list) / cnt
        self.log.info('-- Test loss: {:.4f}, MRR: {:.4f}, Hit: {:.4f}, Success:{:.4f}  --'.format(
            test_loss, mrr, hit_p, success_p
        ))

        return test_loss, mrr, hit_p, success_p, (emb_s_map, emb_t_map)

    # def mapping_train_(self, model, optimizer, mode='s'):
    #     if mode == 's':
    #         source_train_nodes = self.source_train_nodes
    #     else:
    #         source_train_nodes = self.source_anchor_nodes
    #
    #     batch_size = self.map_batchsize
    #     n_iters = len(source_train_nodes)//batch_size
    #     assert n_iters > 0, "batch_size is too large"
    #     if(len(source_train_nodes) % batch_size > 0):
    #         n_iters += 1
    #     print_every = int(n_iters/4) + 1
    #     total_steps = 0
    #     train_dict = None
    #     if mode == 's':
    #         n_epochs = self.supervised_epochs
    #         train_dict = self.train_dict
    #     else:
    #         n_epochs = self.unsupervised_epochs
    #         train_dict = self.full_gt
    #
    #
    #     for epoch in range(1, n_epochs+1):
    #         # for evaluate time
    #         start = time.time()
    #
    #         print("Epoch {0}".format(epoch))
    #         np.random.shuffle(source_train_nodes)
    #         for iter in range(n_iters):
    #             source_batch = source_train_nodes[iter*batch_size:(iter+1)*batch_size]
    #             target_batch = [train_dict[x] for x in source_batch]
    #             source_batch = torch.LongTensor(source_batch)
    #             target_batch = torch.LongTensor(target_batch)
    #             if self.cuda:
    #                 source_batch = source_batch.cuda()
    #                 target_batch = target_batch.cuda()
    #             optimizer.zero_grad()
    #             start_time = time.time()
    #             if mode == 'us':
    #                 loss = model.unsupervised_loss(source_batch, target_batch)
    #             else:
    #                 loss = model.supervised_loss(source_batch, target_batch, alpha=self.alpha)
    #             loss.backward()
    #             optimizer.step()
    #             if total_steps % print_every == 0 and total_steps > 0:
    #                 print("Iter:", '%03d' %iter,
    #                       "train_loss=", "{:.5f}".format(loss.item()),
    #                       )
    #
    #             total_steps += 1
    #         if mode == "s":
    #             self.s_mapping_epoch_time = time.time() - start
    #         else:
    #             self.un_mapping_epoch_time = time.time() - start


# def load_gt(path, id2idx_src, id2idx_trg, format='matrix', convert=False):
#     conversion_src = type(list(id2idx_src.keys())[0])
#     conversion_trg = type(list(id2idx_trg.keys())[0])
#     if format == 'matrix':
#         gt = np.zeros((len(id2idx_src.keys()), len(id2idx_trg.keys())))
#         with open(path) as file:
#             for line in file:
#                 src, trg = line.strip().split()
#                 gt[id2idx_src[conversion_src(src)], id2idx_trg[conversion_trg(trg)]] = 1
#         return gt
#     else:
#         gt = {}
#         with open(path) as file:
#             for line in file:
#                 src, trg = line.strip().split()
#                 if convert:
#                     gt[id2idx_src[conversion_src(src)]] = id2idx_trg[conversion_trg(trg)]
#                 else:
#                     gt[conversion_src(src)] = conversion_trg(trg)
#         return gt


if __name__ == '__main__':
    pass
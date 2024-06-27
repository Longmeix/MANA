# from algorithms.network_alignment_model import NetworkAlignmentModel
import networkx as nx
from embedding_model import PaleEmbedding
from mapping_model import PaleMappingLinear, PaleMappingMlp
# from input.dataset import Dataset
# from utils.graph_utils import load_gt
from utils.general import read_pickle, write_pickle
import torch
import numpy as np
from tqdm import trange
import os
import time
from models.base import UIL
from prep_meta_tasks.task_generate import MetaTaskRandom, MetaTaskNeighbor, MetaTaskSimilarity
import learn2learn as l2l
import config as cfg


class PALE(UIL):
    def __init__(self, adj_s, adj_t, links, args):
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
        # self.source_dataset = source_dataset
        # self.target_dataset = target_dataset
        # self.source_path = args.source_dataset
        super(PALE, self).__init__(links, k=args.top_k)
        self.graph_s = nx.from_numpy_matrix(adj_s, create_using=nx.Graph())
        self.graph_t = nx.from_numpy_matrix(adj_t, create_using=nx.Graph())
        self.adj_s = adj_s

        self.emb_batchsize = args.batch_size_embedding
        self.map_batchsize = args.batch_size_mapping
        self.emb_lr = args.learning_rate1
        self.neg_sample_size = args.neg_sample_size
        self.embedding_dim = args.embedding_dim
        self.emb_epochs = args.embedding_epochs
        self.map_epochs = args.mapping_epochs
        self.mapping_model = args.mapping_model
        self.map_act = args.activate_function
        self.map_lr = args.learning_rate2
        self.args = args
        self.device = args.device
        self.adapt = args.adapt
        self.support = args.support
        self.fast_lr = args.fast_lr
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.save_best_embeds_path = cfg.save_best_embeds_path
        self.save_model_path = cfg.save_model_path

        # self.gt_train = load_gt(args.train_dict, source_dataset.id2idx, target_dataset.id2idx, 'dict')

        self.S = None
        self.source_embedding = None
        self.target_embedding = None
        self.source_after_mapping = None
        # self.source_train_nodes = np.array(list(self.gt_train.keys()))
        # anchors
        self.train_anchor_links, self.test_anchor_links = links
        # self.train_anchors_s, self.train_anchors_t = [list(x) for x in list(zip(*self.train_anchor_links))]
        # self.test_anchors_s, self.test_anchors_t = [list(x) for x in list(zip(*self.test_anchor_links))]


    def get_alignment_matrix(self):
        return self.S

    def get_source_embedding(self):
        return self.source_embedding

    def get_target_embedding(self):
        return self.target_embedding

    def align(self, train_emb=True):
        embeds_dir = self.args.folder_dir + self.args.dataset + '/' + self.args.model + '/'
        if not os.path.exists(embeds_dir):
            os.makedirs(embeds_dir)
        embeds_path = embeds_dir + self.args.save_embeds_file.format(self.args.ratio)
        # if embed file exists
        if not os.path.exists(embeds_path):
            print('No existing embedding file.')
        if train_emb or not os.path.exists(embeds_path):
            print('Training embedding...')
            self.learn_embeddings()
            write_pickle([self.source_embedding.cpu().detach().numpy(),
                          self.target_embedding.cpu().detach().numpy()], embeds_path)
        self.source_embedding, self.target_embedding = [torch.from_numpy(emb).to(self.device)
                                                        for emb in read_pickle(embeds_path)]

        # mapping
        if self.mapping_model == 'linear':
            print("Use linear mapping")
            mapping_model = PaleMappingLinear(
                                        embedding_dim=self.embedding_dim,
                                        source_embedding=self.source_embedding,
                                        target_embedding=self.target_embedding,
                                        )
        else:
            print("Use Mlp mapping")
            mapping_model = PaleMappingMlp(
                                        embedding_dim=self.embedding_dim,
                                        source_embedding=self.source_embedding,
                                        target_embedding=self.target_embedding,
                                        activate_function=self.map_act,
                                        )
        self.mapping_model = mapping_model.to(self.device)
        # self.map_optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.mapping_model.parameters()),
        #                                   lr=self.map_lr)
        self.map_optim = torch.optim.Adam(self.mapping_model.parameters(), lr=self.map_lr)

        n_iters = len(self.train_anchor_links) // self.map_batchsize
        assert n_iters > 0, "batch_size is too large"
        if len(self.train_anchor_links) % self.map_batchsize > 0:
            n_iters += 1
        # print_every = int(n_iters/4) + 1
        # total_steps = 0
        n_epochs = self.map_epochs

        hit_best, hit_best_adapt = 0., 0.
        mrr_best, mrr_best_adapt = 0., 0.
        success_best, success_best_adapt = .0, .0
        meta_bsz = 8  # number of tasks per batch
        fast_lr = self.fast_lr
        n_way = self.n_way
        k_shot = self.k_shot

        if self.support == 'random':
            meta_train = MetaTaskRandom(self.train_anchor_links, self.train_anchor_links, n_way=n_way, k_shot=k_shot, q_query=1)
            meta_test = MetaTaskRandom(self.test_anchor_links, self.train_anchor_links, n_way=1, k_shot=k_shot, q_query=1)
            test_batches = meta_test.get_test_batches()
        elif self.support == 'neighbor':
            meta_train = MetaTaskNeighbor(self.adj_s, self.train_anchor_links, self.train_anchor_links, n_way=n_way, k_shot=k_shot, q_query=1)
            meta_test = MetaTaskNeighbor(self.adj_s, self.test_anchor_links, self.train_anchor_links, n_way=1, k_shot=k_shot, q_query=1)
            test_batches = meta_test.get_test_batches()
        elif self.support == 'similarity':
            embed_s_np = self.source_embedding.detach().cpu().numpy()
            meta_train = MetaTaskSimilarity(self.train_anchor_links, self.train_anchor_links, n_way=n_way, k_shot=k_shot, q_query=1)
            meta_test = MetaTaskSimilarity(self.test_anchor_links, self.train_anchor_links, n_way=1, k_shot=k_shot, q_query=1)
            test_batches = meta_test.get_test_batches(embed_s_np)
        else:
            print('Support Strategy Error')

        maml_map = l2l.algorithms.MAML(self.mapping_model, lr=fast_lr, first_order=True)
        # maml_map = l2l.algorithms.MetaSGD(self.mapping_model, lr=fast_lr)

        for epoch in range(1, n_epochs + 1):
            loss_epc = .0
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
                    loss_batch += self.fast_adapt(task, learner)
                loss_batch /= meta_bsz
                loss_epc += loss_batch
                # update parameters of net
                self.optimize(self.map_optim, loss_batch)

            self.log.info('Epoch: {}, loss: {:.4f}'.format(epoch, loss_epc.item() ))

            # print('Epochs: ', epoch)
            # np.random.shuffle(self.train_anchor_links)
            # for iter in range(n_iters):
            #     # source_batch = self.source_train_nodes[iter*self.map_batchsize:(iter+1)*self.map_batchsize]
            #     # target_batch = [self.gt_train[x] for x in source_batch]
            #     batch = self.train_anchor_links[iter*self.map_batchsize : (iter+1)*self.map_batchsize]
            #     # source_batch, target_batch = list(zip(*batch))
            #     # source_batch = torch.LongTensor(source_batch).to(self.device)
            #     # target_batch = torch.LongTensor(target_batch).to(self.device)
            #     start_time = time.time()
            #     self.map_optim.zero_grad()
            #     loss = self.mapping_model.loss(batch)
            #     loss.backward()
            #     self.map_optim.step()
            #     if total_steps % print_every == 0 and total_steps > 0:
            #         print("Iter:", '%03d' %iter,
            #               "train_loss=", "{:.5f}".format(loss.item()),
            #               "time", "{:.5f}".format(time.time()-start_time)
            #               )
            #     total_steps += 1

            # ======= evaluate ==========
            self.mapping_model.eval()
            emb_s_map, emb_t_map = self.get_embeds(is_map=True)
            mrr, hit_p, success_p = self.eval_metrics_na(emb_s_map, emb_t_map, self.k)
            # if hit_p > hit_best:
            if mrr > mrr_best:
                hit_best = hit_p
                mrr_best = mrr
                success_best = success_p
                if not self.adapt:
                    write_pickle([emb_s_map, emb_t_map], self.save_best_embeds_path)
            self.log.info('== Common best == MRR: {:.4f}, Hit: {:.4f}, Success:{:.4f}'.format(mrr_best, hit_best, success_best))

            if self.adapt and epoch > n_epochs-30 and mrr > mrr_best-0.005:
            # if self.adapt and epoch > 10 and mrr > mrr_best-0.005:
                # adaptive testing
                test_loss, mrr, hit_p, success_p, _ = self.hit_test_adapt(test_batches, maml_map, adaptation_steps=2)
                if mrr > mrr_best_adapt:
                    mrr_best_adapt = mrr
                    hit_best_adapt = hit_p
                    success_best_adapt = success_p

                    torch.save(maml_map.state_dict(), self.save_model_path)
                self.log.info('== Adapt best == MRR: {:.4f}, Hit: {:.4f}. Success: {:.4f} ==='
                              .format(mrr_best_adapt, hit_best_adapt, success_best_adapt))

        self.log.info('--- Final Results ---')
        if self.adapt:
            maml_map.load_state_dict(torch.load(self.save_model_path))
            maml_map.eval()
            res = self.hit_test_adapt(test_batches, maml_map)
            adapted_best_emb = res[-1]
            self.report_hit(*adapted_best_emb)
            adapted_best_emb = [emb.detach().cpu().numpy() for emb in adapted_best_emb]
            write_pickle(adapted_best_emb, cfg.save_adapt_best_embeds_path)
            self.rename_log('/mrr{:.2f}_{}_'.format(mrr_best_adapt * 100, self.adapt).join(
                cfg.log.split('/')
            ))
        else:
            emb_s, emb_t = read_pickle(self.save_best_embeds_path)
            self.report_hit(emb_s, emb_t)
            self.rename_log('/mrr{:.2f}_{}_'.format(mrr_best * 100, self.adapt).join(
                cfg.log.split('/')
            ))

        return self.S

    def get_embeds(self, is_map=False):
        # get node embedding of two networks
        if is_map:
            source_after_mapping = self.mapping_model(self.source_embedding)
        else:
            source_after_mapping = self.source_embedding
        return source_after_mapping, self.target_embedding

    def check_edge_in_edges(self, edge, edges):
        for e in edges:
            if np.array_equal(edge, e):
                return True
        return False

    @staticmethod
    def extend_edges(source_edges, target_edges, links, undirected=True):
        es_s = set(source_edges)
        es_t = set(target_edges)
        s2t = {u: v for u, v in links}
        t2s = {v: u for u, v in links}
        es_s2t = {(s2t.get(u), s2t.get(v)) for u, v in es_s
                  if s2t.get(u) is not None and s2t.get(v) is not None}
        es_s2t = es_s2t | {(e[1], e[0]) for e in es_s2t}  # symmetry
        es_t2s = {(t2s.get(u), t2s.get(v)) for u, v in es_t
                  if t2s.get(u) is not None and t2s.get(v) is not None}
        es_t2s = es_t2s | {(e[1], e[0]) for e in es_t2s}
        es_s = list(es_s | es_t2s)
        es_t = list(es_t | es_s2t)
        return es_s, es_t

    def learn_embeddings(self):
        source_edges = self.graph_s.edges()
        target_edges = self.graph_t.edges()
        source_edges, target_edges = self.extend_edges(source_edges, target_edges, self.train_anchor_links)
        print("Done extend edges")

        self.source_embedding = self.learn_embedding(self.graph_s, np.array(source_edges)) #, 's')
        self.target_embedding = self.learn_embedding(self.graph_t, np.array(target_edges)) #, 't')

    def learn_embedding(self, G, edges):
        num_nodes = len(G.nodes())
        degree = np.array(G.degree())[:, -1]

        embedding_model = PaleEmbedding(
                                        n_nodes = num_nodes,
                                        embedding_dim = self.embedding_dim,
                                        deg= degree,
                                        neg_sample_size = self.neg_sample_size,
                                        device=self.device,
                                        ).to(self.device)

        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, embedding_model.parameters()), lr=self.emb_lr)
        embedding = self.train_embedding(embedding_model, edges, optimizer)

        return embedding

    def train_embedding(self, embedding_model, edges, optimizer):
        n_iters = len(edges) // self.emb_batchsize
        assert n_iters > 0, "batch_size is too large!"
        if len(edges) % self.emb_batchsize > 0:
            n_iters += 1
        print_every = int(n_iters/4) + 1
        total_steps = 0
        n_epochs = self.emb_epochs
        for epoch in range(1, n_epochs + 1):
            # for time evaluate
            start = time.time()

            print("Epoch {0}".format(epoch))
            np.random.shuffle(edges)
            for iter in range(n_iters):
                batch_edges = torch.LongTensor(edges[iter*self.emb_batchsize:(iter+1)*self.emb_batchsize]).to(self.device)
                start_time = time.time()
                optimizer.zero_grad()
                loss, loss0, loss1 = embedding_model.loss(batch_edges[:, 0], batch_edges[:,1])
                loss.backward()
                optimizer.step()
                if total_steps % print_every == 0:
                    print("Iter: %03d" %iter,
                              "train_loss={:.5f}".format(loss.item()),
                              "true_loss={:.5f}".format(loss0.item()),
                              "neg_loss={:.5f}".format(loss1.item()),
                              "time={:.5f}".format(time.time()-start_time)
                          )
                total_steps += 1
            
            # for time evaluate
            self.embedding_epoch_time = time.time() - start
            
        embedding = embedding_model.get_embedding()
        # embedding = embedding.cpu().detach().numpy()
        # embedding = torch.FloatTensor(embedding).to(self.device)

        return embedding

    def fast_adapt(self, task, learner, adaptation_steps=1):
        support_links, query_links = task
        # support_links = query_links  # tr-tr
        if self.adapt:
            support_links = list(filter(None, support_links))
            # Adapt the model
            if len(support_links) > 0:
                for step in range(adaptation_steps):
                    train_loss = learner.module.loss(support_links)
                    learner.adapt(train_loss)

        # Evaluate the adapted model
        valid_loss = learner.module.loss(query_links)

        return valid_loss

    def hit_test_adapt(self, test_batches, base_mdoel, adaptation_steps=2):
        # emb_s, emb_t = self.get_embeds()
        mrr_list, hit_list, success_list = [], [], []
        hit_list_class = []
        test_losses = []
        cnt = 0
        test_tasks = test_batches[0]
        # test_tasks = [test_task.extend(b) for b in test_batches]
        emb_s, emb_t = self.get_embeds(is_map=True)

        for i in trange(len(test_tasks)):
            task = test_tasks[i]
            learner = base_mdoel.clone()
            support_link, query_link = task

            support_link = list(filter(None, support_link))
            if len(support_link) > 0:
                for step in range(adaptation_steps):
                    train_loss = learner.module.loss(support_link)
                    learner.adapt(train_loss)

            # Evaluate the adapted model
            with torch.no_grad():
                source_after_mapping = learner.module(self.source_embedding)
                test_loss = learner.module.loss(query_link)
            test_losses.append(test_loss.item())
            test_num = len(query_link)
            cnt += test_num

            test_sid, test_tid = [list(i) for i in list(zip(*query_link))]
            emb_s[test_sid] = source_after_mapping[test_sid]

            # mrr, hit_p, success_p = self.calculate_hit(source_after_mapping, self.target_embedding,
            #                                      train_link=self.train_anchor_links, test_link=query_link,
            #                                      top_k=top_k)
            # mrr_list.append(mrr.item() * test_num)
            # hit_list.append(hit_p.item() * test_num)
            # success_list.append(success_p.item() * test_num)
            # hit_list_class.append(hit_p.item())

            # plt.bar(list(range(len(hit_list_class))), hit_list_class)
            # plt.show()

        mrr, hit_p, success_p = self.eval_metrics_na(emb_s, emb_t, self.k)
        test_loss = np.mean(test_losses)

        # test_loss, mrr, hit_p, success_p = np.mean(test_losses), np.sum(mrr_list) / cnt, np.sum(hit_list) / cnt, np.sum(success_list) / cnt
        self.log.info('-- Test loss: {:.4f}, MRR: {:.4f}, Hit: {:.4f}, Success:{:.4f}  --'.format(
            test_loss, mrr, hit_p, success_p
        ))

        return test_loss, mrr, hit_p, success_p, (emb_s, emb_t)
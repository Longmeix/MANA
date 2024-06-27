import random

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances
from scipy.spatial.distance import cdist
from itertools import chain
import networkx as nx
from sklearn.utils import shuffle


class MetaTask:
    def __init__(self, test_links, observed_links, n_way=5, k_shot=1, q_query=2):
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_sample = k_shot + q_query
        if type(observed_links) == np.ndarray:
            observed_links = observed_links.tolist()
            test_links = test_links.tolist()
        assert type(observed_links) == list
        self.test_anchor_s, self.test_anchors_t = [list(a) for a in zip(*test_links)]
        self.observed_anchors_s, self.observed_anchors_t = [list(a) for a in zip(*observed_links)]
        self.test_links_dict = dict(zip(self.test_anchor_s, test_links))
        self.observed_links_dict = dict(zip(self.observed_anchors_s, observed_links))
        self.meta_data_df = pd.DataFrame()

    @staticmethod
    def _generate_meta_data(*args, **kwargs):
        return pd.DataFrame()

    def _generate_tasks(self, tasks_data):
        # tasks_data = self._generate_meta_data()
        n_way = self.n_way
        # generate tasks [support set, query set]
        tasks = []
        task_num = len(tasks_data) // n_way + 1

        for i in range(task_num):
            task_data = tasks_data.sample(n=n_way)
            support = list(chain(*list(task_data['support_data'].values)))
            query = list(chain(*list(task_data['query_data'].values)))
            tasks.append([support, query])

        return tasks

    def _meta_batches(self, tasks_data, meta_bsz=8):
        # tasks = self.generate_task()
        tasks_list = self._generate_tasks(tasks_data)
        tasks_df = pd.DataFrame(tasks_list)

        # generate meta batches
        task_num = len(tasks_list)
        batches_num = task_num // meta_bsz
        if task_num % meta_bsz:
            batches_num += 1
        task_batches = []
        # for i in range(0, batches_num):
        #     task_idx = np.random.randint(low=0, high=len(tasks_df), size=meta_bsz)
        #     # task_batch = tasks[i*meta_bsz : (i+1) * meta_bsz]
        #     task_batch = tasks_df.iloc[task_idx].values
        #     task_batches.append(task_batch)

        tasks_df = shuffle(tasks_df)
        for i in range(0, batches_num):
            task_batch = tasks_df.iloc[i * meta_bsz: (i + 1) * meta_bsz].values
            task_batches.append(task_batch)

        return task_batches

    # def _meta_batches1(self, tasks_data, meta_bsz=8):
    #     # tasks = self.generate_task()
    #     tasks_list = self._generate_tasks(tasks_data)
    #     tasks_df = pd.DataFrame(tasks_list)
    #
    #     # generate meta batches
    #     task_num = len(tasks_list)
    #     batches_num = task_num // meta_bsz
    #     task_batches = []
    #
    #     tasks_df = shuffle(tasks_df)
    #     for i in range(0, batches_num):
    #         task_batch = tasks_df.iloc[i*meta_bsz : (i+1) * meta_bsz].values
    #         task_batches.append(task_batch)
    #
    #     return task_batches

    def get_meta_batches(self, *args):
        tasks_data = self._generate_meta_data()
        return self._meta_batches(tasks_data)

    def get_test_batches(self, *args):
        tasks_data = self._generate_meta_data()
        return [self._generate_tasks(tasks_data)]  # all tasks in one batch


class MetaTaskSimilarity(MetaTask):
    def __init__(self, test_links, observed_links, n_way=5, k_shot=1, q_query=1):
        super(MetaTaskSimilarity, self).__init__(test_links, observed_links, n_way, k_shot, q_query)

    @staticmethod
    def find_k_nearest_neighbors(emb_1, emb_2, node_1, node_2, metric='cosine', topk=10):
        emb_node_1 = emb_1[node_1]
        emb_node_2 = emb_2[node_2]
        sim_self_mat = sim(emb_node_1, emb_node_2, metric=metric, normalize=True)
        # sim_self_mat = sim(emb_node_1, emb_node_2, 'cosine', normalize=True, csls_k=5)
        # r, c = np.where(sim_self_mat == 1)
        # sim_self_mat[r, c] = 0.
        if node_1[:10] == node_2[:10]: # for meta-train sets
            np.fill_diagonal(sim_self_mat, 0)  # avoiding to select the query node itself
        node2_loc2idx = pd.Series(node_2, index=range(len(node_2)))

        nearest_loc = np.argpartition(sim_self_mat, -topk, axis=-1)[:, -topk:].flatten()  # top-k nearest nodes
        # nearest_loc = np.argpartition(sim_self_mat, -10, axis=-1)[:, -topk:].flatten()  # top-k nearest nodes
        # near_avg_sim for sorting tasks
        near_sim = np.partition(sim_self_mat, -topk, axis=-1)[:, -topk:]
        near_avg_sim = np.mean(near_sim, axis=-1)
        nearest_idx_mat = node2_loc2idx[nearest_loc].values.reshape(len(node_1), -1)

        return nearest_idx_mat, near_avg_sim

    @staticmethod
    def get_links(links_dict, s_nodes):
        links = []
        for src in s_nodes:
            tgt = links_dict.get(src)
            links.append((src, tgt))
        return links

    def _generate_meta_data(self, embed_s, sim_metric='cosine'):
        # embed_s = self.embed_s
        test_nodes = self.test_anchor_s
        observed_nodes = self.observed_anchors_s
        # self.test_anchor_s, self.test_anchors_t = [list(a) for a in zip(*test_links)]
        # self.observed_anchors_s, self.observed_anchors_t = [list(a) for a in zip(*observed_links)]
        # find top-k nearest nodes as support set for fine-tuning
        topk_similar_ids, near_avg_sim = self.find_k_nearest_neighbors(embed_s, embed_s, test_nodes, observed_nodes, sim_metric, topk=self.k_shot)

        # get_links_dict = self.get_links_dict

        meta_data_df = pd.DataFrame({
            'test_node_s': test_nodes,
            # 'anchor_link': self.get_links(self.links_dict, anchor_s),
            'near_avg_sim': near_avg_sim,
            'topk_similar_ids': topk_similar_ids.tolist(),
            'query_data': [[self.test_links_dict.get(s)] for s in test_nodes]
        })

        # anchor_df.sort_values(by='near_avg_sim', ascending=False, inplace=True)
        meta_data_df['support_data'] = meta_data_df['topk_similar_ids'].apply(
            lambda ids: [self.observed_links_dict.get(id) for id in ids])

        return meta_data_df

    def get_meta_batches(self, embed_s, meta_bsz=8, sim_metric='cosine'):
        tasks_data = self._generate_meta_data(embed_s, sim_metric)
        return self._meta_batches(tasks_data, meta_bsz)

    def get_test_batches(self, embed_s, sim_metric='cosine'):
        def generate_meta_test_task(embed_s, tasks_data, n_per_cluster):
            from sklearn.cluster import KMeans
            tasks = []
            task_num = len(tasks_data) // n_per_cluster  # number of cluster
            test_embed_s = embed_s[tasks_data.test_node_s]
            # sim_test_nodes = sim(test_embed_s, test_embed_s, metric=sim_metric, normalize=True)

            kmeans = KMeans(n_clusters=task_num, random_state=2023)
            kmeans.fit(test_embed_s)
            # 获得每个节点的聚类标签
            cluster_labels = kmeans.labels_
            tasks_data['cluster'] = cluster_labels

            from functools import reduce
            def merge_list(x):
                return list(reduce(lambda a, b: a + b, x))

            grouped_tasks_df = tasks_data.groupby('cluster').agg(
                {'test_node_s': 'first', 'query_data': merge_list, 'support_data': merge_list
                 }).reset_index()

            for i in range(task_num):
                task_data = grouped_tasks_df.loc[i]
                support = task_data['support_data']
                query = task_data['query_data']
                tasks.append([support, query])

            return tasks
        tasks_data = self._generate_meta_data(embed_s, sim_metric)
        return [generate_meta_test_task(embed_s, tasks_data, n_per_cluster=self.n_way)]  # all tasks in one batch


class MetaTaskNeighbor(MetaTask):
    def __init__(self, adj, test_links, observed_links, n_way=5, k_shot=1, q_query=2):
        super(MetaTaskNeighbor, self).__init__(test_links, observed_links, n_way, k_shot, q_query)
        self.adj = adj
        self.G = nx.from_numpy_matrix(adj, create_using=nx.Graph())

    def neighbors_as_support(self):
        anchor, kshot = self.test_anchor_s, self.k_shot
        G = self.G
        select_neighbors_list = []
        for s in anchor:
            neighbors = list(G.neighbors(s))
            while len(neighbors) < kshot:
                n_1_sel = random.sample(neighbors, 1)
                neighbor_2 = list(G.neighbors(n_1_sel[0]))
                neighbors.extend(neighbor_2)
            sel_neighbor = random.sample(neighbors, kshot)
            select_neighbors_list.append(sel_neighbor)
        return np.array(select_neighbors_list)

    def _generate_meta_data(self):
        anchors_s = self.test_anchor_s
        support_neighbors = self.neighbors_as_support()
        # anchors_topk_mat = np.hstack([topk_similar_ids, anchor_s.reshape([-1, 1])])  # [num_anchors, k_shot]

        anchor_df = pd.DataFrame({
            'anchor_s': anchors_s,
            # 'anchor_link': self.get_links(self.links_dict, anchor_s),
            'support_neighbors': support_neighbors.tolist(),
            'query_data': [[self.test_links_dict.get(s)] for s in anchors_s]
        })

        anchor_df['support_data'] = anchor_df['support_neighbors'].apply(
            lambda ids: [self.observed_links_dict.get(id) for id in ids])

        # anchor_df.sort_values(by='near_avg_sim', ascending=False, inplace=True)

        # # generate tasks [support set, query set]
        # tasks = []
        # task_num = len(anchors_s) // n_way + 1
        # # for i in range(task_num):
        # #     loc = np.random.randint(low=0, high=len(anchor_s), size=n_way)
        # #     task_data = anchors_topk_mat[loc]
        # #     support_data = self.get_links(self.links_dict, task_data[:, :k_shot].flatten())
        # #     query_data = self.get_links(self.links_dict, task_data[:, k_shot:].flatten())
        # #     task_data = [support_data, query_data]
        # #     tasks.append(task_data)
        #
        # for i in range(task_num):
        #     # task_data = anchor_df[i*n_way : (i+1)*n_way]
        #     task_data = anchor_df.sample(n=n_way)
        #     # support = [i for i in list(task_data['support_data'].values)]
        #     # query = [i for i in list(task_data['query_data'].values)]
        #     support = list(chain(*list(task_data['support_data'].values)))
        #     query = list(chain(*list(task_data['query_data'].values)))
        #     tasks.append([support, query])
        # # last_cursor = task_num * n_way
        # # if last_cursor < len(anchor_s):
        # #     task_data = anchor_df[last_cursor: ]
        # #     support = list(chain(*list(task_data['support_data'].values)))
        # #     query = list(chain(*list(task_data['query_data'].values)))
        # #     tasks.append([support, query])

        return anchor_df

    def get_meta_batches(self, meta_bsz=4):
        tasks_data = self._generate_meta_data()
        return self._meta_batches(tasks_data, meta_bsz)


class MetaTaskRandom(MetaTask):
    def __init__(self, test_links, observed_links, n_way=5, k_shot=1, q_query=2):
        super(MetaTaskRandom, self).__init__(test_links, observed_links, n_way, k_shot, q_query)


    def _generate_meta_data(self):
        anchors_s = self.test_anchor_s
        observed_anchors_list = list(self.observed_links_dict.values())
        # anchors_topk_mat = np.hstack([topk_similar_ids, anchor_s.reshape([-1, 1])])  # [num_anchors, k_shot]

        anchor_df = pd.DataFrame({
            'anchor_s': anchors_s,
            'query_data': [[self.test_links_dict.get(s)] for s in anchors_s]
        })

        anchor_df['support_data'] = anchor_df['anchor_s'].apply(
            lambda id: random.sample(observed_anchors_list, self.k_shot))

        return anchor_df

    def get_meta_batches(self, meta_bsz=4):
        tasks_data = self._generate_meta_data()
        return self._meta_batches(tasks_data, meta_bsz)


class MetaTestTask(MetaTaskSimilarity):
    def __init__(self, embed_s, test_links, observed_links, n_way, k_shot, q_query):
        super(MetaTestTask, self).__init__(embed_s, test_links, observed_links, n_way, k_shot, q_query)

        assert q_query == 1


def sim(embed1, embed2, metric='inner', normalize=False, csls_k=0):
    """
    Compute pairwise similarity between the two collections of embeddings.

    Parameters
    ----------
    embed1 : matrix_like
        An embedding matrix of size n1*d, where n1 is the number of embeddings and d is the dimension.
    embed2 : matrix_like
        An embedding matrix of size n2*d, where n2 is the number of embeddings and d is the dimension.
    metric : str, optional, inner default.
        The distance metric to use. It can be 'cosine', 'euclidean', 'inner'.
    normalize : bool, optional, default false.
        Whether to normalize the input embeddings.
    csls_k : int, optional, 0 by default.
        K value for csls. If k > 0, enhance the similarity by csls.

    Returns
    -------
    sim_mat : An similarity matrix of size n1*n2.
    """
    if normalize:
        embed1 = preprocessing.normalize(embed1)
        embed2 = preprocessing.normalize(embed2)
    if metric == 'inner':
        sim_mat = np.matmul(embed1, embed2.T)  # numpy.ndarray, float32
    elif metric == 'cosine' and normalize:
        sim_mat = np.matmul(embed1, embed2.T)  # numpy.ndarray, float32
    elif metric == 'euclidean':
        sim_mat = 1 - euclidean_distances(embed1, embed2)
        print(type(sim_mat), sim_mat.dtype)
        sim_mat = sim_mat.astype(np.float32)
    elif metric == 'CSLS':
        sim_mat = np.matmul(embed1, embed2.T)
        sim_mat = csls_sim(sim_mat, csls_k)
    elif metric == 'cosine':
        sim_mat = 1 - cdist(embed1, embed2, metric='cosine')  # numpy.ndarray, float64
        sim_mat = sim_mat.astype(np.float32)
    elif metric == 'manhattan':
        sim_mat = 1 - cdist(embed1, embed2, metric='cityblock')
        sim_mat = sim_mat.astype(np.float32)
    else:
        sim_mat = 1 - cdist(embed1, embed2, metric=metric)
        sim_mat = sim_mat.astype(np.float32)
    # if csls_k > 0:
    #     sim_mat = csls_sim(sim_mat, csls_k)
    return sim_mat

def csls_sim(sim_mat, k):
    """
    Compute pairwise csls similarity based on the input similarity matrix.

    Parameters
    ----------
    sim_mat : matrix-like
        A pairwise similarity matrix.
    k : int
        The number of nearest neighbors.

    Returns
    -------
    csls_sim_mat : A csls similarity matrix of n1*n2.
    """

    nearest_values1 = calculate_nearest_k(sim_mat, k)
    nearest_values2 = calculate_nearest_k(sim_mat.T, k)
    csls_sim_mat = 2 * sim_mat.T - nearest_values1
    csls_sim_mat = csls_sim_mat.T - nearest_values2
    return csls_sim_mat


def calculate_nearest_k(sim_mat, k):
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1)  # -np.sort(-sim_mat1)
    nearest_k = sorted_mat[:, 0:k]
    return np.mean(nearest_k, axis=1)
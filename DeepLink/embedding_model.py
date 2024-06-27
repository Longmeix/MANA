import random

from gensim.models import Word2Vec
import numpy as np


class DeepWalk:
    def __init__(self, G, embed_dim=128, num_walks=10, walk_len=10, window_size=5, \
                 num_cores=8, num_epochs=50):
        """
        Parameters
        ----------
        G: networkx Graph
            Graph
        id2idx: dictionary
            dictionary of keys are ids of nodes and values are index of nodes
        num_walks: int
            number of walks per node
        walk_len: int
            length of each walk
        windows_size: int
            size of windows in skip gram model
        embedding_dim: int
            number of embedding dimensions
        num_cores: int
            number of core when train embedding
        num_epochs: int
            number of epochs in embedding
        """
        # np.random.seed(seed)

        self.G = G
        self.num_walks = num_walks
        self.walk_len = walk_len
        self.window_size = window_size
        # self.id2idx = id2idx
        self.embed_dim = embed_dim
        self.num_cores = num_cores
        self.num_epochs = num_epochs


    def get_embedding(self):
        walks = self.simulate_walks()
        embedding_model = Word2Vec(walks, size=self.embed_dim, window=self.window_size,\
                            min_count=0, negative=5, sg=1, hs=1, workers=self.num_cores, iter=self.num_epochs)
        # embedding = np.zeros((len(self.G.nodes()), self.embed_dim))
        # for i in range(len(self.G.nodes())):
        #     embedding[i] = embedding_model.wv[str(i)]
        embedding = np.array(list(map(embedding_model.wv.get_vector,
                                      map(str, range(self.G.number_of_nodes())))))
        return embedding


    def simulate_walks(self):
        print("Random walk process")
        walks = []
        nodes = list(self.G.nodes)
        for walk_iter in range(self.num_walks):
            # print(str(walk_iter + 1), '/', str(self.num_walks))
            random.shuffle(nodes)
            for node in nodes:
                # walk = [str(self.id2idx[node])]
                walk = [str(node)]
                if self.G.degree(node) == 0:
                    continue
                curr_node = node
                while len(walk) < self.walk_len:
                    next_node = np.random.choice(list(self.G.neighbors(curr_node)))
                    curr_node = next_node
                    if curr_node != node:
                        walk.append(str(curr_node))
                walks.append(walk)
        print("Done walks for", len(nodes), "nodes")
        return walks

    # https: // github.com / thunlp / OpenNE / blob / d9cbf34aff87c9d09fa58a074907ed40a0e06146 / src / openne / walker.py  # L11
    def deepwalk_walk(self, walk_length, start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        G = self.G
        cur = start_node
        walk = [str(start_node)]

        while len(walk) < walk_length:
            cur_nbrs = list(G.neighbors(cur))
            if len(cur_nbrs) > 0:
                next_node = random.choice(cur_nbrs)
                walk.append(str(next_node))
                cur = next_node
            else:
                break
        return walk

    def simulate_walks(self):
        '''
        Repeatedly simulate random walks from each node.
        '''
        G = self.G
        num_walks = self.num_walks
        walk_length = self.walk_len
        walks = []
        nodes = list(G.nodes())
        print('Walk iteration:')
        for walk_iter in range(num_walks):
            # pool = multiprocessing.Pool(processes = 4)
            # print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                # walks.append(pool.apply_async(deepwalk_walk_wrapper, (self, walk_length, node, )))
                walks.append(self.deepwalk_walk(
                    walk_length=walk_length, start_node=node))
            # pool.close()
            # pool.join()
        # print(len(walks))
        return walks

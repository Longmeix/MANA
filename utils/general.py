import pickle
from torch.utils.data import Dataset
import torch


def write_pickle(obj, outfile, protocol=-1):
    with open(outfile, 'wb') as f:
        pickle.dump(obj, f, protocol=protocol)


def read_pickle(infile):
    with open(infile, 'rb') as f:
        return pickle.load(f)

def str2bool(str):
    return True if str.lower() == 'true' else False


class UILDataset(Dataset):
    def __init__(self, links):
        self.links = links

    def __len__(self):
        return len(self.links)

    def __getitem__(self, item):
        return self.links[item]


class PairDataset(Dataset):
    def __init__(self, pairs, mode='train'):
        self.pairs = pairs
        size = len(self.pairs)
        self.labels = torch.ones(size)
        if mode == 'test':
            self.labels = torch.zeros(size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, item):
        pair = self.pairs[item]
        label = self.labels[item]
        return pair, label


if __name__ == '__main__':
    pass
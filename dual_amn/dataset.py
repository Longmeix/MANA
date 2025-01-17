from utils import *
import os
import os.path as osp
from random import shuffle
# import codecs
from dto import *


class EAData:
    def __init__(self, triple1_path, triple2_path, ent_links_path,
                 shuffle_pairs=False, train_ratio=0.5, unsup=False, filter_link=True, **kwargs):
        rel1, ent1, triple1 = self.process_one_graph(triple1_path)
        rel2, ent2, triple2 = self.process_one_graph(triple2_path)
        self.unsup = unsup
        if self.unsup:
            print('use unsupervised mode')
        self.rel1, self.ent1, self.triple1 = rel1, ent1, triple1
        self.rel2, self.ent2, self.triple2 = rel2, ent2, triple2
        self.link = self.process_link(ent_links_path, ent1, ent2, filter_link)
        self.rels = [rel1, rel2]
        self.ents = [ent1, ent2]
        self.triples = [triple1, triple2]
        self.train_cnt = 0 if unsup else int(train_ratio * len(self.link))
        if shuffle_pairs:
            shuffle(self.link)

        for k, v in kwargs.items():
            setattr(self, k, v)

    @staticmethod
    def load(path):
        return readobj(path)

    def save(self, path):
        saveobj(self, path)

    def get_train(self):
        if self.unsup:
            if hasattr(self, 'semi_link'):
                return self.semi_link
            else:
                raise RuntimeError('No unsupervised pairs!!')
        now = np.array(self.link[:self.train_cnt])
        if hasattr(self, 'semi_link') and self.semi_link is not None:
            return np.concatenate([now, self.semi_link], axis=0)
        return now

    def set_train(self, link):
        self.semi_link = link

    train = property(get_train, set_train)

    @property
    def test(self):
        return self.link[self.train_cnt:]

    def save_eakit_format(self, path):
        lg1e = len(self.ent1)
        lg1r = len(self.rel1)

        new_g2e = {k: v + lg1e for k, v in self.ent2.items()}
        new_g2r = {k: v + lg1r for k, v in self.rel2.items()}
        new_g2t = [(h + lg1e, r + lg1r, t + lg1e) for h, r, t in self.triple1]
        new_pair = [(e1, e2 + lg1e) for e1, e2 in self.link]
        ents = [self.ent1, new_g2e]
        rels = [self.rel1, new_g2r]
        triples = [self.triple1, new_g2t]
        for i in range(1, 3):
            save_map(ents[i - 1], osp.join(path, 'ent_ids_{}'.format(i)),
                     reverse_kv=True, sort_by_key=True)
            save_map(rels[i - 1], osp.join(path, 'rel_ids_{}'.format(i)),
                     reverse_kv=True, sort_by_key=True)
            make_file(osp.join(path, 'training_attrs_{}'.format(i)))
            save_array(triples[i - 1], osp.join(path, 'triples_{}'.format(i)))
        # raise NotImplementedError
        save_array(new_pair, osp.join(path, 'ill_ent_ids'), sort_by=0)

    def save_openea_format(self, path):
        pass

    @staticmethod
    def process_one_graph(rel_pos: str):
        print('load relation file:', rel_pos)
        triples, rel_idx, ent_idx = [], {}, {}
        with codecs.open(rel_pos, "r", 'utf-8') as f:
            for line in f.readlines():
                now = line.strip().split('\t')
                ent_idx, s = add_cnt_for(ent_idx, now[0])
                rel_idx, p = add_cnt_for(rel_idx, now[1])
                ent_idx, o = add_cnt_for(ent_idx, now[2])
                triples.append([s, p, o])
        return rel_idx, ent_idx, triples

    @staticmethod
    def process_link(links_list, ent1, ent2, filter_link=True):
        link = []
        link1 = set()
        link2 = set()
        if not isinstance(links_list, tuple):
            links_list = (links_list,)
        for links_pos in links_list:
            print('load ill file:', links_pos)
            with codecs.open(links_pos, "r", 'utf-8') as f:
                for line in f.readlines():
                    now = line.strip().split('\t')
                    if (now[0] in ent1 and now[1] in ent2) or (not filter_link):
                        ent1, src = add_cnt_for(ent1, now[0])
                        ent2, trg = add_cnt_for(ent2, now[1])

                        if src in link1 or trg in link2:
                            continue
                        link1.add(src)
                        link2.add(trg)
                        link.append((src, trg))
        return link

    @staticmethod
    def ill(pairs, device='cuda'):
        return torch.tensor(pairs, dtype=torch.long, device=device).t()

    def get_pairs(self, device='cuda'):
        return self.ill(self.link, device)

    def size(self, which=None):
        if which is None:
            return [self.size(0), self.size(1)]

        return len(self.ents[which])

    def __repr__(self):
        return argprint(
            triple1=len(self.triple1),
            triple2=len(self.triple2),
            ent1=len(self.ent1),
            ent2=len(self.ent2),
            rel1=len(self.rel1),
            rel2=len(self.rel2),
            link=len(self.link)
        )


class OpenEAData(EAData):
    def __init__(self, path, **kwargs):
        try:
            super().__init__(osp.join(path, 'rel_triples_1'),
                             osp.join(path, 'rel_triples_2'),
                             osp.join(path, 'ent_links'), **kwargs)
        except FileNotFoundError:
            super().__init__(osp.join(path, 'triples_1'),
                             osp.join(path, 'triples_2'),
                             osp.join(path, 'ent_links'), **kwargs)


class LargeScaleEAData(EAData):
    def __init__(self, path, lang, strict=False, shuffle_pairs=False, train_ratio=0.3, **kwargs):
        super().__init__(*(osp.join(path, '{0}_triples{1}_{2}'.format(lang, ['', '_strict'][strict], i))
                           for i in range(1, 3)),
                         osp.join(path, '{}_ent_links'.format(lang)),
                         shuffle_pairs=shuffle_pairs, train_ratio=train_ratio)


class DBPData(EAData):
    def __init__(self, path, **kwargs):
        try:
            super().__init__(osp.join(path, 'triples_1'),
                             osp.join(path, 'triples_2'),
                             (osp.join(path, 'sup_ent_ids'),
                              osp.join(path, 'ref_ent_ids'),), **kwargs)
        except FileNotFoundError:

            super().__init__(osp.join(path, 'triples_1'),
                             osp.join(path, 'triples_2'),
                             osp.join(path, 'ref_ent_ids'), **kwargs)

class NAData:
    def __init__(self, path, train_ratio, **kwargs):
        triple1_path = osp.join(path, 'rel_triples_1')
        triple2_path = osp.join(path, 'rel_triples_2')

        rel1, ent1, triple1 = self.process_one_graph(triple1_path)
        rel2, ent2, triple2 = self.process_one_graph(triple2_path)
        self.rel1, self.ent1, self.triple1 = rel1, ent1, triple1
        self.rel2, self.ent2, self.triple2 = rel2, ent2, triple2

        train_links = self.read_links(path + f'train_ratio_{train_ratio}/' + 'train_links')
        test_links = self.read_links(path + f'train_ratio_{train_ratio}/' + 'test_links')

        assert len(train_links) > 0 and len(test_links) > 0

        self.train = self.uris_pair_2ids(train_links, ent1, ent2)
        self.test = self.uris_pair_2ids(test_links, ent1, ent2)

        # self.link = self.process_link(ent_links_path, ent1, ent2, filter_link)
        self.rels = [rel1, rel2]
        self.ents = [ent1, ent2]
        self.triples = [triple1, triple2]
        # self.train_cnt = 0 if unsup else int(train_ratio * len(self.link))
        # if shuffle_pairs:
        #     shuffle(self.link)

    @staticmethod
    def process_one_graph(rel_pos: str):
        print('load relation file:', rel_pos)
        triples, rel_idx, ent_idx = [], {}, {}
        with codecs.open(rel_pos, "r", 'utf-8') as f:
            for line in f.readlines():
                now = line.strip().split('\t')
                ent_idx, s = add_cnt_for(ent_idx, now[0])
                rel_idx, p = add_cnt_for(rel_idx, now[1])
                ent_idx, o = add_cnt_for(ent_idx, now[2])
                triples.append([s, p, o])
        return rel_idx, ent_idx, triples

    @staticmethod
    def read_links(file_path):
        print("read links:", file_path)
        links = list()
        refs = list()
        reft = list()
        file = open(file_path, 'r', encoding='utf8')
        for line in file.readlines():
            params = line.strip('\n').split('\t')
            assert len(params) == 2
            e1 = params[0].strip()
            e2 = params[1].strip()
            refs.append(e1)
            reft.append(e2)
            links.append((e1, e2))
        assert len(refs) == len(reft)
        return links


    @staticmethod
    def uris_pair_2ids(uris, ids1, ids2):
        id_uris = list()
        for u1, u2 in uris:
            # assert u1 in ids1
            # assert u2 in ids2
            if u1 in ids1 and u2 in ids2:
                id_uris.append((ids1[u1], ids2[u2]))
        # assert len(id_uris) == len(set(uris))
        return id_uris


def load_dataset(scale='small', ds='ids', dataset='FT', train_ratio=0.5, shuffle=False):
    if scale == 'small':
        if ds == 'ids':
            # ea = OpenEAData('../OpenEA_dataset_v2.0/EN_{}_15K_V1/'.format(lang.upper()), train_ratio=train_ratio,
            #                 shuffle_pairs=shuffle)
            ea = OpenEAData('../../dataset/OpenEA_dataset_v2.0/EN_FR_15K_V1/', train_ratio=train_ratio,
                            shuffle_pairs=shuffle)
            # ea = OpenEAData(f'../../dataset/{dataset}/EA_data/',
            #                 train_ratio=train_ratio,
            #                 shuffle_pairs=shuffle)
        elif ds == 'dbp':
            ea = DBPData('../DUAL2/data/{}_en/'.format(lang), train_ratio=train_ratio, shuffle_pairs=shuffle)
        elif ds == 'srp':
            ea = DBPData('../DUAL2/data/en_{}_15k_V1//'.format(lang), train_ratio=train_ratio, shuffle_pairs=shuffle)
        elif ds == 'na':
            ea = NAData(f'./dataset/{dataset}/EA_data/', train_ratio)
        else:
            raise NotImplementedError

    elif scale == 'medium':
        if ds == 'ids':
            ea = OpenEAData('../OpenEA_dataset_v2.0/EN_{}_100K_V1/'.format(lang.upper()), train_ratio=train_ratio,
                            shuffle_pairs=shuffle)
        elif ds == 'srp':
            ea = DBPData('../DUAL2/data/en_{}_100k_V1/'.format(lang), train_ratio=train_ratio, shuffle_pairs=shuffle)
        else:
            raise NotImplementedError
    else:
        ea = LargeScaleEAData('../mkdata/', lang=lang, train_ratio=train_ratio, shuffle_pairs=False)

    print(ea)
    return ea

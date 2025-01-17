import numpy as np
import scipy.sparse as sp


def normalize_adj(adj):
    print('normalize adj')
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return d_mat_inv_sqrt.dot(adj).transpose().dot(d_mat_inv_sqrt).T


def load_triples(file_name):
    triples = []
    entity = set()
    rel = set([0])
    for line in open(file_name, 'r'):
        head, r, tail = [int(item) for item in line.split()]
        entity.add(head);
        entity.add(tail);
        rel.add(r + 1)
        triples.append((head, r + 1, tail))
    return entity, rel, triples


def load_alignment_pair(file_name):
    alignment_pair = []
    c = 0
    for line in open(file_name, 'r'):
        e1, e2 = line.split()
        alignment_pair.append((int(e1), int(e2)))
    return alignment_pair


def get_matrix_faster(triples, ent_size, rel_size):
    from dto import readobj, saveobj
    tmp_file_name = f'tmp/triples_{len(triples)}_{ent_size}_{rel_size}'
    try:
        print('try load from cache:', tmp_file_name)
        objs = readobj(tmp_file_name)
        return objs
    except:
        print('load failed, build adj')
        import time
        tic = time.time()
        adj_matrix = sp.lil_matrix((ent_size, ent_size))
        adj_features = sp.lil_matrix((ent_size, ent_size))

        radj = []
        rel_in = sp.lil_matrix((ent_size, rel_size * 2))
        # rel_out = sp.lil_matrix((ent_size, rel_size))

        for i in range(ent_size):
            adj_features[i, i] = 1

        for h, r, t in triples:
            adj_matrix[h, t] = 1
            adj_matrix[t, h] = 1
            adj_features[h, t] = 1
            adj_features[t, h] = 1
            radj.append([(h, t), r])
            radj.append([(t, h), r + rel_size])
            rel_in[t, r] += 1
            rel_in[h, r + rel_size] += 1

        count = -1
        s = set()
        d = {}
        r_index, r_val = [], []
        for (h, t), r in sorted(radj, key=lambda x: x[0]):
            if (h, t) in s:
                r_index.append([count, r])
                r_val.append(1)
                d[count] += 1
            else:
                count += 1
                d[count] = 1
                s.add((h, t))
                r_index.append([count, r])
                r_val.append(1)
        for i in range(len(r_index)):
            r_val[i] /= d[r_index[i][0]]

        # rel_features = np.concatenate([rel_in, rel_out], axis=1)
        adj_features = normalize_adj(adj_features)
        rel_features = normalize_adj(rel_in)
        print('construct time:', time.time() - tic)
        saveobj((adj_matrix, r_index, r_val, adj_features, rel_features), tmp_file_name)
        return adj_matrix, r_index, r_val, adj_features, rel_features


def get_matrix(triples, ent_size, rel_size):
    print(ent_size, rel_size)
    adj_matrix = sp.lil_matrix((ent_size, ent_size))
    adj_features = sp.lil_matrix((ent_size, ent_size))
    radj = []
    rel_in = np.zeros((ent_size, rel_size))
    rel_out = np.zeros((ent_size, rel_size))

    for i in range(ent_size):
        adj_features[i, i] = 1

    for h, r, t in triples:
        adj_matrix[h, t] = 1
        adj_matrix[t, h] = 1
        adj_features[h, t] = 1
        adj_features[t, h] = 1
        radj.append([h, t, r])
        radj.append([t, h, r + rel_size])
        rel_out[h][r] += 1;
        rel_in[t][r] += 1

    count = -1
    s = set()
    d = {}
    r_index, r_val = [], []
    for h, t, r in sorted(radj, key=lambda x: x[0] * 10e10 + x[1] * 10e5):
        if ' '.join([str(h), str(t)]) in s:
            r_index.append([count, r])
            r_val.append(1)
            d[count] += 1
        else:
            count += 1
            d[count] = 1
            s.add(' '.join([str(h), str(t)]))
            r_index.append([count, r])
            r_val.append(1)
    for i in range(len(r_index)):
        r_val[i] /= d[r_index[i][0]]

    rel_features = np.concatenate([rel_in, rel_out], axis=1)
    adj_features = normalize_adj(adj_features)
    rel_features = normalize_adj(sp.lil_matrix(rel_features))
    return adj_matrix, r_index, r_val, adj_features, rel_features


def load_data(lang, train_ratio=0.3):
    entity1, rel1, triples1 = load_triples(lang + 'triples_1')
    entity2, rel2, triples2 = load_triples(lang + 'triples_2')
    # modified here #
    if "_en" in lang or True:
        alignment_pair = load_alignment_pair(lang + 'ref_ent_ids')
        np.random.shuffle(alignment_pair)
        train_pair, dev_pair = alignment_pair[0:int(len(alignment_pair) * train_ratio)], alignment_pair[int(
            len(alignment_pair) * train_ratio):]
    else:
        train_pair = load_alignment_pair(lang + 'sup_ent_ids')
        dev_pair = load_alignment_pair(lang + 'ref_ent_ids')
        ae_features = None

    adj_matrix, r_index, r_val, adj_features, rel_features = get_matrix(triples1 + triples2,
                                                                        len(entity1.union(entity2)),
                                                                        len(rel1.union(rel2)))

    return np.array(train_pair), np.array(dev_pair), adj_matrix, np.array(r_index), np.array(
        r_val), adj_features, rel_features

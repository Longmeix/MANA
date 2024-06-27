import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SGConv, GCNConv, GATConv
import config as cfg


class NetEncode(nn.Module):
    def __init__(self, in_dim=256, out_dim=64):
        super(NetEncode, self).__init__()
        hid_dim = 128
        # self.sgcs = nn.Sequential(
        #     SGConv(in_dim, hid_dim),
        #     # nn.ReLU(),
        #     SGConv(hid_dim, out_dim)
        # )
        self.sgc = SGConv(in_dim, out_dim, K=2, bias=False)
        # self.w_sgc = nn.Parameter(self.init_randn_uni(out_dim, out_dim))
        # self.lin = nn.Linear(out_dim, out_dim)
            # nn.Sequential(
            # nn.Linear(in_dim, out_dim),
            # nn.Linear(out_dim, out_dim)
        # )

    def forward(self, x, edge_index):
        x = self.sgc(x, edge_index)  # 不用自定义的参数比较好
        # x = self.lin(x)
        # x = F.relu(x)
        # x = F.tanh(x)
        # x = self.sgc(x, edge_index) @ self.w_sgc
        # for i in range(len(self.sgcs)):
        #     x = self.sgcs[i](x, edge_index)
        return x

    @staticmethod
    # 产生均匀分布的数
    def init_randn_uni(size, dim):
        emb = nn.Parameter(torch.randn(size, dim))
        emb.data = F.normalize(emb.data)
        return emb


class GNN(nn.Module):
    def __init__(self, in_dim=256, out_dim=64,
                 alpha=1, gcn_out=None,
                 gat_out=None, activator=None):
        super(GNN, self).__init__()
        if gcn_out is None:
            gcn_out = in_dim
        if gat_out is None:
            gat_out = in_dim
        self.gcn = GCNConv(in_dim, gcn_out)
        self.gat = GATConv(in_dim, gat_out)
        self.w_gcn = nn.Parameter(self.init_randn_uni(gcn_out, out_dim))  # 自动被添加到模型的参数列表中
        self.w_gat = nn.Parameter(self.init_randn_uni(gat_out, out_dim))
        self.alpha = alpha
        if activator is None:
            self.activator = nn.Tanh()  # 为什么用tanh
            # self.activator = lambda x: x
        else:
            self.activator = activator

    @staticmethod
    def init_randn_uni(size, dim):
        emb = nn.Parameter(torch.randn(size, dim))
        emb.data = F.normalize(emb.data)
        return emb

    def forward(self, x, edge_index):
        x_gcn = self.gcn(x, edge_index)
        x_gat = self.gat(x, edge_index)
        x = x_gcn @ self.w_gcn + self.alpha * x_gat @ self.w_gat  # 矩阵乘法 公式10
        return self.activator(x)

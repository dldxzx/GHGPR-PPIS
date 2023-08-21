import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

SEED = 2020
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    torch.cuda.manual_seed(SEED)


class NormLayer(nn.Module):
    def __init__(self, norm_mode, norm_scale):
        """
            mode:
              'None' : No normalization
              'PN'   : PairNorm
              'PN-SI'  : Scale-Individually version of PairNorm
              'PN-SCS' : Scale-and-Center-Simultaneously version of PairNorm
              'LN': LayerNorm
              'CN': ContraNorm
        """
        super(NormLayer, self).__init__()
        self.mode = norm_mode
        self.scale = norm_scale

    def forward(self, x, adj=None, tau=1.0):
        if self.mode == 'None':
            return x
        if self.mode == 'LN':
            x = x - x.mean(dim=1, keepdim=True)
            x = nn.functional.normalize(x, dim=1)

        col_mean = x.mean(dim=0)
        if self.mode == 'PN':
            x = x - col_mean
            rownorm_mean = (1e-6 + x.pow(2).sum(dim=1).mean()).sqrt()
            x = self.scale * x / rownorm_mean

        if self.mode == 'CN':
            norm_x = nn.functional.normalize(x, dim=1)
            sim = norm_x @ norm_x.T / tau
            # if adj.size(1) == 2:
            #     sim[adj[0], adj[1]] = -np.inf
            # else:
            sim.masked_fill_(adj > 1e-5, -np.inf)
            sim = nn.functional.softmax(sim, dim=1)
            x_neg = sim @ x
            x = (1 + self.scale) * x - self.scale * x_neg

        if self.mode == 'PN-SI':
            x = x - col_mean
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual

        if self.mode == 'PN-SCS':
            rownorm_individual = (1e-6 + x.pow(2).sum(dim=1, keepdim=True)).sqrt()
            x = self.scale * x / rownorm_individual - col_mean

        return x


class HeatLayer(nn.Module):
    def __init__(self, nfeats_in_dim, nfeats_out_dim, edge_dim=2):
        super(HeatLayer, self).__init__()
        self.nfeats_out_dim = nfeats_out_dim
        self.fc = nn.Linear(nfeats_in_dim, nfeats_out_dim, bias=False)

        self.attn_fc = nn.Linear(2 * nfeats_out_dim + edge_dim, 1, bias=False)
        self.fc_edge_for_att_calc = nn.Linear(edge_dim, edge_dim, bias=False)
        self.fc_eFeatsDim_to_nFeatsDim = nn.Linear(edge_dim, nfeats_out_dim, bias=False)

        self.attn_edge = nn.Parameter(torch.FloatTensor(size=(1, 1, nfeats_out_dim)))
        self.fc_edge = nn.Linear(edge_dim, nfeats_out_dim * 1, bias=False)

        self.weight = nn.Parameter(torch.Tensor(2, nfeats_in_dim, nfeats_out_dim))  # [K+1, 1, in_c, out_c]
        init.xavier_normal_(self.weight)

        self.bias = nn.Parameter(torch.Tensor(1, nfeats_out_dim))
        init.zeros_(self.bias)
        self.K = 2
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_fc.weight, gain=gain)

        nn.init.xavier_normal_(self.fc_edge_for_att_calc.weight, gain=gain)
        nn.init.xavier_normal_(self.fc_eFeatsDim_to_nFeatsDim.weight, gain=gain)

        nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_edge, gain=gain)


    def edge_attention(self, edges):

        h_edge = edges.data['ex']
        feat_edge = self.fc_edge(h_edge).view(-1, 1, self.nfeats_out_dim)
        ee = (feat_edge * self.attn_edge).sum(dim=-1)


        z2 = torch.cat([edges.src['z'], edges.dst['z'], edges.data['ex']], dim=1)
        a = self.attn_fc(z2)
        a = a + ee
        ez = self.fc_eFeatsDim_to_nFeatsDim(edges.data['ex'])

        return {'e': F.leaky_relu(a), 'ez': ez, }

    def message_func(self, edges):
        return {'z': edges.src['z'], 'e': edges.data['e'], 'ez': edges.data['ez']}

    def reduce_func(self, nodes):
        attn_w = F.softmax(nodes.mailbox['e'], dim=1)
        h = torch.sum(attn_w * nodes.mailbox['z'], dim=1)
        h = h + torch.sum(attn_w * nodes.mailbox['ez'], dim=1)

        return {'h': h}

    def forward(self, g, h, e, adj):

        z = self.fc(h)
        L = HeatLayer.get_laplacian(adj)
        N = adj.shape[0]
        w = self.heat_weight(L, N)
        result = torch.matmul(w, z)
        result = torch.matmul(result, self.weight)
        z = torch.sum(result, dim=0) + self.bias

        g.ndata['z'] = z
        ex = self.fc_edge_for_att_calc(e)
        g.edata['ex'] = ex
        g.apply_edges(self.edge_attention)
        g.update_all(self.message_func, self.reduce_func)

        return g.ndata.pop('h')

    def weight_wavelet(self, s, lamb, U, k, N):
        multi_order_laplacian = torch.zeros([self.K, N, N]).cuda()
        multi_order_laplacian[0] = torch.eye(N).cuda()
        for m in range(1, k):
            for i in range(len(lamb)):
                lamb[i] = math.pow(math.e, -lamb[i] * s * m)
            Weight = torch.mm(torch.mm(U, torch.diag(lamb)), torch.transpose(U, 0, 1))
            multi_order_laplacian[m] = Weight

        return multi_order_laplacian

    def sort(self, lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]

    def heat_weight(self, laplacian, N):
        lamb, U = torch.linalg.eigh(laplacian)
        lamb, U = self.sort(lamb, U)
        weight = self.weight_wavelet(2, lamb, U, self.K, N)
        return weight

    @staticmethod
    def get_laplacian(graph):
        """
        return the laplacian of the graph.

        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
        L = torch.eye(graph.size(0)).cuda() - torch.mm(torch.mm(D, graph), D)

        return L


class HESGAT(nn.Module):
    def __init__(self, in_features, out_features):
        super(HESGAT, self).__init__()
        self.agat = HeatLayer(in_features, out_features)
        self.act_fn = nn.ReLU()
        self.dropout = 0.1
        self.norm = NormLayer('CN', norm_scale=0.55)

    def forward(self, x, adj_sc, g, H0, adj_ca):
        # use sc coordinate
        x = x.float()
        out = self.agat(g, x, g.edata['ex'], adj_sc)
        out = F.dropout(out, self.dropout, training=self.training)
        out = self.act_fn(out)
        out = self.norm(0.7 * out + 0.3 * H0, adj_sc)
        return out


class GPR_HESGAT(nn.Module):
    '''
    propagation class for GPR_GNN
    '''

    def __init__(self, in_feature, out_feature, alpha, K):
        super(GPR_HESGAT, self).__init__()
        self.K = K
        self.alpha = alpha
        TEMP = alpha * (1 - alpha) ** np.arange(K + 1)
        TEMP[-1] = (1 - alpha) ** K
        self.temp = nn.Parameter(torch.tensor(TEMP))

        self.hesgat = HESGAT(in_feature, out_feature)

        self.mlp = nn.Sequential(
            nn.Linear(256, 128),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )
        self.fc = nn.Linear(62, 256)

        self.act_fn = nn.ReLU()


    def reset_parameters(self):
        torch.nn.init.zeros_(self.temp)
        for k in range(self.K + 1):
            self.temp.data[k] = self.alpha * (1 - self.alpha) ** k
        self.temp.data[-1] = (1 - self.alpha) ** self.K

    def forward(self, x, adj_sc, g, adj_ca):
        x = x.float()
        x = self.act_fn(self.fc(x))
        H = []
        H.append(x)
        hidden = x * (self.temp[0])
        for k in range(self.K):
            x = self.hesgat(x, adj_sc, g, H[0], adj_ca)
            gamma = self.temp[k + 1]
            hidden = hidden + gamma * x
        output = self.mlp(hidden)
        return output

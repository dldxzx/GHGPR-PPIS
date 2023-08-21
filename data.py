import pickle

import dgl
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


# codes = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
# protein_to_int = dict((c, i + 1) for i, c in enumerate(codes))
#
# MAP_TYPE = 'd'


def normalize(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = (rowsum ** -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = np.diag(r_inv)
    result = r_mat_inv @ mx @ r_mat_inv
    return result


def graph_collate(samples):
    sequence_name, label, node_features, adj_sc, G, adj_ca = map(list, zip(*samples))
    label = torch.Tensor(np.array(label))
    G_batch = dgl.batch(G)
    node_features = torch.cat(node_features)
    adj_sc = torch.Tensor(np.array(adj_sc))
    adj_ca = torch.Tensor(np.array(adj_ca))
    return sequence_name, label, node_features, adj_sc, G_batch, adj_ca


feature_path = './Feature/'


class Feature_Get():
    def __init__(self):
        self.path_map_sc = feature_path+'map_sc.pkl'
        self.f_map_sc = open(self.path_map_sc, 'rb')
        self.data_map_sc = pickle.load(self.f_map_sc)

        self.path_map_c = feature_path+'map_c.pkl'
        self.f_map_c = open(self.path_map_c, 'rb')
        self.data_map_c = pickle.load(self.f_map_c)

        self.path_map_ca = feature_path+'map.pkl'
        self.f_map_ca = open(self.path_map_ca, 'rb')
        self.data_map_ca = pickle.load(self.f_map_ca)


        self.path_pssm = feature_path+'pssm_741.pkl'
        self.f_pssm = open(self.path_pssm, 'rb')
        self.data_pssm = pickle.load(self.f_pssm)

        self.path_hmm = feature_path+'hmm_741.pkl'
        self.f_hmm = open(self.path_hmm, 'rb')
        self.data_hmm = pickle.load(self.f_hmm)

        self.path_dssp = feature_path+'dssp_741.pkl'
        self.f_dssp = open(self.path_dssp, 'rb')
        self.data_dssp = pickle.load(self.f_dssp)

        self.path_res_atom = feature_path+'res_atom_706.pkl'
        self.f_res_atom = open(self.path_res_atom, 'rb')
        self.data_res_atom = pickle.load(self.f_res_atom)

        self.embed_pos = feature_path+'pos_sc_706.pkl'
        self.f_embed_pos = open(self.embed_pos, 'rb')
        self.data_embed_pos = pickle.load(self.f_embed_pos)

    def load_pssm(self, sequence_name):
        feature = self.data_pssm[sequence_name]
        return feature.astype(np.float32)

    def load_hmm(self, sequence_name):
        feature = self.data_hmm[sequence_name]
        return feature.astype(np.float32)

    def load_dssp(self, sequence_name):
        feature = self.data_dssp[sequence_name]
        return feature.astype(np.float32)

    def load_res_atom(self, sequence_name):
        feature = self.data_res_atom[sequence_name]
        return feature.astype(np.float32)

    def load_pos(self, sequence_name):
        feature = self.data_embed_pos[sequence_name]
        return feature.astype(np.float32)

    def load_map(self, sequence_name, order):
        feature = self.data_map_sc[sequence_name]
        mask = ((feature >= 0) * (feature <= 14))
        adj = mask.astype(np.int)
        # adjacency_matrix = adjacency_matrix + np.eye(mask.shape[0])
        norm_matrix = normalize(adj.astype(np.float32))
        adj_real = adj
        # norm_matrix = adjacency_matrix.astype(np.float32)
        return norm_matrix, adj_real, feature

    def cal_edges(self, sequence_name, radius=14):  # to get the index of the edges
        feature = self.data_map_sc[sequence_name]
        mask = ((feature >= 0) * (feature <= radius))
        adjacency_matrix = mask.astype(np.int)
        radius_index_list = np.where(adjacency_matrix == 1)
        radius_index_list = [list(nodes) for nodes in radius_index_list]
        return radius_index_list

    def load_map_ca(self, sequence_name):
        feature = self.data_map_ca[sequence_name]
        mask = ((feature >= 0) * (feature <= 14))
        adj = mask.astype(np.int)
        norm_matrix = normalize(adj.astype(np.float32))
        return norm_matrix


class ProDatasetTrain(Dataset):
    def __init__(self, dataframe):
        self.names = dataframe['ID'].values
        self.site_labels = dataframe['label'].values
        self.feature_get = Feature_Get()
        self.dist = 15
        self.radius = 14

    def __getitem__(self, index):
        sequence_name = self.names[index]
        sequence_name = sequence_name[:4].lower() + sequence_name[4:5]
        site_label = np.array(self.site_labels[index]).astype(np.float32)
        nodes_num = site_label.shape[0]
        pssm = self.feature_get.load_pssm(sequence_name[:5])
        hmm = self.feature_get.load_hmm(sequence_name[:5])
        dssp = self.feature_get.load_dssp(sequence_name[:5])
        res_atom = self.feature_get.load_res_atom(sequence_name[:5])
        pos = self.feature_get.load_pos(sequence_name[:5])

        reference_res_psepos = pos[0]
        pos = pos - reference_res_psepos
        pos = torch.from_numpy(pos)

        node_feature = np.concatenate((pssm, hmm, dssp), axis=1)
        node_feature = np.concatenate((node_feature, res_atom), axis=1)
        node_feature = torch.from_numpy(node_feature)
        node_features = torch.cat(
            [node_feature, torch.sqrt(torch.sum(pos * pos, dim=1)).unsqueeze(-1) / self.dist], dim=-1)

        adj, adj_real, in_adj = self.feature_get.load_map(sequence_name[:5], 0)
        adj_ca = self.feature_get.load_map_ca(sequence_name[:5])

        radius_index_list = self.feature_get.cal_edges(sequence_name, 14)
        # print(len(radius_index_list[0]))

        edge_feat = self.cal_edge_attr(radius_index_list, pos)
        edge_feat = np.transpose(edge_feat, (1, 2, 0))
        edge_feat = edge_feat.squeeze(1)
        edge_feature = edge_feat

        G = dgl.DGLGraph()
        G.add_nodes(nodes_num)

        self.add_edges_custom(G,
                              radius_index_list,
                              edge_feature, adj
                              )

        return sequence_name, site_label, node_features, adj, G, adj_ca

    def __len__(self):
        return len(self.site_labels)

    def cal_edge_attr(self, index_list, pos):
        pdist = nn.PairwiseDistance(p=2, keepdim=True)
        cossim = nn.CosineSimilarity(dim=1)

        distance = (pdist(pos[index_list[0]], pos[index_list[1]]) / self.radius).detach().numpy()
        cos = ((cossim(pos[index_list[0]], pos[index_list[1]]).unsqueeze(-1) + 1) / 2).detach().numpy()
        radius_attr_list = np.array([distance, cos])
        return radius_attr_list

    def add_edges_custom(self, G, radius_index_list, edge_features, adj):
        src, dst = radius_index_list[1], radius_index_list[0]
        if len(src) != len(dst):
            print('source and destination array should have been of the same length: src and dst:', len(src), len(dst))
            raise Exception
        G.add_edges(src, dst)
        G.edata['ex'] = torch.tensor(edge_features)
        G.ndata['adj'] = torch.tensor(adj)

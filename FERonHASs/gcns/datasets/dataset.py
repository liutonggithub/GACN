# -*- coding: utf-8 -*-
import numpy as np

from utils import (read_meta, read_probs, l2norm, knns2ordered_nbrs,
                   intdict2ndarray, Timer,
                   build_knns,fast_knns2spmat,build_symmetric_adj,row_normalize,knns2spmat)

import os.path as osp
from tqdm import tqdm


class Dataset(object):
    def __init__(self, cfg):
        feat_path = cfg['feat_path']
        label_path = cfg.get('label_path', None)
        knn_graph_path = cfg['knn_graph_path']

        self.k_at_hop = cfg['k_at_hop']
        self.depth = len(self.k_at_hop)
        self.active_connection = cfg['active_connection']
        self.feature_dim = cfg['feature_dim']
        self.is_norm_feat = cfg.get('is_norm_feat', True)
        self.is_sort_knns = cfg.get('is_sort_knns', True)
        self.is_test = cfg.get('is_test', False)
        self.confs = np.load(cfg.pred_confs)['pred_confs']

        self.myhopsknn = './data'
        self.myhopsknn_k = 10

        with Timer('read meta and feature'):# idx2lb(每个样本对应的标签)
            if label_path is not None:
                _, self.idx2lb = read_meta(label_path)
                self.inst_num = len(self.idx2lb)
                self.labels = intdict2ndarray(self.idx2lb)
                self.ignore_label = False
            else:
                self.labels = None
                self.inst_num = -1
                self.ignore_label = True
            self.features = read_probs(feat_path, self.inst_num,
                                       self.feature_dim)
            if self.is_norm_feat:
                self.features = l2norm(self.features)
            if self.inst_num == -1:
                self.inst_num = self.features.shape[0]
            self.size = self.inst_num

        with Timer('read knn graph'):
            knns = np.load(knn_graph_path)['data']
            self.knn_graph_dists, self.knn_graph = knns2ordered_nbrs(knns, sort=self.is_sort_knns)
            # 为了构建HASs
            adj = fast_knns2spmat(knns, self.myhopsknn_k, 0, use_sim=True)
            # build symmetric adjacency matrix
            adj = build_symmetric_adj(adj, self_loop=True)
            self.adj = row_normalize(adj)

        assert np.mean(self.k_at_hop) >= self.active_connection

        print('feature shape: {}, norm_feat: {}, sort_knns: {} '
              'k_at_hop: {}, active_connection: {}'.format(
                  self.features.shape, self.is_norm_feat, self.is_sort_knns,
                  self.k_at_hop, self.active_connection))

    def __getitem__(self, index):
        '''
        return the vertex feature and the adjacent matrix, together
        with the indices of the center node and its 1-hop and 2-hop neighbors
        '''
        if index is None or index > self.size:
            raise ValueError('index({}) is not in the range of {}'.format(
                index, self.size))

        # 选择出大于置信度的节点
        center_node = index

        # hops[0] 保存 1-hop neighbors, hops[1] 保存 2-hop neighbors
        # nbr：center_node的邻居的索引 idxs:大于中心节点置信度的邻居节点索引
        nbr = self.knn_graph[center_node]
        idxs = np.where(self.confs[nbr] > self.confs[center_node])[0]
        hops = []
        hops.append(set(self.knn_graph[center_node][idxs]))

        for d in range(1, self.depth):
            hops.append(set())
            for h in hops[-2]:
                nbr_2 = self.knn_graph[h]
                idxs_2 = np.where(self.confs[nbr_2] > self.confs[h])[0]
                hops[-1].update(set(self.knn_graph[h][idxs_2]))

        self.hops_set = set([h for hop in hops for h in hop])

        self.hops_set.update([
            center_node,
        ])
        uniq_nodes = np.array(list(self.hops_set), dtype=np.int64)

        # uniq_nodes_map保存了所选节点的原始索引uniq_nodes和重构图中的新索引，
        # 预测时针对重构图中的新索引，其原始中心节点的索引对应为center_idx
        # 通过center_idx可以从重构图中找到原始节点在pred(重构图中的索引)
        # 求出了由大于中心节点置信度的1跳和2跳节点构成的重构图中每个节点的类别
        # pred（one_hop_labels中存的也是这个索引次序，即预测时是针对这个索引预测的）
        uniq_nodes_map = {j: i for i, j in enumerate(uniq_nodes)}

        # 通过中心节点的原始索引center_node找到重构图中的中心节点的新索引，
        # pred[center_idx]即为中心节点的预测标签
        center_idx = np.array([uniq_nodes_map[center_node]], dtype=np.int64)

        one_hop_idxs = np.array([uniq_nodes_map[i] for i in self.hops_set],
                                dtype=np.int64)

        #根据1跳邻居点和2跳邻居构造HASs
        nbr = self.knn_graph[uniq_nodes]
        dist = self.knn_graph_dists[uniq_nodes]
        center_feat = self.features[center_node]
        feat = self.features[uniq_nodes]

        knn_prefix = osp.join(self.myhopsknn, 'knns', 'myhops_knn')
        knns = build_knns(knn_prefix, feat, 'faiss', self.myhopsknn_k)
        adj = fast_knns2spmat(knns, self.myhopsknn_k, 0, use_sim=True)

        # build symmetric adjacency matrix
        adj = build_symmetric_adj(adj, self_loop=True)
        adj = row_normalize(adj)
        A=adj.A
        A = A.astype(np.float32)

        # num_nodes = len(uniq_nodes)
        #A = np.zeros([num_nodes, num_nodes], dtype=feat.dtype)
        # print("调试。。。。。")

        max_num_nodes = self.k_at_hop[0] * (self.k_at_hop[1] + 1) + 1
        num_nodes = len(uniq_nodes)
        res_num_nodes = max_num_nodes - num_nodes
        if res_num_nodes > 0:
            pad_feat = np.zeros([res_num_nodes, self.feature_dim],
                                dtype=feat.dtype)
            feat = np.concatenate([feat, pad_feat], axis=0)

        # 根据1跳邻居和2跳邻居生成HASs和邻接矩阵
        A_ = np.zeros([max_num_nodes, max_num_nodes], dtype=A.dtype)
        l=len(A[0])
        r = len(A[1])
        A_[:l, :r] = A

        if self.ignore_label:
            return (feat, A, center_idx, one_hop_idxs)

        labels = self.labels[uniq_nodes]
        one_hop_labels = labels[one_hop_idxs]
        one_hop_labels = one_hop_labels.astype(np.int64)
        center_label = labels[center_idx]
        edge_labels = (center_label == one_hop_labels).astype(np.int64)

        if self.is_test:
            if res_num_nodes > 0:
                pad_nodes = np.zeros(res_num_nodes, dtype=uniq_nodes.dtype)
                uniq_nodes = np.concatenate([uniq_nodes, pad_nodes], axis=0)
            return (feat, A_, one_hop_idxs,
                    one_hop_labels), center_idx, uniq_nodes
        else:
            return (feat, A_, one_hop_idxs, one_hop_labels)

    def __len__(self):
        return self.size

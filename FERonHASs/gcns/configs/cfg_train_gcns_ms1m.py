# -*- coding: utf-8 -*-
import os.path as osp

# data locations
prefix = './data'
train_name = 'part0_train'
test_name = 'part1_test'

knn = 80
knn_method = 'faiss'

train_data = dict(
    feat_path=osp.join(prefix, 'features', '{}.bin'.format(train_name)),
    label_path=osp.join(prefix, 'labels', '{}.meta'.format(train_name)),
    knn_graph_path=osp.join(prefix, 'knns', train_name,
                            '{}_k_{}.npz'.format(knn_method, knn)), #读取data/knns/part0_train/faiss_k_80.npz
    k_at_hop=[150, 20],
    active_connection=10,
    is_norm_feat=True,
    is_sort_knns=True,
    pred_confs=osp.join(prefix, 'pred_confs', '{}.npz'.format(train_name)),
    data_name=train_name
)

test_data = dict(
    feat_path=osp.join(prefix, 'features', '{}.bin'.format(test_name)),
    label_path=osp.join(prefix, 'labels', '{}.meta'.format(test_name)),
    knn_graph_path=osp.join(prefix, 'knns', test_name,
                            '{}_k_{}.npz'.format(knn_method, knn)),
    k_at_hop=[150, 20],
    active_connection=10,
    is_norm_feat=True,
    is_sort_knns=True,
    is_test=True,
    pred_confs=osp.join(prefix, 'pred_confs', '{}.npz'.format(test_name)),
    data_name=test_name
)

# model
# model = dict(type='gcns', kwargs=dict(feature_dim=256))
# model = dict(type='gcns', kwargs=dict(feature_dim=468))
model = dict(type='gcns', kwargs=dict(feature_dim=512))#利用置信度网络提取的特征

# training args
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=1e-4)
optimizer_config = {}

lr_config = dict(
    policy='step',
    step=[1, 2, 3],
)

# batch_size_per_gpu = 16
batch_size_per_gpu = 1
total_epochs = 2
workflow = [('train', 1)]

# testing args
max_sz = 300
step = 0.6
pool = 'avg'

metrics = ['pairwise', 'bcubed', 'nmi']

# misc
workers_per_gpu = 1

checkpoint_config = dict(interval=1)

log_level = 'INFO'
log_config = dict(interval=200, hooks=[
    dict(type='TextLoggerHook'),
])

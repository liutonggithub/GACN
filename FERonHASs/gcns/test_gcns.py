# -*- coding: utf-8 -*-
from __future__ import division

import os
import torch
import numpy as np
import torch.nn.functional as F

from mmcv.runner import load_checkpoint
from mmcv.parallel import MMDataParallel

from gcns.datasets import build_dataset, build_dataloader
from gcns.online_evaluation import online_evaluate

from utils import (clusters2labels, intdict2ndarray, get_cluster_idxs,
                   write_meta)
from proposals.graph import graph_clustering_dynamic_th
from evaluation import evaluate

import numpy as np
from evaluation import precision, recall, accuracy


def test(model, dataset, cfg, logger):
    if cfg.load_from:
        print('load from {}'.format(cfg.load_from))
        load_checkpoint(model, cfg.load_from, strict=True, logger=logger)

    losses = []
    edges = []
    scores = []
    class_pre_mysave=[]

    if cfg.gpus == 1:
        data_loader = build_dataloader(dataset,
                                       cfg.batch_size_per_gpu,
                                       cfg.workers_per_gpu,
                                       train=False)

        model = MMDataParallel(model, device_ids=range(cfg.gpus))
        if cfg.cuda:
            model.cuda()

        model.eval()
        for i, (data, cid, node_list) in enumerate(data_loader):
            with torch.no_grad():#gtmat：(求准确率时)
                _, _, h1id, gtmat = data
                pred, loss = model(data, return_loss=True)#pred：预测值(求准确率时)
                losses += [loss.item()]
                pred = F.softmax(pred, dim=1)
                if i % cfg.log_config.interval == 0:
                    if dataset.ignore_label:
                        logger.info('[Test] Iter {}/{}'.format(
                            i, len(data_loader)))
                    else:
                        acc, p, r = online_evaluate(gtmat, pred)
                        logger.info(
                            '[Test] Iter {}/{}: Loss {:.4f}, '
                            'Accuracy {:.4f}, Precision {:.4f}, Recall {:.4f}'.
                            format(i, len(data_loader), loss, acc, p, r))

                # 此处中心节点的索引对应所有所选预测节点中的中心节点索引，
                # 找到预测的中心节点所在位置，从而取出中心节点的预测值
                class_pre = torch.argmax(pred.cpu(), dim=1).long()
                class_pre_mysave.append(class_pre[cid].int().item())#将张量类型保存为数组类型，用.int().item()

                node_list = node_list.numpy()
                bs = len(cid)
                h1id_num = len(h1id[0])
                for b in range(bs):
                    cidb = cid[b].int().item()

                    nlst = node_list[b]
                    center_idx = nlst[cidb]
                    for j, n in enumerate(h1id[b]):
                        edges.append([center_idx, nlst[n.item()]])
                        scores.append(pred[b * h1id_num + j, 1].item())

    else:
        raise NotImplementedError

    #保存预测的中心节点标签
    # print(class_pre_mysave)
    class_pre_mysave=np.array(class_pre_mysave)
    np.savetxt('./pred_labels_my.txt',
               class_pre_mysave,
               delimiter=' ', fmt='%d')


    if not dataset.ignore_label:
        avg_loss = sum(losses) / len(losses)
        logger.info('[Test] Overall Loss {:.4f}'.format(avg_loss))

    return np.array(edges), np.array(scores),np.array(class_pre_mysave), len(dataset)


def test_gcns(model, cfg, logger):
    for k, v in cfg.model['kwargs'].items():
        setattr(cfg.test_data, k, v)
    dataset = build_dataset(cfg.test_data)

    if not dataset.ignore_label:
        print('==> evaluation')
        gt_labels = dataset.labels
        # 保存真实的标签
        # print(gt_labels)
        gt_labels_save=gt_labels
        np.savetxt('./gt_labels_my.txt', gt_labels_save,
                   delimiter=' ',fmt='%d')

        # 评价指标
        acc_my=accuracy(gt_labels, class_pre_mysave)
        p_my=precision(gt_labels, class_pre_mysave)
        r_my = recall(gt_labels, class_pre_mysave)
        logger.info(
            '[Test] {}, '
            'Accuracy {:.4f}, Precision {:.4f}, Recall {:.4f}'.
                format(inst_num, acc_my, p_my, r_my))
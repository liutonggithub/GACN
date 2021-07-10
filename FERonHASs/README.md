# FER

## Requirements
* Python >= 3.6
* PyTorch >= 0.4.0
* [faiss](https://github.com/facebookresearch/faiss)
* [mmcv](https://github.com/open-mmlab/mmcv)

Install dependencies
```bash
conda install faiss-gpu -c pytorch
pip install -r requirements.txt
```

## Test
Test GCN-A
```bash
sh scripts/gcna/test_gcn_a_multipie.sh
```
Test GCN-S
```bash
sh scripts/gcna/test_gcn_s_multipie.sh
```
## Train
Train GCN-A
```bash
sh scripts/gcna/train_gcn_a_multipie.sh
```
Train GCN-S
```bash
sh scripts/gcns/train_gcn_s_multipie.sh
```

## Face Recognition

For training face recognition and feature extraction, you may use framework below:

https://github.com/deepinsight/insightface



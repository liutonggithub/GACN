cfg_name=cfg_train_gcns_ms1m
config=gcns/configs/$cfg_name.py

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

# train
python gcns/main.py \
    --config $config \
    --phase 'train'



config=gcna/configs/cfg_test_gcna_multipie.py
load_from=data/pretrained_models/pretrained_gcna_multipie.pth

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=.

python gcna/main.py \
    --config $config \
    --phase 'test' \
    --load_from $load_from \
    --save_output \
    --force

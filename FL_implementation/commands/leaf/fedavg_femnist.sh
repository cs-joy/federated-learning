#!/bin/sh

# FedAvg experiments for LEAF FEMNIST Dataset
python 3 main.py \
    --exp_name FEMNIST --seed 42 --device cpu \
    --dataset FEMNIST \
    --split_type pre --test_size 0.1 \
    --model_name FEMNISTCNN --resize 28 --hidden_size 64 \
    --algorithm fedavg --eval_function 1 --eval_type local --eval_every 50 --eval_metrics acc1 \
    --R 5000 --E 5 --C 0.003 --B 10 --beta1 0 \
    --optimizer SGD --lr 0.0003 --lr_deacy 1 --lr_decay_step 1 --criterion CrossEntropyLoss
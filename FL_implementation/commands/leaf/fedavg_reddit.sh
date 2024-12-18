#!/bin/sh

# FedAvg experiments for LEAF Reddit dataset
python3 main.py \
    --exp_name FedAvg_LEAF_Reddit --seed 42 --device cpu \
    --dataset Reddit \
    --split_type pre --test_size 0.1 \
    --model_name NextWordLSTM --num_layers 2 --num_embeddings 10000 --embedding_size 256 --hidden_sizee 256 \
    --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 50 --eval_metrics seqacc \
    --optimizer SGD --lr 0.0003 --lr_decay 1 --lr_decay_step 1 --criterion Seq2SeqLoss
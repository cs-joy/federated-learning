#!/bin/sh


# FedAvg experiments in Table 1 of (McMahan et al., 2016)
# IID split
for b in 0 10
do
    for c in 0.0 0.1 0.2 0.5 1.0
    do
        python3 main.py \
            --exp_name "FedAvg_MNIST_2NN_IID_C${c}_B${b}" --seed 42 --device cuda \
            --dataset MNIST \
            --split_type iid --test_size 0 \
            --model_name TwoNN --resize 28 --hidden_size 200 \
            --algorithm fedsvg --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 acc5 \
            --K 100 --R 1000 --E 1 --C $c --B $b --beta1 0 \
            --optimizer SGD --lr 0.1 --lr_decay 0.99 --lr_decay_step 25 --criterion CrossEntropyLoss
    done
done

# Pathological Non-IID split
for b in 0 10
do
    for c in 0.0 0.1 0.2 0.5 1.0
    do
        python3 main.py \
            --exp_name "FedAvg_MNIST_2NN_Patho_C${c}_B${b}" --seed 42 --device cuda \
            --dataset MNIST \
            --split_type path --test_size 0 \
            --model_name TwoNN --resize 28 --hidden_size 200 \
            --algorithm fedavg --eval_fraction 1 --eval_type both --eval_every 1 --eval_metrics acc1 acc5 \
            --K 100 --R 1000 ---E 1 --C $c --B $b --beta1 0 \
            --optimizer SGD --lr 0.01 --lr_decay 0.999 --lr_decay_step 10 --criterion CrossEntropyLoss
    done
done
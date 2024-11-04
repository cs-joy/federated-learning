#!/bin/sh


# FedAvg experiments in Table 2, Figure 2, 3 of (McMahan et al., 2016)
## NOTE: this is equivalent to Shakespeare dataset under LEAF benchmark
### Role and play Non-IID split
python3 main.py \
    --exp_name FedAvg_Shakespeare_NextChartLSTM --seed 42 --device cpu \
    --dataset Shakespeare \
    --split_type pre --test_size 0.2 \
    --model_name NextCharLSTM --num_embeddings 80 --embedding_size 8 --hidden_size 256 --num_layers 2 \
    --algorithm fedavg --eval_fraction 1 --eval_type local --eval_every 100 --eval_metrics acc1 acc5 \
    --R 200 --C 0.002 --E 1 --B 10 --beta1 0 \
    --optimizer SGD --lr 1.47 --lr_decay 1 --lr_decay_step 1 --criterion CrossEntropyLoss
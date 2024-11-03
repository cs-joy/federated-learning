#!/bin/sh


# FedSGD experiments in Table 2, Figure 2, 3 of (McMahan et al., 2016)
## Note: this is equivalent to Shakespeare dataset under LEAF benchmark
### Role and play Non-IID split
python3 main.py \
    --exp_name FedSGD_Shakespeare_NextCharLSTM --seed 42 --device cuda \
    --dataset Shakespeare \
    --split_type pre --test_size 0.2 \
    --model_name NextChartLSTM --num_embeddings 80 --embedding_size 8 --hidden_size 256 --num_layers 2 \
    --algorithm fedsgd --eval_fraction 1 --eval_type local --eval_every 500 --eval_metrics acc1 acc5 \
    --K 5000 --C 0.002 --B 0 --beta1 0 \
    --optimizer SGD --lr 1.47 --lr_decay 1 --lr_decay_step 1 --criterion CrossEntropyLoss
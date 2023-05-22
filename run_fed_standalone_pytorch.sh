#!/usr/bin/env bash

GPU=0

CLIENT_NUM=20

WORKER_NUM=8

BATCH_SIZE=32

ALPHA=1

DATASET=fashion_mnist
DISTILLATION_DATASET=mnist

MODEL=DaFKD

DISTRIBUTION=hetero

ROUND=61

CLIENT_OPTIMIZER=adam

EPOCH=20

GAN_EPOCH=5

D_EPOCH=1

ED_EPOCH=40
NOISE_DIMENSION=100

TEMPERATURE=20.0
LR=0.0001
ES_LR=0.0001

AGGREGATION=FedDF
CI=0
AGGREGATION="FedDF"

BASELINE="DaFKD"



python3 main_fed.py \
--gpu $GPU \
--dataset $DATASET \
--model $MODEL \
--partition_method $DISTRIBUTION  \
--alpha $ALPHA  \
--aggregation_method $AGGREGATION  \
--client_num_in_total $CLIENT_NUM \
--client_num_per_round $WORKER_NUM \
--comm_round $ROUND \
--epochs $EPOCH \
--client_optimizer $CLIENT_OPTIMIZER \
--batch_size $BATCH_SIZE \
--d_epoch $D_EPOCH \
--ed_epoch $ED_EPOCH \
--gan_epoch $GAN_EPOCH \
--noise_dimension $NOISE_DIMENSION \
--temperature $TEMPERATURE \
--lr $LR \
--es_lr $ES_LR \
--ci $CI \
--baseline $BASELINE \
--distillation_dataset $DISTILLATION_DATASET

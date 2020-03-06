#!/bin/bash

source ./path.sh

task_slot_filling=NN # NN, NN_crf
task_intent_detection=CLS # CLS, max, CLS_max
balance_weight=0.5

pretrained_model_type=bert
pretrained_model_name=bert-base-uncased #bert-large-uncased-whole-word-masking #bert-base-uncased

dataroot=data/atis-2
dataset=atis

batch_size=32 # 16, 32

optimizer=bertadam #bertadam, adamw, adam, sgd
learning_rate=5e-5 # 1e-5, 5e-5, 1e-4, 1e-3
max_norm_of_gradient_clip=1 # working for adamw, adam, sgd
dropout_rate=0.1 # 0.1, 0.5

max_epoch=30

device=0
# device=0 means auto-choosing a GPU
# Set deviceId=-1 if you are going to use cpu for training.
experiment_output_path=exp

source ./utils/parse_options.sh

python scripts/slot_tagging_and_intent_detection_with_pure_transformer.py --task_st $task_slot_filling --task_sc $task_intent_detection --dataset $dataset --dataroot $dataroot --lr $learning_rate --dropout $dropout_rate --batchSize $batch_size --optim $optimizer --max_norm $max_norm_of_gradient_clip --experiment $experiment_output_path --deviceId $device --max_epoch $max_epoch --st_weight ${balance_weight} --pretrained_model_type ${pretrained_model_type} --pretrained_model_name ${pretrained_model_name}

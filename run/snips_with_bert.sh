#!/bin/bash

source ./path.sh

task_slot_filling=$1 #slot_tagger, slot_tagger_with_crf, slot_tagger_with_focus
task_intent_detection=hiddenAttention # none, hiddenAttention, hiddenCNN, maxPooling, 2tails
balance_weight=0.5

bert_model_name=bert-base-uncased

dataroot=data/snips
dataset=snips

lstm_hidden_size=200 # 100, 200
lstm_layers=1
slot_tag_embedding_size=100  ## for slot_tagger_with_focus
batch_size=32 # 16, 32

#optimizer=adam
learning_rate=1e-5 # 1e-5, 5e-5, 1e-4, 1e-3
#max_norm_of_gradient_clip=5
dropout_rate=0.1 # 0.1, 0.5

max_epoch=30

device=0
# device=0 means auto-choosing a GPU
# Set deviceId=-1 if you are going to use cpu for training.
experiment_output_path=exp

python scripts/slot_tagging_and_intent_detection_with_bert.py --task_st $task_slot_filling --task_sc $task_intent_detection --dataset $dataset --dataroot $dataroot --bidirectional --lr $learning_rate --dropout $dropout_rate --batchSize $batch_size --experiment $experiment_output_path --deviceId $device --max_epoch $max_epoch --hidden_size $lstm_hidden_size --num_layers ${lstm_layers} --tag_emb_size $slot_tag_embedding_size --st_weight ${balance_weight} --bert_model_name ${bert_model_name}

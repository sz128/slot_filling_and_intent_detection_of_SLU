#!/bin/bash

source ~/ve3_cu80_pt4/bin/activate

task=slot_tagger_with_focus #slot_tagger, slot_tagger_with_crf, slot_tagger_with_focus
dataroot=data/atis-2.raw_DIGIT.multi_intents
dataset=atis-2.raw_DIGIT.multi_intents

es=128
hs=128

opt=adam
lr=0.001
mn=5
dp=0.5
bs=16

me=100
mwf=1 #If you want to replace rare word with <UNK>, please set $mwf large than 1.

# Set deviceId=-1 if you are going to use cpu for training.

python3 scripts/slot_tagging.py --task $task --dataset $dataset --dataroot $dataroot --bidirectional --lr $lr --dropout $dp --batchSize $bs --optim $opt --max_norm $mn --experiment exp --deviceId 0 --max_epoch $me --emb_size $es --hidden_size $hs --mini_word_freq $mwf 

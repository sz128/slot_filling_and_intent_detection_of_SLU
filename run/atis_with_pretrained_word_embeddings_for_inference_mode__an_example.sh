#!/bin/bash

source ./path.sh

task_slot_filling=slot_tagger #slot_tagger, slot_tagger_with_crf, slot_tagger_with_focus
task_intent_detection=hiddenAttention # none, hiddenAttention, hiddenCNN, maxPooling, 2tails
balance_weight=0.5

#cased_word_vectors=./local/word_embeddings/elmo_1024_cased_for_atis.txt
cased_word_vectors=./local/word_embeddings/glove-kazumachar_400_cased_for_atis.txt
read_word2vec_inText=${cased_word_vectors}
word_lowercase=false
fix_word2vec_inText=false
word_digit_features=false #false, true

dataroot=data/atis-2
dataset=atis

word_embedding_size=400 #1024
lstm_hidden_size=200
lstm_layers=1
slot_tag_embedding_size=100  ## for slot_tagger_with_focus
batch_size=20

#optimizer=adam
#learning_rate=0.001
#max_norm_of_gradient_clip=5
#dropout_rate=0.5

#max_epoch=50

device=0
# device=0 means auto-choosing a GPU
# Set deviceId=-1 if you are going to use cpu for training.
#experiment_output_path=exp

if [[ $word_lowercase != true && $word_lowercase != True ]]; then
  unset word_lowercase
fi
if [[ $fix_word2vec_inText != true && $fix_word2vec_inText != True ]]; then
  unset fix_word2vec_inText
fi
if [[ $word_digit_features != true && $word_digit_features != True ]]; then
  unset word_digit_features
fi

saved_model_dir=exp/model_slot_tagger__and__hiddenAttention__and__single_cls_CE/data_atis/bidir_True__emb_dim_400__hid_dim_200_x_1__bs_20__dropout_0.5__optimizer_adam__lr_0.001__mn_5.0__me_50__tes_100__alpha_0.5__preEmb_in
#BEST RESULT: 	Epoch : 21	best valid (Loss: (0.05, 0.15) F1 : 98.10 cls-F1 : 98.40)	best test (Loss: (0.18, 0.15) F1 : 95.55 cls-F1 : 98.66) 

python scripts/slot_tagging_and_intent_detection.py --task_st $task_slot_filling --task_sc $task_intent_detection --dataset $dataset --dataroot $dataroot --bidirectional --batchSize $batch_size --deviceId $device --emb_size $word_embedding_size --hidden_size $lstm_hidden_size --num_layers ${lstm_layers} --tag_emb_size $slot_tag_embedding_size --st_weight ${balance_weight} ${word_lowercase:+--word_lowercase} ${read_word2vec_inText:+--read_input_word2vec ${read_word2vec_inText}} ${fix_word2vec_inText:+--fix_input_word2vec} ${word_digit_features:+--word_digit_features} --testing --read_model ${saved_model_dir}/model --read_vocab ${saved_model_dir}/vocab --out_path ${saved_model_dir}
